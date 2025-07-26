# DEBUG: Log config paths at startup
import os
import logging
import sys # Required for sys.path manipulation
# --- Setup Logging ---
# Placed logging config once at the top for clarity
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

logging.info(f"DEBUG: Current working directory: {os.getcwd()}")
from f5_tts.config import settings
logging.info(f"DEBUG: DATA_PATH resolved to: {settings.DATA_PATH} (type: {type(settings.DATA_PATH)})")
logging.info(f"DEBUG: vocab_path resolved to: {settings.vocab_path} (type: {type(settings.vocab_path)})")
from pathlib import Path # Ensure Path is imported early for sys.path

# This block ensures that the 'src' directory (parent of 'f5_tts' package)
# is in sys.path when the script is run directly. This allows absolute imports
# like 'from f5_tts.config import settings' to work in that scenario.
_current_file_path = Path(__file__).resolve()
_src_path = _current_file_path.parent.parent # F5-TTS/src/f5_tts -> F5-TTS/src
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import shutil
import tempfile
import random
import csv
import re ### NEW/MODIFIED ###: Import regular expressions for number finding
from contextlib import asynccontextmanager
import asyncio # Add asyncio for locking
from typing import Optional, Tuple, List, Dict, Any

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from omegaconf import OmegaConf
from hydra.utils import get_class
from importlib.resources import files
import queue
import threading
import torchaudio
import wave

from f5_tts.infer.utils_infer import (
    chunk_text,
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)


### NEW/MODIFIED ###: Import the number-to-word conversion library
logger.info(f"sys.path before num2fawords import: {sys.path}")
# You must install this library first: pip install num2fawords
try:
    from num2fawords import words as num2fawords
    num2fawords_available = True
    logger.info("Successfully imported num2fawords library (words function).")
except ImportError:
    num2fawords_available = False
    logger.warning("Could not import 'num2fawords'. Number-to-word conversion will be disabled.")
    logger.warning("Please install it using: pip install num2fawords")


from f5_tts.api import F5TTS
from f5_tts.config import settings # Changed back to absolute import

# --- Global State (Reverted from AppState/app.state) ---
tts_api: Optional[F5TTS] = None
preselected_voices: List[Dict[str, Any]] = [] # Renamed from PRESELECTED_VOICES for consistency
streaming_processor: Optional["TTSStreamingProcessor"] = None
# CRITICAL CHANGE: Use a threading.Lock for multi-thread safety
inference_lock = threading.Lock()


# --- Streaming Classes (from socket_server.py) ---

class AudioFileWriterThread(threading.Thread):
    """Threaded file writer to avoid blocking the TTS streaming process."""

    def __init__(self, output_file, sampling_rate):
        super().__init__()
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.audio_data = []

    def run(self):
        """Process queued audio data and write it to a file."""
        logger.info("AudioFileWriterThread started.")
        with wave.open(self.output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sampling_rate)

            while not self.stop_event.is_set() or not self.queue.empty():
                try:
                    chunk = self.queue.get(timeout=0.1)
                    if chunk is not None:
                        chunk = np.int16(chunk * 32767)
                        self.audio_data.append(chunk)
                        wf.writeframes(chunk.tobytes())
                except queue.Empty:
                    continue

    def add_chunk(self, chunk):
        """Add a new chunk to the queue."""
        self.queue.put(chunk)

    def stop(self):
        """Stop writing and ensure all queued data is written."""
        self.stop_event.set()
        self.join()
        logger.info("Audio writing completed.")


class TTSStreamingProcessor:
    def __init__(self, model, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "xpu"
            if torch.xpu.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        self.model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model_arc = model_cfg.model.arch
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
        self.sampling_rate = model_cfg.model.mel_spec.target_sample_rate

        self.model = self.load_ema_model(ckpt_file, vocab_file, dtype)
        self.vocoder = self.load_vocoder_model()

        self.update_reference(ref_audio, ref_text)
        self._warm_up()
        self.file_writer_thread = None
        self.first_package = True

    def load_ema_model(self, ckpt_file, vocab_file, dtype):
        return load_model(
            self.model_cls,
            self.model_arc,
            ckpt_path=ckpt_file,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

    def load_vocoder_model(self):
        return load_vocoder(vocoder_name=self.mel_spec_type, is_local=False, local_path=None, device=self.device)

    def update_reference(self, ref_audio, ref_text):
        self.ref_audio, self.ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
        self.audio, self.sr = torchaudio.load(self.ref_audio)

        ref_audio_duration = self.audio.shape[-1] / self.sr
        ref_text_byte_len = len(self.ref_text.encode("utf-8"))
        self.max_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration))
        self.few_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 2)
        self.min_chars = int(ref_text_byte_len / (ref_audio_duration) * (25 - ref_audio_duration) / 4)

    def _warm_up(self):
        logger.info("Warming up the model...")
        gen_text = "Warm-up text for the model."
        for _ in infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            [gen_text],
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
        ):
            pass
        logger.info("Warm-up completed.")

    def generate_stream(self, text: str):
        """
        A generator that yields audio chunks as bytes.
        This version is decoupled from the network layer.
        """
        # --- Text chunking logic is the same ---
        text_batches = chunk_text(text, max_chars=self.max_chars)
        if self.first_package:
            text_batches = chunk_text(text_batches[0], max_chars=self.few_chars) + text_batches[1:]
            text_batches = chunk_text(text_batches[0], max_chars=self.min_chars) + text_batches[1:]
            self.first_package = False # Note: You'll need to manage this state per-request

        audio_stream = infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            text_batches,
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
            chunk_size=2048, # Or another suitable chunk size
        )

        logger.info("Starting to yield audio chunks from generator.")
        for audio_chunk, _ in audio_stream:
            if len(audio_chunk) > 0:
                # Convert numpy float array to 16-bit PCM bytes
                chunk_bytes = np.int16(audio_chunk * 32767).tobytes()
                yield chunk_bytes
        
        logger.info("Finished yielding audio chunks.")

# --- Helper Functions ---

### NEW/MODIFIED ###: Start of new functions for number conversion
def _replace_number_with_words(match: "re.Match[str]") -> str:
    """
    Takes a regex match object, converts the number to an integer
    (handling Persian and Arabic numerals), and returns the word form.
    """
    number_str = match.group(0)
    # Translation table for Persian/Arabic numerals to Western numerals
    persian_to_english_map = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    english_num_str = number_str.translate(persian_to_english_map)

    try:
        number_int = int(english_num_str)
        # Convert the integer to Persian words using the imported library
        return num2fawords(number_int)
    except (ValueError, NameError): # NameError if num2fawords is not imported
        # If conversion fails, return the original number string
        return number_str

def convert_numbers_to_persian_words(text: str) -> str:
    """
    Finds all numbers in a string and converts them to their Persian
    written equivalent. Handles both '8' and '۸'.
    Example: "قیمت 35 هزار تومان است" -> "قیمت سی و پنج هزار تومان است"
    """
    if not num2fawords_available:
        logger.warning("Skipping number-to-word conversion because 'num2fawords' is not available.")
        return text

    # Regex to find sequences of Western or Persian/Arabic digits
    number_pattern = re.compile(r'[\d۰-۹]+')
    return number_pattern.sub(_replace_number_with_words, text)
### NEW/MODIFIED ###: End of new functions for number conversion

def _cleanup_temp_file(path: str):
    """Safely remove a temporary file."""
    try:
        if path and Path(path).exists():
            Path(path).unlink() # Using Path.unlink for consistency
            logger.info(f"Cleaned up temp file: {path}")
    except Exception as e:
        logger.error(f"Error cleaning up temp file {path}: {e}", exc_info=True)

def _get_random_reference() -> Tuple[Optional[str], Optional[str]]:
    """Selects a random reference audio file and its transcript."""
    if not settings.metadata_path.is_file():
        logger.warning(f"Metadata file not found for random reference: {settings.metadata_path}")
        return None, None

    try:
        with open(settings.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            lines = [line for line in reader if len(line) >= 2]
        
        if not lines:
            logger.warning(f"Metadata file {settings.metadata_path} is empty or has no valid lines.")
            return None, None

        audio_file_name, text_transcript = random.choice(lines)
        if not audio_file_name.lower().endswith(".wav"):
            audio_file_name += ".wav"

        audio_file_path = settings.wavs_path / audio_file_name
        if not audio_file_path.is_file():
            logger.warning(f"Selected random reference audio file not found: {audio_file_path}")
            return None, None

        return str(audio_file_path), text_transcript
    except Exception as e:
        logger.error(f"Error getting random reference: {e}")
        return None, None

def _pad_short_text(text: str) -> str:
    """Pads very short text to prevent potential model artifacts."""
    words = text.strip().split()
    if 0 < len(words) < settings.SHORT_TEXT_THRESHOLD_WORDS:
        padding = settings.SHORT_TEXT_PADDING
        padded_text = f"{padding} {text.strip()} {padding}"
        logger.info(f"Short text detected. Padding to: '{padded_text}'")
        return padded_text
    return text

# --- Startup Logic ---

def _perform_startup_checks() -> bool:
    """Performs critical checks before attempting to load the model. Returns True if all checks pass."""
    logger.info("--- Running Startup Configuration Checks ---")
    # Simplified check_map from an earlier version
    check_map = {
        "Checkpoint File": settings.CHECKPOINT_FILE,
        "Data Path": settings.DATA_PATH,
        "Project Vocab File": settings.vocab_path,
    }
    
    all_ok = True
    for name, path_obj in check_map.items():
        path_to_check = Path(path_obj) # Ensure it's a Path object
        is_file_check = "File" in name or "Vocab" in name # Heuristic for file vs dir
        
        if is_file_check:
            if path_to_check.is_file():
                logger.info(f"✅ [OK]   {name}: {path_to_check}")
            else:
                logger.error(f"❌ [FAIL] {name} check failed for path: {path_to_check}")
                all_ok = False
        else: # Assumed to be a directory check
            if path_to_check.is_dir():
                logger.info(f"✅ [OK]   {name}: {path_to_check}")
            else:
                logger.error(f"❌ [FAIL] {name} check failed for path: {path_to_check}")
                all_ok = False
            
    if settings.DEVICE == "cuda":
        if torch.cuda.is_available():
            logger.info("✅ [OK]   CUDA Available")
        else:
            logger.error("❌ [FAIL] CUDA is not available, but DEVICE is set to 'cuda'")
            all_ok = False
    else:
        logger.info(f"ℹ️ [SKIP] CUDA check (device is '{settings.DEVICE}')")

    return all_ok

def _load_preset_voices():
    """Loads a deterministic, evenly-spaced subset of voices from metadata."""
    global preselected_voices
    preselected_voices = [] # Clear previous, if any
    if not settings.metadata_path.is_file():
        logger.warning(f"Cannot load preset voices: metadata file not found at {settings.metadata_path}")
        return

    try:
        with open(settings.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            all_entries = sorted(
                [(line[0], line[1]) for line in reader if len(line) >= 2],
                key=lambda x: x[0]
            )

        if not all_entries:
            logger.warning("No valid entries in metadata file for preset voices.")
            return

        num_to_select = min(len(all_entries), settings.PRESET_VOICE_COUNT)
        if num_to_select > 0:
            indices = [int(i * (len(all_entries) - 1) / (num_to_select - 1)) for i in range(num_to_select)] if num_to_select > 1 else [0]
            
            for i, entry_idx in enumerate(indices):
                audio_filename, transcript = all_entries[entry_idx]
                if not audio_filename.lower().endswith(".wav"):
                    audio_filename += ".wav"
                
                full_audio_path = settings.wavs_path / audio_filename
                if full_audio_path.is_file():
                    preselected_voices.append({
                        "id": f"preset_{i}", # Reverted ID format
                        "name": Path(audio_filename).stem, # Keep stem as name for some user-friendliness
                        "text": transcript,
                        "audio_path": str(full_audio_path) # Internal path
                    })
                else:
                    logger.warning(f"Audio file for preset '{audio_filename}' not found. Skipping.")
        logger.info(f"Successfully loaded {len(preselected_voices)} pre-selected voices.")

    except Exception as e:
        logger.error(f"Failed to load pre-selected voices: {e}", exc_info=True)
        preselected_voices = []

def _initialize_tts_model() -> Optional[F5TTS]:
    """Initializes and returns the F5TTS instance."""
    logger.info("--- Attempting TTS API Initialization ---")
    try:
        model = F5TTS(
            model=settings.EXP_NAME,
            ckpt_file=str(settings.CHECKPOINT_FILE),
            vocab_file=str(settings.vocab_path),
            device=settings.DEVICE,
            use_ema=settings.USE_EMA,
        )
        logger.info("✅ [OK] TTS API initialized successfully.")
        return model
    except Exception as e:
        logger.critical(f"❌ [FATAL] TTS API initialization failed: {e}", exc_info=True)
        return None

def _run_startup_test(model: F5TTS):
    """Performs a quick synthesis to ensure the model is working."""
    logger.info("--- Performing Startup Test Synthesis ---")
    # Using a fixed filename for the startup test output, as in earlier versions
    output_file_test = Path("startup_test_output.wav")
    ref_audio_path, ref_text = _get_random_reference()
    
    if not (ref_audio_path and ref_text):
        logger.warning("Could not get a random reference for startup test. Test may be less representative.")

    try:
        model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text.lower().strip() if ref_text else None,
            # Using a generic test text, as STARTUP_TEST_TEXT might be removed from config
            gen_text="This is a startup test audio.",
            nfe_step=10, 
            speed=settings.SPEED,
            remove_silence=settings.REMOVE_SILENCE,
            file_wave=str(output_file_test),
        )
        logger.info(f"✅ [OK] Startup test synthesis successful. Output: {output_file_test.resolve()}")
    except Exception as e:
        logger.error(f"❌ [FAIL] Startup test synthesis failed: {e}", exc_info=True)


# --- FastAPI Application ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    global tts_api, streaming_processor # Using global tts_api
    if _perform_startup_checks():
        _load_preset_voices()
        tts_api = _initialize_tts_model()
        if tts_api:
            _run_startup_test(tts_api)
            # --- Initialize the streaming processor ---
            try:
                logger.info("--- Initializing TTSStreamingProcessor ---")
                # Use a preset voice for initialization if available
                if preselected_voices:
                    ref_audio_path = preselected_voices[0]["audio_path"]
                    ref_text = preselected_voices[0]["text"]
                    logger.info(f"Using preset '{preselected_voices[0]['id']}' for streaming processor.")
                else:
                    # Fallback to a random reference if no presets
                    ref_audio_path, ref_text = _get_random_reference()
                    logger.info("Using random reference for streaming processor.")

                if ref_audio_path and ref_text:
                    streaming_processor = TTSStreamingProcessor(
                        model=settings.EXP_NAME,
                        ckpt_file=str(settings.CHECKPOINT_FILE),
                        vocab_file=str(settings.vocab_path),
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        device=settings.DEVICE
                    )
                    logger.info("✅ [OK] TTSStreamingProcessor initialized successfully.")
                else:
                    logger.error("❌ [FAIL] Could not find a reference audio/text to initialize TTSStreamingProcessor.")
                    streaming_processor = None

            except Exception as e:
                logger.critical(f"❌ [FATAL] Failed to initialize TTSStreamingProcessor: {e}", exc_info=True)
                streaming_processor = None
    else:
        logger.critical("Critical startup checks failed. TTS service will be unavailable.")
    
    logger.info("--- Application startup complete. ---")
    yield
    # --- Shutdown ---
    logger.info("--- Application Shutting Down ---")
    _cleanup_temp_file("startup_test_output.wav") # Cleanup fixed filename

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS, # Keep CORS from settings
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health") # No response_model for simplicity in reverted version
async def health_check():
    """Health check endpoint."""
    if tts_api is not None: # Direct check of global tts_api
        return {"status": "ok", "message": "TTS service is running."}
    else:
        raise HTTPException(status_code=503, detail="TTS service is not available.")

@app.get("/list_preset_references") # No response_model
async def list_preset_references():
    """Lists available pre-selected reference voices."""
    # Return only id, name, text as frontend likely expects this
    return [{"id": v["id"], "name": v["name"], "text": v["text"]} for v in preselected_voices]

@app.post("/synthesize/")
async def synthesize_speech(
    # Reverted to individual Form parameters
    ref_audio_file: UploadFile = File(None, description="User-uploaded WAV file for reference."),
    reference_mode: str = Form("upload", description="Mode for reference: 'upload', 'random', or 'preset'."),
    preset_reference_id: Optional[str] = Form(None, description="ID of the preset reference (e.g., 'preset_0')."),
    ref_text: Optional[str] = Form(None, description="Transcript of the user-uploaded reference audio."),
    gen_text: str = Form(..., description="The text to be synthesized."),
    nfe_step: int = Form(settings.NFE_STEP),
    speed: float = Form(settings.SPEED),
    seed: int = Form(settings.SEED),
    remove_silence: bool = Form(settings.REMOVE_SILENCE)
):
    if tts_api is None: # Direct check of global tts_api
        raise HTTPException(status_code=503, detail="TTS service is unavailable. Check server logs for initialization errors.")

    ref_audio_path_str: Optional[str] = None
    ref_text_internal: Optional[str] = None
    temp_upload_path: Optional[str] = None

    try:
        ### NEW/MODIFIED ###: Convert numbers in the input text to Persian words
        logger.info(f"Original text received: '{gen_text}'")
        processed_gen_text = convert_numbers_to_persian_words(gen_text)
        logger.info(f"Text after number conversion: '{processed_gen_text}'")

        # --- Determine Reference Audio and Text (Inlined and simplified) ---
        if reference_mode == "upload":
            if ref_audio_file:
                if not ref_text:
                    raise HTTPException(status_code=400, detail="Reference text is required for uploaded reference audio.")
                # Basic content type check (can be expanded if needed)
                if ref_audio_file.content_type not in ["audio/wav", "audio/x-wav"]:
                     logger.warning(f"Uploaded file content type '{ref_audio_file.content_type}' is not standard WAV. Proceeding, but may cause issues.")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_upload:
                    shutil.copyfileobj(ref_audio_file.file, tmp_upload)
                    temp_upload_path = tmp_upload.name
                ref_audio_path_str = temp_upload_path
                ref_text_internal = ref_text
            # If no file, proceed zero-shot (ref_audio_path_str remains None)
        
        elif reference_mode == "record":
            if ref_audio_file:
                if not ref_text:
                    raise HTTPException(status_code=400, detail="Reference text is required for recorded reference audio.")
                # Accept any audio type from the recorder, but log if not WAV
                if ref_audio_file.content_type not in ["audio/wav", "audio/x-wav"]:
                    logger.warning(f"Recorded file content type '{ref_audio_file.content_type}' is not standard WAV. Proceeding, but may cause issues.")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_upload:
                    shutil.copyfileobj(ref_audio_file.file, tmp_upload)
                    temp_upload_path = tmp_upload.name
                ref_audio_path_str = temp_upload_path
                ref_text_internal = ref_text
            # If no file, proceed zero-shot (ref_audio_path_str remains None)

        elif reference_mode == "preset":
            if not preset_reference_id:
                raise HTTPException(status_code=400, detail="`preset_reference_id` is required for 'preset' mode.")
            
            preset = next((p for p in preselected_voices if p["id"] == preset_reference_id), None)
            if not preset:
                raise HTTPException(status_code=404, detail=f"Preset reference '{preset_reference_id}' not found.")
            ref_audio_path_str = preset["audio_path"]
            ref_text_internal = preset["text"]

        elif reference_mode == "random":
            ref_audio_path_str, ref_text_internal = _get_random_reference()
            if not ref_audio_path_str:
                logger.warning("Random reference requested but none could be found. Proceeding zero-shot.")
        
        # Implicit "zero_shot" if reference_mode is not upload/preset/random and no file is provided.
        # Or if reference_mode is an unknown string.
        elif reference_mode != "zero_shot": # If it's not an explicit zero_shot, but also not other known modes
             logger.warning(f"Unknown reference_mode '{reference_mode}'. Proceeding as zero-shot if no reference audio is determined.")


        # --- Prepare for Inference ---
        ### NEW/MODIFIED ###: Use the text that has had numbers converted
        padded_gen_text = _pad_short_text(processed_gen_text)
        current_seed = None if seed == -1 else seed
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_temp_file = tmp_out.name

        logger.info(f"Request to synthesize speech with reference_mode='{reference_mode}'. Acquiring lock...")
        async with inference_lock:
            logger.info("Inference lock acquired. Synthesizing...")
            tts_api.infer(
                ref_file=ref_audio_path_str,
                ref_text=ref_text_internal.lower().strip() if ref_text_internal else None,
                ### NEW/MODIFIED ###: Pass the fully processed text to the model
                gen_text=padded_gen_text.lower().strip(),
                nfe_step=nfe_step,
                speed=speed,
                remove_silence=remove_silence,
                file_wave=output_temp_file,
                seed=current_seed,
            )
            # The seed is only reliable right after inference, before the lock is released.
            logger.info(f"Synthesis complete. Seed used: {tts_api.seed}. Releasing lock.")

        logger.info(f"Request complete. Sending file: {output_temp_file}")
        return FileResponse(
            path=output_temp_file,
            media_type="audio/wav",
            filename="synthesized_speech.wav",
            background=BackgroundTask(_cleanup_temp_file, output_temp_file)
        )

    except Exception as e:
        logger.error(f"Error during synthesis request: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An internal error occurred during synthesis: {str(e)}")
    finally:
        if temp_upload_path:
            _cleanup_temp_file(temp_upload_path)


@app.post("/AI_TTS")
async def ai_tts(
    gen_text: str = Form(..., description="Text to synthesize with a preset voice."),
    nfe_step: int = Form(settings.NFE_STEP),
    speed: float = Form(settings.SPEED),
    seed: int = Form(settings.SEED),
    remove_silence: bool = Form(settings.REMOVE_SILENCE),
    preset_reference_id: str = Form("preset_8", description="ID of the preset reference to use.")
):
    """
    Synthesizes speech using a specified preset voice and returns the audio file.
    Accepts synthesis parameters to control the output.
    """
    if tts_api is None:
        raise HTTPException(status_code=503, detail="TTS service is unavailable. Check server logs for initialization errors.")

    preset = next((p for p in preselected_voices if p["id"] == preset_reference_id), None)

    if not preset:
        raise HTTPException(status_code=404, detail=f"Preset reference '{preset_reference_id}' not found.")

    ref_audio_path_str = preset["audio_path"]
    ref_text_internal = preset["text"]
    
    try:
        logger.info(f"Original text received for /AI_TTS: '{gen_text}'")
        processed_gen_text = convert_numbers_to_persian_words(gen_text)
        logger.info(f"Text after number conversion: '{processed_gen_text}'")
        padded_gen_text = _pad_short_text(processed_gen_text)
        
        current_seed = None if seed == -1 else seed

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_temp_file = tmp_out.name
        
        logger.info(f"Request for /AI_TTS with preset '{preset_reference_id}'. Acquiring lock...")
        async with inference_lock:
            logger.info("Inference lock acquired for /AI_TTS. Synthesizing...")
            tts_api.infer(
                ref_file=ref_audio_path_str,
                ref_text=ref_text_internal.lower().strip() if ref_text_internal else None,
                gen_text=padded_gen_text.lower().strip(),
                nfe_step=nfe_step,
                speed=speed,
                remove_silence=remove_silence,
                file_wave=output_temp_file,
                seed=current_seed,
            )
            logger.info(f"Synthesis for /AI_TTS complete. Seed used: {tts_api.seed}. Releasing lock.")
        logger.info(f"Request for /AI_TTS complete. Sending file: {output_temp_file}")
        return FileResponse(
            path=output_temp_file,
            media_type="audio/wav",
            filename="ai_tts_output.wav",
            background=BackgroundTask(_cleanup_temp_file, output_temp_file)
        )
    except Exception as e:
        logger.error(f"Error during /AI_TTS synthesis: {e}", exc_info=True)
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"An internal error occurred during /AI_TTS synthesis: {str(e)}")


@app.post("/synthesize_stream/")
async def synthesize_speech_stream(
    gen_text: str = Form(..., description="Text to synthesize."),
):
    if streaming_processor is None:
        raise HTTPException(status_code=503, detail="Streaming TTS service is not available.")

    async def stream_generator(queue: asyncio.Queue):
        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
            await asyncio.sleep(0.001)

    # This blocking function runs in a separate thread
    def run_tts_and_fill_queue(text: str, queue: asyncio.Queue):
        # CRITICAL CHANGE: Acquire the threading.Lock here
        with inference_lock:
            logger.info("Inference lock acquired for streaming request in thread.")
            try:
                streaming_processor.first_package = True
                
                processed_text = convert_numbers_to_persian_words(text)
                padded_text = _pad_short_text(processed_text)
                
                audio_generator = streaming_processor.generate_stream(padded_text)
                for chunk in audio_generator:
                    # Use asyncio's thread-safe call to put items in the queue
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), asyncio.get_running_loop())
            except Exception as e:
                logger.error(f"Error in TTS thread: {e}", exc_info=True)
            finally:
                # IMPORTANT: Put the sentinel value to signal the end
                asyncio.run_coroutine_threadsafe(queue.put(None), asyncio.get_running_loop())
                logger.info("Inference lock released for streaming request in thread.")


    # --- Main Endpoint Logic ---
    try:
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()
        
        # Run the blocking function in the default thread pool executor
        # CRITICAL CHANGE: The lock is no longer managed here
        loop.run_in_executor(
            None,
            run_tts_and_fill_queue,
            gen_text,
            q
        )
        
        # Return the streaming response that reads from the queue
        # Minor refinement: Changed media_type for clarity
        return StreamingResponse(stream_generator(q), media_type="audio/raw")

    except Exception as e:
        logger.error(f"Error setting up streaming response: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start audio stream.")


if __name__ == "__main__":
    logger.info("Starting FastAPI TTS server...")
    try:
        # Basic check for CHECKPOINT_FILE from settings
        if not settings.CHECKPOINT_FILE or not Path(settings.CHECKPOINT_FILE).is_file():
             raise FileNotFoundError(f"CHECKPOINT_FILE '{settings.CHECKPOINT_FILE}' not found or not configured.")
        
        import uvicorn
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
    except (FileNotFoundError, AttributeError, ValueError) as e:
        logger.critical(f"Failed to start server. A configuration value may be missing or incorrect. Error: {e}")
        logger.critical("Please check your .env file or environment variables against config.py.")