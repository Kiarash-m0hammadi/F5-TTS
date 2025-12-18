# -*- coding: utf-8 -*-
"""
A FastAPI web server for the F5 Text-to-Speech (TTS) model.

This script provides a web interface to an F5 TTS model, offering both standard
(file-based) and streaming synthesis endpoints. It includes features like
dynamic voice cloning from uploaded audio, preset voices, and real-time
audio streaming.
"""

# --- Standard Library Imports ---
import asyncio
import csv
import logging
import queue
import random
import re
import shutil
import sys
import tempfile
import threading
import wave
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

# --- Third-Party Imports ---
import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from hydra.utils import get_class
from omegaconf import OmegaConf
from starlette.background import BackgroundTask

# --- Local Application Imports ---
# This block ensures that the 'src' directory is in sys.path for absolute imports.
_current_file_path = Path(__file__).resolve()
_src_path = _current_file_path.parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

from f5_tts.api import F5TTS
from f5_tts.config import settings
from f5_tts.infer.utils_infer import (
    chunk_text,
    infer_batch_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

# You must install this library first: pip install num2fawords
try:
    from num2fawords import words as num2fawords
    num2fawords_available = True
except ImportError:
    num2fawords = None  # Define it as None to prevent NameError
    num2fawords_available = False

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

if not num2fawords_available:
    logger.warning("Could not import 'num2fawords'. Number-to-word conversion will be disabled.")
    logger.warning("Please install it using: pip install num2fawords")
else:
    logger.info("Successfully imported num2fawords library.")

# --- Global State ---
tts_api: Optional[F5TTS] = None
preselected_voices: List[Dict[str, Any]] = []
streaming_processor: Optional["TTSStreamingProcessor"] = None
inference_lock = threading.Lock()


# --- Streaming Classes ---

class AudioFileWriterThread(threading.Thread):
    """A thread that writes audio chunks from a queue to a WAV file."""

    def __init__(self, output_file: str, sampling_rate: int):
        """
        Initialize the audio file writer thread.

        Args:
            output_file (str): The path to the output WAV file.
            sampling_rate (int): The sampling rate of the audio.
        """
        super().__init__()
        self.output_file = output_file
        self.sampling_rate = sampling_rate
        self.queue: queue.Queue[Optional[np.ndarray[Any, Any]]] = queue.Queue()
        self.stop_event = threading.Event()

    def run(self) -> None:
        """Process queued audio data and write it to a file."""
        logger.info("AudioFileWriterThread started.")
        try:
            with wave.open(self.output_file, "wb") as wf:
                # Explicitly type hint to resolve linter false positive
                wave_writer: wave.Wave_write = wf
                wave_writer.setnchannels(1)
                wave_writer.setsampwidth(2)  # 16-bit audio
                wave_writer.setframerate(self.sampling_rate)

                while not self.stop_event.is_set() or not self.queue.empty():
                    try:
                        chunk = self.queue.get(timeout=0.1)
                        if chunk is not None:
                            # Convert float32 to int16
                            chunk_int16 = np.int16(chunk * 32767)
                            wave_writer.writeframes(chunk_int16.tobytes())
                    except queue.Empty:
                        continue
        except Exception as e:
            logger.error("Error in AudioFileWriterThread: %s", e, exc_info=True)
        finally:
            logger.info("AudioFileWriterThread finished.")

    def add_chunk(self, chunk: np.ndarray[Any, Any]) -> None:
        """Add a new audio chunk to the queue."""
        self.queue.put(chunk)

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        logger.info("Stopping AudioFileWriterThread...")
        self.stop_event.set()
        self.join()
        logger.info("Audio writing completed.")


class TTSStreamingProcessor:
    """
    Handles the streaming generation of audio from text using a pre-loaded F5-TTS model.
    This class is optimized for real-time, chunked output.
    """

    def __init__(
        self,
        model: str,
        ckpt_file: str,
        vocab_file: str,
        ref_audio: str,
        ref_text: str,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
        target_sr: int = 16000,
    ):
        """
        Initializes the streaming processor.

        Args:
            model (str): The name of the model configuration.
            ckpt_file (str): Path to the model checkpoint file.
            vocab_file (str): Path to the vocabulary file.
            ref_audio (str): Path to the reference audio file.
            ref_text (str): Transcript of the reference audio.
            device (Optional[str]): The device to run the model on (e.g., "cuda", "cpu").
            dtype (torch.dtype): The data type for model computations.
            target_sr (int): The target sample rate for the output audio.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{model}.yaml")))
        self.model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        self.model_arc = model_cfg.model.arch
        self.mel_spec_type = model_cfg.model.mel_spec.mel_spec_type

        self.native_sampling_rate = model_cfg.model.mel_spec.target_sample_rate
        self.target_sampling_rate = target_sr

        logger.info(
            "TTS Model Native SR: %d, Target Output SR: %d",
            self.native_sampling_rate,
            self.target_sampling_rate,
        )

        self.model: torch.nn.Module = self._load_ema_model(ckpt_file, vocab_file, dtype)
        self.vocoder: torch.nn.Module = load_vocoder(
            vocoder_name=self.mel_spec_type, is_local=False, local_path="", device=self.device
        )

        # --- Resampling Setup ---
        self.resampler: Optional[torch.nn.Module] = None
        if self.native_sampling_rate != self.target_sampling_rate:
            logger.info("Initializing audio resampler...")
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=self.native_sampling_rate,
                new_freq=self.target_sampling_rate,
                resampling_method="kaiser_window",
                dtype=torch.float32,
            ).to(self.device)
        else:
            logger.info("Native and target sample rates are the same. No resampling needed.")

        self.ref_audio_path: str = ""
        self.ref_text: str = ""
        self.audio: Optional[torch.Tensor] = None
        self.sr: Optional[int] = None
        self.max_chars, self.few_chars, self.min_chars = 135, 60, 30
        self.update_reference(ref_audio, ref_text)

        # Initialize first_package before calling _warm_up
        self.first_package = True
        self._warm_up()

    def _load_ema_model(self, ckpt_file: str, vocab_file: str, dtype: torch.dtype) -> torch.nn.Module:
        """Loads the exponential moving average (EMA) model."""
        model: torch.nn.Module = load_model(
            self.model_cls,
            self.model_arc,
            ckpt_path=ckpt_file,
            mel_spec_type=self.mel_spec_type,
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        )
        return model.to(self.device, dtype=dtype)

    def update_reference(self, ref_audio_path: str, ref_text: str) -> None:
        """Updates the reference audio and text for voice cloning."""
        self.ref_audio_path, self.ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        self.audio, self.sr = torchaudio.load(self.ref_audio_path)

        ref_audio_duration = self.audio.shape[-1] / self.sr
        ref_text_byte_len = len(self.ref_text.encode("utf-8"))

        if ref_audio_duration > 0:
            chars_per_second = ref_text_byte_len / ref_audio_duration
            self.max_chars = int(chars_per_second * (25 - ref_audio_duration))
            self.few_chars = int(self.max_chars / 2)
            self.min_chars = int(self.max_chars / 4)

        logger.info("Updated reference. Max chars for chunking: %d", self.max_chars)

    def _warm_up(self) -> None:
        """Performs a quick inference to warm up the model and avoid first-run delays."""
        logger.info("Warming up the streaming model...")
        gen_text = "This is a warm-up text for the model."
        for _ in self.generate_stream(gen_text):
            pass
        logger.info("Warm-up completed.")

    def generate_stream(self, text: str) -> Generator[bytes, None, None]:
        """
        A generator that yields resampled audio chunks as bytes.

        Args:
            text (str): The text to synthesize.

        Yields:
            bytes: Raw audio chunks in 16-bit PCM format.
        """
        text_batches: List[str] = chunk_text(text, max_chars=self.max_chars)
        if not text_batches:
            return

        if self.first_package and len(text_batches) > 0:
            first_batch_chunks = chunk_text(text_batches[0], max_chars=self.few_chars)
            if first_batch_chunks:
                micro_chunks = chunk_text(first_batch_chunks[0], max_chars=self.min_chars)
                text_batches = micro_chunks + first_batch_chunks[1:] + text_batches[1:]
            self.first_package = False

        audio_stream_generator: Generator[Tuple[np.ndarray[Any, Any], int], None, None] = infer_batch_process(
            (self.audio, self.sr),
            self.ref_text,
            text_batches,
            self.model,
            self.vocoder,
            progress=None,
            device=self.device,
            streaming=True,
            chunk_size=4096,
        )

        logger.info("Starting to yield audio chunks from generator.")
        for audio_chunk_np, _ in audio_stream_generator:
            if audio_chunk_np is not None and len(audio_chunk_np) > 0:
                audio_chunk_tensor = torch.from_numpy(audio_chunk_np).to(self.device, dtype=torch.float32)

                if self.resampler:
                    resampled_chunk_tensor = self.resampler(audio_chunk_tensor)
                else:
                    resampled_chunk_tensor = audio_chunk_tensor

                chunk_bytes = np.int16(resampled_chunk_tensor.cpu().numpy() * 32767).tobytes()
                yield chunk_bytes

        logger.info("Finished yielding audio chunks.")

# --- Helper Functions ---

def _replace_number_with_words(match: "re.Match[str]") -> str:
    """Converts a regex match of a number to its Persian word form."""
    number_str = match.group(0)
    persian_to_english_map = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    english_num_str = number_str.translate(persian_to_english_map)

    try:
        number_int = int(english_num_str)
        return num2fawords(number_int) if num2fawords else number_str
    except (ValueError, TypeError):
        return number_str


def convert_numbers_to_persian_words(text: str) -> str:
    """Finds all numbers in a string and converts them to Persian words."""
    if not num2fawords_available:
        logger.warning("Skipping number-to-word conversion: 'num2fawords' is not available.")
        return text

    number_pattern = re.compile(r'[\d۰-۹]+')
    return number_pattern.sub(_replace_number_with_words, text)


def _cleanup_temp_file(path: str) -> None:
    """Safely remove a temporary file."""
    try:
        if path and Path(path).exists():
            Path(path).unlink()
            logger.info("Cleaned up temp file: %s", path)
    except OSError as e:
        logger.error("Error cleaning up temp file %s: %s", path, e, exc_info=True)


def _get_random_reference() -> Tuple[Optional[str], Optional[str]]:
    """Selects a random reference audio file and its transcript from metadata."""
    if not settings.metadata_path.is_file():
        logger.warning("Metadata file not found for random reference: %s", settings.metadata_path)
        return None, None

    try:
        with open(settings.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            lines = [line for line in reader if len(line) >= 2]

        if not lines:
            logger.warning("Metadata file %s is empty or invalid.", settings.metadata_path)
            return None, None

        audio_file_name, text_transcript = random.choice(lines)
        if not audio_file_name.lower().endswith(".wav"):
            audio_file_name += ".wav"

        audio_file_path = settings.wavs_path / audio_file_name
        if not audio_file_path.is_file():
            logger.warning("Selected random reference audio file not found: %s", audio_file_path)
            return None, None

        return str(audio_file_path), text_transcript
    except (IOError, IndexError) as e:
        logger.error("Error getting random reference: %s", e, exc_info=True)
        return None, None


def _pad_short_text(text: str) -> str:
    """Pads very short text to prevent potential model artifacts."""
    words = text.strip().split()
    if 0 < len(words) < settings.SHORT_TEXT_THRESHOLD_WORDS:
        padding = settings.SHORT_TEXT_PADDING
        padded_text = f"{padding} {text.strip()} {padding}"
        logger.info("Short text detected. Padding to: '%s'", padded_text)
        return padded_text
    return text


# --- Startup Logic ---

def _perform_startup_checks() -> bool:
    """Performs critical checks before loading the model. Returns True if all pass."""
    logger.info("--- Running Startup Configuration Checks ---")
    all_ok = True

    paths_to_check = {
        "Checkpoint File": (settings.CHECKPOINT_FILE, "file"),
        "Data Path": (settings.DATA_PATH, "dir"),
        "Project Vocab File": (settings.vocab_path, "file"),
    }

    for name, (path, path_type) in paths_to_check.items():
        path_obj = Path(path)
        is_valid = (path_type == "file" and path_obj.is_file()) or \
                   (path_type == "dir" and path_obj.is_dir())
        if not is_valid:
            logger.error("❌ [FAIL] %s check failed for path: %s", name, path_obj)
            all_ok = False
        else:
            logger.info("✅ [OK]   %s: %s", name, path_obj)

    if settings.DEVICE == "cuda" and not torch.cuda.is_available():
        logger.error("❌ [FAIL] CUDA is not available, but DEVICE is set to 'cuda'")
        all_ok = False
    else:
        logger.info("✅ [OK]   CUDA check passed (or skipped for device '%s')", settings.DEVICE)

    return all_ok


def _load_preset_voices() -> None:
    """Loads a deterministic, evenly-spaced subset of voices from metadata."""
    global preselected_voices
    preselected_voices = []
    if not settings.metadata_path.is_file():
        logger.warning("Cannot load preset voices: metadata file not found at %s", settings.metadata_path)
        return

    try:
        with open(settings.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            all_entries = sorted(
                [(line[0], line[1]) for line in reader if len(line) >= 2], key=lambda x: x[0]
            )

        if not all_entries:
            logger.warning("No valid entries in metadata file for preset voices.")
            return

        num_to_select = min(len(all_entries), settings.PRESET_VOICE_COUNT)
        if num_to_select <= 0:
            return

        indices = [0] if num_to_select == 1 else \
                  [int(i * (len(all_entries) - 1) / (num_to_select - 1)) for i in range(num_to_select)]

        for i, entry_idx in enumerate(indices):
            audio_filename, transcript = all_entries[entry_idx]
            if not audio_filename.lower().endswith(".wav"):
                audio_filename += ".wav"

            full_audio_path = settings.wavs_path / audio_filename
            if full_audio_path.is_file():
                preselected_voices.append({
                    "id": f"preset_{i}",
                    "name": Path(audio_filename).stem,
                    "text": transcript,
                    "audio_path": str(full_audio_path)
                })
            else:
                logger.warning("Audio file for preset '%s' not found. Skipping.", audio_filename)
        logger.info("Successfully loaded %d pre-selected voices.", len(preselected_voices))

    except (IOError, IndexError) as e:
        logger.error("Failed to load pre-selected voices: %s", e, exc_info=True)
        preselected_voices = []


def _initialize_tts_api() -> Optional[F5TTS]:
    """Initializes and returns the main F5TTS API instance."""
    logger.info("--- Attempting TTS API Initialization ---")
    try:
        api = F5TTS(
            model=settings.EXP_NAME,
            ckpt_file=str(settings.CHECKPOINT_FILE),
            vocab_file=str(settings.vocab_path),
            device=settings.DEVICE,
            use_ema=settings.USE_EMA,
        )
        logger.info("✅ [OK] TTS API initialized successfully.")
        return api
    except Exception as e:
        logger.critical("❌ [FATAL] TTS API initialization failed: %s", e, exc_info=True)
        return None


def _run_startup_test(api: F5TTS) -> None:
    """Performs a quick synthesis to ensure the model is working."""
    logger.info("--- Performing Startup Test Synthesis ---")
    output_file_test = Path("startup_test_output.wav")
    ref_audio_path, ref_text = _get_random_reference()

    if not (ref_audio_path and ref_text):
        logger.warning("Could not get a random reference for startup test.")

    try:
        api.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text.lower().strip() if ref_text else None,
            gen_text="This is a startup test audio.",
            nfe_step=10,
            speed=settings.SPEED,
            remove_silence=settings.REMOVE_SILENCE,
            file_wave=str(output_file_test),
        )
        logger.info("✅ [OK] Startup test synthesis successful. Output: %s", output_file_test.resolve())
    except Exception as e:
        logger.error("❌ [FAIL] Startup test synthesis failed: %s", e, exc_info=True)


# --- FastAPI Application ---

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Handles application startup and shutdown logic."""
    global tts_api, streaming_processor
    if _perform_startup_checks():
        _load_preset_voices()
        tts_api = _initialize_tts_api()
        if tts_api:
            _run_startup_test(tts_api)
            try:
                logger.info("--- Initializing TTSStreamingProcessor ---")
                preset = next((p for p in preselected_voices if p["id"] == "preset_7"), None) \
                         or (preselected_voices[0] if preselected_voices else None)

                if preset:
                    ref_audio_path, ref_text = preset["audio_path"], preset["text"]
                    logger.info("Using preset '%s' for streaming processor.", preset['id'])
                else:
                    ref_audio_path, ref_text = _get_random_reference()
                    logger.info("Using random reference for streaming processor.")

                if ref_audio_path and ref_text:
                    streaming_processor = TTSStreamingProcessor(
                        model=settings.EXP_NAME,
                        ckpt_file=str(settings.CHECKPOINT_FILE),
                        vocab_file=str(settings.vocab_path),
                        ref_audio=ref_audio_path,
                        ref_text=ref_text,
                        device=settings.DEVICE,
                        target_sr=16000,
                    )
                    logger.info("✅ [OK] TTSStreamingProcessor initialized successfully.")
                else:
                    logger.error("❌ [FAIL] Could not find a reference to initialize TTSStreamingProcessor.")
            except Exception as e:
                logger.critical("❌ [FATAL] Failed to initialize TTSStreamingProcessor: %s", e, exc_info=True)
    else:
        logger.critical("Critical startup checks failed. TTS service will be unavailable.")

    logger.info("--- Application startup complete. ---")
    yield
    logger.info("--- Application Shutting Down ---")
    _cleanup_temp_file("startup_test_output.wav")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify service availability."""
    if tts_api:
        return {"status": "ok", "message": "TTS service is running."}
    raise HTTPException(status_code=503, detail="TTS service is not available.")


@app.get("/list_preset_references")
async def list_preset_references() -> List[Dict[str, str]]:
    """Lists available pre-selected reference voices."""
    if not preselected_voices:
        return []
    return [{"id": v["id"], "name": v["name"], "text": v["text"]} for v in preselected_voices]


@app.post("/synthesize/")
async def synthesize_speech(
    ref_audio_file: Optional[UploadFile] = File(None, description="User-uploaded WAV file for reference."),
    reference_mode: str = Form("upload", description="Mode: 'upload', 'random', 'preset', or 'zero_shot'."),
    preset_reference_id: Optional[str] = Form(None, description="ID of the preset reference (e.g., 'preset_0')."),
    ref_text: Optional[str] = Form(None, description="Transcript of the user-uploaded reference audio."),
    gen_text: str = Form(..., description="The text to be synthesized."),
    nfe_step: int = Form(settings.NFE_STEP),
    speed: float = Form(settings.SPEED),
    seed: int = Form(settings.SEED),
    remove_silence: bool = Form(settings.REMOVE_SILENCE),
):
    """Endpoint for standard, file-based speech synthesis."""
    if tts_api is None:
        raise HTTPException(status_code=503, detail="TTS service is unavailable. Check server logs.")

    ref_audio_path_str: Optional[str] = None
    ref_text_internal: Optional[str] = None
    temp_upload_path: Optional[str] = None

    try:
        processed_gen_text = convert_numbers_to_persian_words(gen_text)

        if reference_mode in ["upload", "record"]:
            if not ref_audio_file or not ref_text:
                raise HTTPException(status_code=400, detail=f"Audio and text are required for '{reference_mode}' mode.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                shutil.copyfileobj(ref_audio_file.file, tmp)
                temp_upload_path = tmp.name
            ref_audio_path_str, ref_text_internal = temp_upload_path, ref_text
        elif reference_mode == "preset":
            if not preset_reference_id:
                raise HTTPException(status_code=400, detail="`preset_reference_id` is required.")
            preset = next((p for p in preselected_voices if p["id"] == preset_reference_id), None)
            if not preset:
                raise HTTPException(status_code=404, detail=f"Preset '{preset_reference_id}' not found.")
            ref_audio_path_str, ref_text_internal = preset["audio_path"], preset["text"]
        elif reference_mode == "random":
            ref_audio_path_str, ref_text_internal = _get_random_reference()
            if not ref_audio_path_str:
                logger.warning("Random reference requested but none found. Proceeding zero-shot.")
        elif reference_mode != "zero_shot":
            raise HTTPException(status_code=400, detail=f"Unknown reference_mode '{reference_mode}'.")

        padded_gen_text = _pad_short_text(processed_gen_text)
        current_seed = None if seed == -1 else seed

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
            output_temp_file = tmp_out.name

        logger.info("Request to synthesize speech with mode='%s'. Acquiring lock...", reference_mode)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, inference_lock.acquire)
        try:
            logger.info("Inference lock acquired. Synthesizing...")
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
            logger.info("Synthesis complete. Seed used: %s. Releasing lock.", tts_api.seed)
        finally:
            inference_lock.release()

        return FileResponse(
            path=output_temp_file,
            media_type="audio/wav",
            filename="synthesized_speech.wav",
            background=BackgroundTask(_cleanup_temp_file, output_temp_file),
        )
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError) as e:
        logger.warning("Client error during synthesis: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error("Internal error during synthesis request: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.") from e
    finally:
        if temp_upload_path:
            _cleanup_temp_file(temp_upload_path)


@app.post("/synthesize_stream/")
async def synthesize_speech_stream(
    gen_text: str = Form(..., description="Text to synthesize."),
):
    """Endpoint for streaming, real-time speech synthesis."""
    if streaming_processor is None:
        raise HTTPException(status_code=503, detail="Streaming TTS service is not available.")

    async def stream_generator(q: asyncio.Queue[Optional[bytes]]) -> AsyncGenerator[bytes, None]:
        """Yields audio chunks from the queue as they become available."""
        while True:
            chunk = await q.get()
            if chunk is None:
                break
            yield chunk
            await asyncio.sleep(0.001)

    def run_tts_in_thread(text: str, q: asyncio.Queue[Optional[bytes]], loop: asyncio.AbstractEventLoop):
        """The blocking TTS function that runs in a separate thread."""
        with inference_lock:
            logger.info("Inference lock acquired for streaming request.")
            try:
                streaming_processor.first_package = True
                processed_text = convert_numbers_to_persian_words(text)
                padded_text = _pad_short_text(processed_text)

                audio_generator = streaming_processor.generate_stream(padded_text)
                for chunk in audio_generator:
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop)
            except Exception as e:
                logger.error("Error in TTS streaming thread: %s", e, exc_info=True)
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop)
                logger.info("Inference lock released for streaming request.")

    try:
        queue_ = asyncio.Queue()
        main_loop = asyncio.get_running_loop()

        main_loop.run_in_executor(
            None, run_tts_in_thread, gen_text, queue_, main_loop
        )

        return StreamingResponse(stream_generator(queue_), media_type="audio/raw")
    except Exception as e:
        logger.error("Error setting up streaming response: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start audio stream.") from e


if __name__ == "__main__":
    logger.info("Starting FastAPI TTS server...")
    try:
        if not settings.CHECKPOINT_FILE or not Path(settings.CHECKPOINT_FILE).is_file():
            raise FileNotFoundError(f"CHECKPOINT_FILE '{settings.CHECKPOINT_FILE}' not found.")

        import uvicorn
        uvicorn.run(app, host=settings.HOST, port=settings.PORT)
    except (FileNotFoundError, AttributeError, ValueError) as e:
        logger.critical("Failed to start server. A configuration value may be missing or incorrect: %s", e)
        logger.critical("Please check your .env file or environment variables.")