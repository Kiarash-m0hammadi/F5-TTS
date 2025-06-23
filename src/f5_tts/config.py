from pathlib import Path
from typing import List # Keep List for CORS_ORIGINS
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Holds all configuration for the application, loaded from environment variables or a .env file."""
    
    # --- Core Paths & Model Config ---
    PROJECT_NAME: str = "F5_Mana_char"
    EXP_NAME: str = "F5TTS_v1_Base"
    DATA_PATH: Path = Path("data")
    CHECKPOINT_FILE: Path

    # --- Default TTS Parameters ---
    NFE_STEP: int = 32
    SPEED: float = 1.0
    SEED: int = -1
    REMOVE_SILENCE: bool = False
    USE_EMA: bool = True
    DEVICE: str = "cuda"

    # --- Application Behavior ---
    SHORT_TEXT_THRESHOLD_WORDS: int = 5
    SHORT_TEXT_PADDING: str = ".........." # This was present in an earlier version
    PRESET_VOICE_COUNT: int = 10
    # STARTUP_TEST_FILENAME and STARTUP_TEST_TEXT are removed

    # --- Server Config ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    @property
    def vocab_path(self) -> Path:
        return self.DATA_PATH / self.PROJECT_NAME / "vocab.txt"

    @property
    def metadata_path(self) -> Path:
        return self.DATA_PATH / self.PROJECT_NAME / "metadata.csv"

    @property
    def wavs_path(self) -> Path:
        return self.DATA_PATH / self.PROJECT_NAME / "wavs"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        extra = "ignore"

settings = Settings()