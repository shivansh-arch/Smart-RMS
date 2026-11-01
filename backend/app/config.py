from pathlib import Path
from typing import List, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    CORS_ORIGINS: Union[str, List[str]] = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]

    # AI Configuration
    AI_PROVIDER: str = "gemini"  # "mock" or "gemini"
    NLP_MODEL_PROVIDER: str = "deterministic"  # "deterministic", "tfidf_logistic", "tfidf_svm", "sentence_transformer"
    CLAIM_VERIFIER_PROVIDER: str = "deterministic"  # "deterministic" or "nli"
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-xsmall"
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"
    AI_CONFIDENCE_THRESHOLD: float = 0.75

    # Privacy & Safety
    ENABLE_PII_REDACTION: bool = True
    PII_MASK_CHARACTER: str = "*"
    AUDIT_LOG_ENABLED: bool = True

    # Vector Store & RAG
    VECTOR_STORE_TYPE: str = "mock"  # "mock" or "chroma"
    CHROMA_PERSIST_DIRECTORY: str = "./chroma_data"
    EMBEDDING_MODEL: str = "text-embedding-004"

    # Integration Mode
    INTEGRATION_MODE: str = "mock"

    # File Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent
    MOCK_DATA_DIR: Path = ROOT_DIR / "data" / "mock"

    @field_validator("CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return [i.strip() for i in v.split(",") if i.strip()]
        return v

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
