"""
Application configuration — all environment variables loaded via pydantic-settings.
Copy .env.example → .env and fill values before running.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment ────────────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"

    # ── Server ─────────────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8001
    # Public-facing base URL used inside Twilio TwiML callbacks and WebSocket
    # connection strings.  Set to your real domain in production — do NOT set
    # this to an ngrok URL on the production server.
    # Example (prod): "https://api.kallabot.com"
    # Example (local): "https://abc123.ngrok.io"  ← local dev only
    base_url: str = "http://localhost:8001"

    # ── Database (PostgreSQL) ──────────────────────────────────────────────────
    database_url: str = "postgresql://kallabot:kallabot@localhost:5432/kallabot"
    db_min_connections: int = 2
    db_max_connections: int = 20

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # ── AWS (S3 recording storage) ────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "kallabot-recordings"

    # ── Azure (speech, optional) ──────────────────────────────────────────────
    azure_speech_key: str = ""
    azure_speech_region: str = ""

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str = ""

    # ── Deepgram ──────────────────────────────────────────────────────────────
    deepgram_api_key: str = ""

    # ── ElevenLabs ────────────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""

    # ── Stripe (billing) ──────────────────────────────────────────────────────
    stripe_secret_key: str = ""
    stripe_webhook_secret: str = ""

    # ── Pinecone / vector DB (optional) ───────────────────────────────────────
    pinecone_api_key: str = ""
    pinecone_environment: str = ""

    # ── Internal ──────────────────────────────────────────────────────────────
    # Max concurrent calls per organisation (enforced in middleware)
    max_concurrent_calls_per_org: int = 10
    # JWT / HMAC secret if needed for future token auth
    secret_key: str = "change-me-in-production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached singleton Settings instance."""
    return Settings()
