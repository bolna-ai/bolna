"""
Kallabot API — main FastAPI application entry point.

All 93 routes (91 HTTP + 2 WebSocket) are mounted here via separate router
modules.  Startup/shutdown lifecycle events initialise the DB connection pool
and Redis client so every request handler has them available via dependency
injection.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.dependencies import shutdown_db, shutdown_redis, startup_db, startup_redis

# ── Routers ───────────────────────────────────────────────────────────────────
from app.routers import (
    accounts,
    agents,
    calls,
    campaigns,
    contacts,
    countries,
    health,
    knowledgebase,
    phone_numbers,
    regulatory,
    tools,
    transfers,
    webhooks,
    websockets,
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: start → yield → shutdown."""
    # Startup
    await startup_db()
    await startup_redis()

    yield

    # Shutdown
    await shutdown_redis()
    await shutdown_db()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Kallabot API",
        version="2.0.0",
        description="Voice-AI calling platform built on Bolna OSS",
        lifespan=lifespan,
        # Disable docs in production if desired
        docs_url=None if settings.environment == "production" else "/docs",
        redoc_url=None if settings.environment == "production" else "/redoc",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Mount routers ─────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(accounts.router)
    app.include_router(agents.router)
    app.include_router(agents.agents_list_router)
    app.include_router(calls.router)
    app.include_router(campaigns.router)
    app.include_router(webhooks.router)
    app.include_router(websockets.router)
    app.include_router(tools.router)
    app.include_router(knowledgebase.router)
    app.include_router(contacts.router)
    app.include_router(phone_numbers.router)
    app.include_router(transfers.router)
    app.include_router(regulatory.router)
    app.include_router(countries.router)

    return app


app = create_app()
