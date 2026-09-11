from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.api.auth_routes import router as auth_router
from app.api.chat_routes import router as chat_router
from app.api.health import router as health_router
from app.api.trip_routes import router as trip_router
from app.config import Settings, get_settings
from app.rag.vector_store import init_vector_store


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """管理应用启动和关闭。"""
    settings = application.state.settings

    if settings.environment != "test":
        checkpoint_url = settings.database_url.replace(
            "postgresql+asyncpg://",
            "postgresql://",
            1,
        )

        async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as checkpointer:
            await checkpointer.setup()

    init_vector_store()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    current_settings = settings or get_settings()

    application = FastAPI(
        title=current_settings.app_name,
        version=current_settings.app_version,
        debug=current_settings.debug,
        lifespan=lifespan,
    )
    application.state.settings = current_settings
    application.include_router(
        health_router,
        prefix=current_settings.api_prefix,
    )
    application.include_router(
        auth_router,
        prefix=current_settings.api_prefix,
    )
    application.include_router(
        trip_router,
        prefix=current_settings.api_prefix,
    )
    application.include_router(
        chat_router,
        prefix=current_settings.api_prefix,
    )

    return application


app = create_app()
