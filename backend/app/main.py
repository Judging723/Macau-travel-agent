from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    current_settings = settings or get_settings()

    application = FastAPI(
        title=current_settings.app_name,
        version=current_settings.app_version,
        debug=current_settings.debug,
    )
    application.state.settings = current_settings
    application.include_router(
        health_router,
        prefix=current_settings.api_prefix,
    )

    return application


app = create_app()