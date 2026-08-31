from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_endpoint() -> None:
    settings = Settings(
        app_name="Travel Agent Test API",
        environment="test",
        api_prefix="/api/v1",
    )
    app = create_app(settings)

    response = TestClient(app).get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Travel Agent Test API",
        "environment": "test",
    }