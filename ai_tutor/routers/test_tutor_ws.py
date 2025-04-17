import uuid
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from agents import Runner
from ai_tutor.api import app
from ai_tutor.routers.tutor_ws import session_manager
from ai_tutor.dependencies import SUPABASE_CLIENT

# Monkeypatch Supabase auth for all tests
def test_supabase_auth(monkeypatch):
    class DummyAuth:
        def get_user(self, jwt):
            class User:
                id = "user-id"
            return type("R", (), {"user": User()})
    monkeypatch.setattr(SUPABASE_CLIENT, "auth", DummyAuth())
    # Assert dummy works
    user_response = SUPABASE_CLIENT.auth.get_user("dummy")
    assert hasattr(user_response, 'user')

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_supabase(monkeypatch):
    # Ensure JWT validation always returns a valid user
    class DummyAuth:
        def get_user(self, jwt):
            class User:
                id = "user-id"
            return type("R", (), {"user": User()})
    monkeypatch.setattr(SUPABASE_CLIENT, "auth", DummyAuth())


def test_ws_session_not_found(client):
    # Simulate no context found
    session_manager.get_session_context = AsyncMock(return_value=None)
    session_id = uuid.uuid4()
    with client.websocket_connect(
        f"/api/v1/ws/session/{session_id}",
        headers={"Authorization": "Bearer token"}
    ) as ws:
        # First message should be an error event
        data = ws.receive_json()
        assert data.get("type") == "error"
        assert "not found" in data.get("detail", "").lower()


def test_ws_streaming_success(client):
    # Prepare dummy TutorContext
    from ai_tutor.context import TutorContext, UserModelState
    dummy_ctx = TutorContext(
        session_id=uuid.UUID(int=0),
        user_id="user-id",
        folder_id=uuid.UUID(int=1),
        vector_store_id=None,
        analysis_result=None,
        user_model_state=UserModelState()
    )
    session_manager.get_session_context = AsyncMock(return_value=dummy_ctx)

    # Monkeypatch Runner.run_streamed to yield one event
    class DummyEvent:
        def model_dump(self):
            return {"hello": "world"}

    class DummyStream:
        async def stream_events(self):
            yield DummyEvent()

    Runner.run_streamed = lambda agent, input, context, run_config: DummyStream()

    with client.websocket_connect(
        f"/api/v1/ws/session/{dummy_ctx.session_id}",
        headers={"Authorization": "Bearer token"}
    ) as ws:
        # Send a JSON message to start streaming
        ws.send_json({"message": "test"})
        # Should receive our dummy event
        data = ws.receive_json()
        assert data == {"hello": "world"} 