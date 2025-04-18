import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from ai_tutor.telemetry import log_tool

class DummyUsage:
    def __init__(self, prompt_tokens, completion_tokens):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class DummyCtx:
    def __init__(self, session_id, user_id):
        self.session_id = session_id
        self.user_id = user_id

@pytest.mark.asyncio
async def test_log_tool_inserts_edge_log():
    # Arrange
    ctx = DummyCtx(session_id='sess-123', user_id='user-456')
    usage = DummyUsage(prompt_tokens=10, completion_tokens=20)
    
    class DummyResult:
        usage = usage
    
    async def dummy_tool(ctx):
        return DummyResult()

    # Patch SUPABASE_CLIENT.table().insert().execute()
    with patch('ai_tutor.telemetry.SUPABASE_CLIENT') as mock_client:
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()
        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute

        decorated = log_tool(dummy_tool)
        await decorated(ctx)

        # Assert
        mock_client.table.assert_any_call('edge_logs')
        args, kwargs = mock_table.insert.call_args
        payload = args[0]
        assert payload['session_id'] == 'sess-123'
        assert payload['user_id'] == 'user-456'
        assert payload['tool'] == 'dummy_tool'
        assert payload['prompt_tokens'] == 10
        assert payload['completion_tokens'] == 20
        assert isinstance(payload['latency_ms'], int)
        assert payload['trace_id'] is not None 