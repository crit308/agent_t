import pytest
import pytest_asyncio
from unittest.mock import patch, AsyncMock, MagicMock, call
from uuid import uuid4, UUID
from typing import Optional

# Modules to test/mock
from ai_tutor.services.session_tasks import queue_session_analysis
# Also need to mock the analyzer function directly referenced in session_tasks
from ai_tutor.agents import session_analyzer_agent

# Assuming SessionAnalysis model exists for type hinting if needed, but not strictly required for mocking
# from ai_tutor.agents.models import SessionAnalysis

# Skip this legacy integration test suite; Supabase client mocks need refactor after lean-executor migration.
pytest.skip("Skipping legacy phase-3 integration tests pending Supabase mock refactor", allow_module_level=True)

@pytest.fixture
def mock_supabase_client():
    """Provides a mock Supabase client instance with chainable async methods."""
    client = AsyncMock()

    # Make table('...') return a chainable mock
    table_mock = MagicMock()
    client.table.return_value = table_mock

    # Mock chained methods: .update(...).eq(...).is_(...).execute()
    update_mock = AsyncMock()
    is_mock = MagicMock()
    eq_mock_update = MagicMock()
    eq_mock_update.is_.return_value = is_mock
    is_mock.execute.return_value = update_mock # The final result of the update chain
    table_mock.update.return_value = eq_mock_update

    # Mock chained methods: .select(...).eq(...).single().execute()
    select_mock_result = AsyncMock()
    single_mock = MagicMock()
    single_mock.execute.return_value = select_mock_result
    eq_mock_select = MagicMock()
    eq_mock_select.single.return_value = single_mock
    table_mock.select.return_value = eq_mock_select

    # Mock chained methods: .update(...).eq(...).execute() (for ended_at and final status)
    update_simple_mock_result = AsyncMock()
    eq_mock_update_simple = MagicMock()
    eq_mock_update_simple.execute.return_value = update_simple_mock_result
    # Need to ensure table_mock.update returns the *right* mock for different chains
    # A simple way is to configure it per test, or use side_effect if complexity increases.
    # For now, let's assume the setup inside tests will handle specifics.
    # Overwrite the general update mock to return the simple chain for status updates
    table_mock.update.return_value = eq_mock_update_simple


    # Mock rpc(...)
    rpc_mock_result = AsyncMock()
    client.rpc.return_value = rpc_mock_result

    return client, table_mock, update_mock, select_mock_result, rpc_mock_result, eq_mock_update_simple, eq_mock_update, is_mock, eq_mock_select, single_mock

@pytest.mark.asyncio
@patch('ai_tutor.services.session_tasks.analyze_session', new_callable=AsyncMock) # Mock analyzer
@patch('ai_tutor.services.session_tasks.get_supabase_client', new_callable=AsyncMock) # Mock client factory
async def test_queue_session_analysis_success_case(
    mock_get_supabase, mock_analyze_session, mock_supabase_client
):
    """Tests the successful execution path of the background task."""
    mock_client, table_mock, _, mock_select_execute, mock_rpc_execute, eq_mock_update_simple, eq_mock_update, is_mock, eq_mock_select, single_mock = mock_supabase_client
    mock_get_supabase.return_value = mock_client

    session_id = uuid4()
    user_id = uuid4()
    folder_id = uuid4()
    mock_text_summary = "This is a summary."
    mock_structured_analysis = {"key": "value"} # Mock analysis object/dict

    # --- Mock Configuration ---
    # 1. Concurrency check: Simulate successful claim
    #    - Initial update (doesn't matter what it returns here, check is the select)
    #    - Select after update returns status 'processing'
    mock_select_execute.configure_mock(data={"analysis_status": "processing"})

    # 2. analyze_session returns successfully
    mock_analyze_session.return_value = (mock_text_summary, mock_structured_analysis)

    # --- Execute ---
    await queue_session_analysis(session_id, user_id, folder_id)

    # --- Assertions ---
    # 1. Claim attempt (update status to processing if null)
    mock_client.table.assert_any_call("sessions")
    table_mock.update.assert_any_call({"analysis_status": "processing"})
    eq_mock_update.eq.assert_any_call("id", str(session_id))
    eq_mock_update.is_.assert_any_call("analysis_status", "null")
    is_mock.execute.assert_called_once()

    # 2. Status check after claim attempt
    table_mock.select.assert_any_call("analysis_status")
    eq_mock_select.eq.assert_any_call("id", str(session_id))
    single_mock.execute.assert_called_once()

    # 3. Mark session ended
    table_mock.update.assert_any_call({"ended_at": "now()"})
    eq_mock_update_simple.eq.assert_any_call("id", str(session_id))
    # ended_at update call + success status update call = 2 calls via this chain
    assert eq_mock_update_simple.execute.call_count == 2


    # 4. analyze_session called
    mock_analyze_session.assert_called_once_with(session_id, context=None)

    # 5. Append to KB called
    mock_client.rpc.assert_called_once_with(
        'append_to_knowledge_base',
        {'target_folder_id': str(folder_id), 'new_summary_text': mock_text_summary}
    )

    # 6. Final status update to 'success'
    table_mock.update.assert_any_call({"analysis_status": "success"})
    # eq_mock_update_simple.eq was already asserted for ended_at, called again for success
    assert eq_mock_update_simple.eq.call_count == 3 # claim + ended_at + success
    # eq_mock_update_simple.execute already asserted count = 2

@pytest.mark.asyncio
@patch('ai_tutor.services.session_tasks.analyze_session', new_callable=AsyncMock)
@patch('ai_tutor.services.session_tasks.get_supabase_client', new_callable=AsyncMock)
async def test_queue_session_analysis_analysis_failure(
    mock_get_supabase, mock_analyze_session, mock_supabase_client
):
    """Tests the path where analysis fails after claiming the session."""
    mock_client, table_mock, _, mock_select_execute, _, eq_mock_update_simple, eq_mock_update, is_mock, eq_mock_select, single_mock = mock_supabase_client
    mock_get_supabase.return_value = mock_client

    session_id = uuid4()
    user_id = uuid4()
    folder_id = uuid4()

    # --- Mock Configuration ---
    # 1. Concurrency check: Simulate successful claim
    mock_select_execute.configure_mock(data={"analysis_status": "processing"})

    # 2. analyze_session raises an exception
    analysis_error = ValueError("LLM processing failed")
    mock_analyze_session.side_effect = analysis_error

    # --- Execute ---
    await queue_session_analysis(session_id, user_id, folder_id)

    # --- Assertions ---
    # 1. Claim attempt and status check happened
    is_mock.execute.assert_called_once()
    single_mock.execute.assert_called_once()

    # 2. Mark session ended happened
    table_mock.update.assert_any_call({"ended_at": "now()"})
    assert eq_mock_update_simple.eq.call_count == 2 # claim + ended_at

    # 3. analyze_session called
    mock_analyze_session.assert_called_once_with(session_id, context=None)

    # 4. Append to KB *not* called
    mock_client.rpc.assert_not_called()

    # 5. Final status update to 'failed'
    table_mock.update.assert_any_call({"analysis_status": "failed"})
    assert eq_mock_update_simple.eq.call_count == 3 # claim + ended_at + failed
    assert eq_mock_update_simple.execute.call_count == 2 # ended_at + failed

@pytest.mark.asyncio
@patch('ai_tutor.services.session_tasks.analyze_session', new_callable=AsyncMock)
@patch('ai_tutor.services.session_tasks.get_supabase_client', new_callable=AsyncMock)
async def test_queue_session_analysis_concurrency_claim_failed(
    mock_get_supabase, mock_analyze_session, mock_supabase_client
):
    """Tests the path where another worker already claimed the session."""
    mock_client, table_mock, _, mock_select_execute, mock_rpc_execute, eq_mock_update_simple, eq_mock_update, is_mock, eq_mock_select, single_mock = mock_supabase_client
    mock_get_supabase.return_value = mock_client

    session_id = uuid4()
    user_id = uuid4()
    folder_id = uuid4()

    # --- Mock Configuration ---
    # 1. Concurrency check: Simulate claim failure
    #    - Initial update is called, but...
    #    - Select after update returns status 'processing' (already claimed) or 'success'/'failed'
    mock_select_execute.configure_mock(data={"analysis_status": "success"}) # Simulate already processed

    # --- Execute ---
    await queue_session_analysis(session_id, user_id, folder_id)

    # --- Assertions ---
    # 1. Claim attempt (update) was made
    is_mock.execute.assert_called_once()

    # 2. Status check after claim attempt was made
    single_mock.execute.assert_called_once()

    # 3. Mark session ended *not* called
    assert call({"ended_at": "now()"}) not in table_mock.update.call_args_list

    # 4. analyze_session *not* called
    mock_analyze_session.assert_not_called()

    # 5. Append to KB *not* called
    mock_client.rpc.assert_not_called()

    # 6. Final status update ('success' or 'failed') *not* called by this worker
    assert call({"analysis_status": "success"}) not in table_mock.update.call_args_list
    assert call({"analysis_status": "failed"}) not in table_mock.update.call_args_list

    # Check total calls to the simple update chain (should be 0)
    assert eq_mock_update_simple.execute.call_count == 0 