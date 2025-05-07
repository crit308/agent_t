import json
import logging
import asyncio
from pydantic import ValidationError

log = logging.getLogger(__name__)

async def retry_on_json_error(async_func, *args, retries=3, temp_increment=0.1, **kwargs):
    """
    A wrapper that retries an async function if it raises json.JSONDecodeError or pydantic.ValidationError.

    Args:
        async_func: The async function to call.
        *args: Positional arguments for async_func.
        retries: Maximum number of retries.
        temp_increment: Amount to increment temperature by on each retry (if 'temperature' is in kwargs).
        **kwargs: Keyword arguments for async_func.

    Returns:
        The result of async_func.

    Raises:
        The last caught exception if all retries fail.
    """
    last_exception = None
    for attempt in range(retries):
        try:
            return await async_func(*args, **kwargs)
        except (json.JSONDecodeError, ValidationError) as e:
            log.warning(f"Attempt {attempt + 1} of {retries} failed with {type(e).__name__}: {e}. Retrying...")
            last_exception = e
            if 'temperature' in kwargs and isinstance(kwargs['temperature'], (int, float)):
                kwargs['temperature'] += temp_increment
                log.info(f"Increased temperature to {kwargs['temperature']:.2f}")
            await asyncio.sleep(1)  # Optional: add a small delay before retrying
        except Exception as e:
            # For any other unexpected errors, re-raise immediately
            log.error(f"Unexpected error during retry wrapper: {e}", exc_info=True)
            raise e
    
    log.error(f"All {retries} retries failed. Re-raising last exception: {last_exception}")
    if last_exception:
        raise last_exception
    # This line should ideally not be reached if retries > 0 and an error occurred
    # but as a fallback, if last_exception is None for some reason (e.g. retries=0),
    # it's better to raise a generic error than return None implicitly.
    raise RuntimeError("Function call failed after multiple retries, but no exception was captured.") 