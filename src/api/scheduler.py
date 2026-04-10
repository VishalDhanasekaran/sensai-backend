from apscheduler.schedulers.asyncio import AsyncIOScheduler
from api.db.task import publish_scheduled_tasks
from api.db.prompt_cache import (
    cleanup_old_prompt_cache_stats,
    cleanup_old_llm_response_cache,
)
from api.cron import (
    check_memory_and_raise_alert,
)
from api.settings import settings
from datetime import timezone, timedelta
import logging
import sentry_sdk
from functools import wraps
from typing import Callable, Any

# Create IST timezone
ist_timezone = timezone(timedelta(hours=5, minutes=30))

scheduler = AsyncIOScheduler(timezone=ist_timezone)


def with_error_reporting(context: str):
    """Decorator to add Sentry error reporting to scheduled tasks"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logging.error(
                    f"Error in scheduled task '{context}': {e}", exc_info=True
                )
                if settings.sentry_dsn:
                    sentry_sdk.capture_exception(e)
                raise

        return wrapper

    return decorator


# Check for tasks to publish every minute
@scheduler.scheduled_job("interval", minutes=1)
@with_error_reporting("scheduled_task_publish")
async def check_scheduled_tasks():
    await publish_scheduled_tasks()


@scheduler.scheduled_job("cron", hour=23, minute=55, timezone=ist_timezone)
@with_error_reporting("memory_check")
async def check_memory():
    await check_memory_and_raise_alert()


@scheduler.scheduled_job("interval", hours=1)
@with_error_reporting("prompt_cache_cleanup")
async def cleanup_prompt_cache():
    await cleanup_old_prompt_cache_stats(hours=48)


@scheduler.scheduled_job("interval", hours=1)
@with_error_reporting("llm_response_cache_cleanup")
async def cleanup_llm_response_cache():
    await cleanup_old_llm_response_cache(hours=48)
