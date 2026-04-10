from datetime import datetime, timedelta
from api.utils.db import get_new_db_connection
from api.config import prompt_cache_stats_table_name


async def log_prompt_cache_stat(
    cache_key: str,
    model: str,
    is_hit: bool,
    prompt_tokens: int | None = None,
    cached_tokens: int | None = None,
):
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""INSERT INTO {prompt_cache_stats_table_name}
                (cache_key, model, is_hit, prompt_tokens, cached_tokens)
                VALUES (?, ?, ?, ?, ?)""",
            (cache_key, model, is_hit, prompt_tokens, cached_tokens),
        )
        await conn.commit()


async def cleanup_old_prompt_cache_stats(hours: int = 48):
    cutoff = datetime.now() - timedelta(hours=hours)
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()
        await cursor.execute(
            f"""DELETE FROM {prompt_cache_stats_table_name}
                WHERE created_at < ?""",
            (cutoff.isoformat(),),
        )
        await conn.commit()


async def get_prompt_cache_stats() -> dict:
    async with get_new_db_connection() as conn:
        cursor = await conn.cursor()

        await cursor.execute(
            f"""SELECT COUNT(*) FROM {prompt_cache_stats_table_name}"""
        )
        total = (await cursor.fetchone())[0]

        await cursor.execute(
            f"""SELECT COUNT(*) FROM {prompt_cache_stats_table_name} WHERE is_hit = 1"""
        )
        hits = (await cursor.fetchone())[0]

        await cursor.execute(
            f"""SELECT model, COUNT(*) as count,
                SUM(CASE WHEN is_hit = 1 THEN 1 ELSE 0 END) as hits,
                SUM(prompt_tokens) as total_prompt_tokens,
                SUM(cached_tokens) as total_cached_tokens
                FROM {prompt_cache_stats_table_name}
                GROUP BY model"""
        )
        by_model = []
        for row in await cursor.fetchall():
            by_model.append(
                {
                    "model": row[0],
                    "total_requests": row[1],
                    "cache_hits": row[2],
                    "total_prompt_tokens": row[3] or 0,
                    "total_cached_tokens": row[4] or 0,
                }
            )

    return {
        "total_requests": total,
        "cache_hits": hits,
        "cache_misses": total - hits,
        "hit_rate": hits / total if total > 0 else 0,
        "by_model": by_model,
    }
