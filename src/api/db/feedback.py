import json
from typing import Dict, List, Optional

from api.config import feedback_attempts_table_name, feedback_items_table_name
from api.utils.db import execute_db_operation, execute_many_db_operation


def _parse_evidence(evidence_json: Optional[str]) -> List[Dict]:
    if not evidence_json:
        return []

    try:
        evidence = json.loads(evidence_json)
        return evidence if isinstance(evidence, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


async def create_feedback_attempt(
    user_id: int,
    task_id: int,
    question_id: Optional[int],
    rubric_version: str,
    rubric_hash: str,
    model_used: str,
    overall_score: float,
    feedback_summary: str,
    raw_feedback_json: Dict,
) -> int:
    return await execute_db_operation(
        f"""
        INSERT INTO {feedback_attempts_table_name}
        (user_id, task_id, question_id, rubric_version, rubric_hash, model_used, overall_score, feedback_summary, raw_feedback_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            task_id,
            question_id,
            rubric_version,
            rubric_hash,
            model_used,
            overall_score,
            feedback_summary,
            json.dumps(raw_feedback_json),
        ),
        get_last_row_id=True,
    )


async def create_feedback_items(attempt_id: int, criteria: List[Dict]):
    if not criteria:
        return

    params = [
        (
            attempt_id,
            criterion["criterion_name"],
            criterion["score"],
            criterion.get("max_score", 0),
            criterion.get("pass_score", 0),
            json.dumps(criterion.get("evidence", [])),
            criterion.get("next_step", ""),
            criterion.get("severity", "medium"),
            "open",
        )
        for criterion in criteria
    ]

    await execute_many_db_operation(
        f"""
        INSERT INTO {feedback_items_table_name}
        (attempt_id, criterion_name, score, max_score, pass_score, evidence_json, next_step, severity, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        params,
    )


async def get_feedback_items_for_attempt(attempt_id: int) -> List[Dict]:
    rows = await execute_db_operation(
        f"""
        SELECT id, attempt_id, criterion_name, score, max_score, pass_score, evidence_json, next_step, severity, status
        FROM {feedback_items_table_name}
        WHERE attempt_id = ? AND deleted_at IS NULL
        ORDER BY id ASC
        """,
        (attempt_id,),
        fetch_all=True,
    )

    return [
        {
            "id": row[0],
            "attempt_id": row[1],
            "criterion_name": row[2],
            "score": row[3],
            "max_score": row[4],
            "pass_score": row[5],
            "evidence": _parse_evidence(row[6]),
            "next_step": row[7],
            "severity": row[8],
            "status": row[9],
        }
        for row in rows
    ]


async def get_latest_feedback_attempt(
    user_id: int,
    task_id: int,
    question_id: Optional[int],
) -> Optional[Dict]:
    if question_id is None:
        row = await execute_db_operation(
            f"""
            SELECT id, user_id, task_id, question_id, rubric_version, rubric_hash, model_used, overall_score, feedback_summary, raw_feedback_json, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id IS NULL AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, task_id),
            fetch_one=True,
        )
    else:
        row = await execute_db_operation(
            f"""
            SELECT id, user_id, task_id, question_id, rubric_version, rubric_hash, model_used, overall_score, feedback_summary, raw_feedback_json, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id = ? AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, task_id, question_id),
            fetch_one=True,
        )

    if not row:
        return None

    raw_feedback_json = None
    if row[9]:
        try:
            raw_feedback_json = json.loads(row[9])
        except (json.JSONDecodeError, TypeError):
            raw_feedback_json = None

    return {
        "id": row[0],
        "user_id": row[1],
        "task_id": row[2],
        "question_id": row[3],
        "rubric_version": row[4],
        "rubric_hash": row[5],
        "model_used": row[6],
        "overall_score": row[7],
        "feedback_summary": row[8],
        "raw_feedback_json": raw_feedback_json,
        "created_at": row[10],
    }


async def get_previous_feedback_attempt(
    user_id: int,
    task_id: int,
    question_id: Optional[int],
    current_attempt_id: int,
) -> Optional[Dict]:
    if question_id is None:
        row = await execute_db_operation(
            f"""
            SELECT id, user_id, task_id, question_id, rubric_version, rubric_hash, model_used, overall_score, feedback_summary, raw_feedback_json, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id IS NULL AND id < ? AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, task_id, current_attempt_id),
            fetch_one=True,
        )
    else:
        row = await execute_db_operation(
            f"""
            SELECT id, user_id, task_id, question_id, rubric_version, rubric_hash, model_used, overall_score, feedback_summary, raw_feedback_json, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id = ? AND id < ? AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id, task_id, question_id, current_attempt_id),
            fetch_one=True,
        )

    if not row:
        return None

    raw_feedback_json = None
    if row[9]:
        try:
            raw_feedback_json = json.loads(row[9])
        except (json.JSONDecodeError, TypeError):
            raw_feedback_json = None

    return {
        "id": row[0],
        "user_id": row[1],
        "task_id": row[2],
        "question_id": row[3],
        "rubric_version": row[4],
        "rubric_hash": row[5],
        "model_used": row[6],
        "overall_score": row[7],
        "feedback_summary": row[8],
        "raw_feedback_json": raw_feedback_json,
        "created_at": row[10],
    }


async def get_feedback_attempts(
    user_id: int,
    task_id: int,
    question_id: Optional[int],
    limit: int = 20,
) -> List[Dict]:
    if question_id is None:
        rows = await execute_db_operation(
            f"""
            SELECT id, overall_score, feedback_summary, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id IS NULL AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, task_id, limit),
            fetch_all=True,
        )
    else:
        rows = await execute_db_operation(
            f"""
            SELECT id, overall_score, feedback_summary, created_at
            FROM {feedback_attempts_table_name}
            WHERE user_id = ? AND task_id = ? AND question_id = ? AND deleted_at IS NULL
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, task_id, question_id, limit),
            fetch_all=True,
        )

    return [
        {
            "attempt_id": row[0],
            "overall_score": row[1],
            "feedback_summary": row[2],
            "created_at": row[3],
        }
        for row in rows
    ]
