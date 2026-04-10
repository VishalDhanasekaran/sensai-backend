import os
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator, Optional, Dict, List, Any
import json
import re
import hashlib
from copy import deepcopy
from pydantic import BaseModel, Field, create_model
from api.config import openai_plan_to_model_name
from api.models import (
    AIChatRequest,
    ChatResponseType,
    TaskType,
    QuestionType,
    ReevaluateFeedbackRequest,
    ReevaluateFeedbackResponse,
    FeedbackAttemptSummary,
)
from api.llm import (
    run_llm_with_openai,
    stream_llm_with_openai,
)
from api.settings import settings
from api.db.task import (
    get_task_metadata,
    get_question,
    get_task,
    get_scorecard,
)
from api.db.chat import (
    get_question_chat_history_for_user,
    get_task_chat_history_for_user,
)
from api.db.feedback import (
    create_feedback_attempt,
    create_feedback_items,
    get_previous_feedback_attempt,
    get_feedback_items_for_attempt,
    get_feedback_attempts,
)
from api.db.utils import construct_description_from_blocks
from api.utils.s3 import (
    download_file_from_s3_as_bytes,
    get_media_upload_s3_key_from_uuid,
)
from api.utils.audio import prepare_audio_input_for_ai
from api.utils.file_analysis import extract_submission_file
from api.db.user import get_user_first_name
from langfuse import get_client, observe
from api.prompts import compile_prompt
from api.prompts.router import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from api.prompts.rewrite_query import REWRITE_QUERY_SYSTEM_PROMPT, REWRITE_QUERY_USER_PROMPT
from api.prompts.objective_question import OBJECTIVE_QUESTION_SYSTEM_PROMPT, OBJECTIVE_QUESTION_USER_PROMPT
from api.prompts.subjective_question import SUBJECTIVE_QUESTION_SYSTEM_PROMPT, SUBJECTIVE_QUESTION_USER_PROMPT
from api.prompts.doubt_solving import DOUBT_SOLVING_SYSTEM_PROMPT, DOUBT_SOLVING_USER_PROMPT
from api.prompts.assignment import ASSIGNMENT_SYSTEM_PROMPT, ASSIGNMENT_USER_PROMPT

router = APIRouter()

langfuse = get_client()


def convert_chat_history_to_prompt(chat_history: list[dict]) -> str:
    role_to_label = {
        "user": "Student",
        "assistant": "AI",
    }
    return "\n".join(
        [
            f"<{role_to_label[message['role']]}>\n{message['content']}\n</{role_to_label[message['role']]}>"
            for message in chat_history
        ]
    )


def get_latest_file_uuid_from_chat_history(chat_history: list[dict]) -> Optional[str]:
    """
    Extract the latest file_uuid from chat history.
    """
    if not chat_history:
        return None

    # Iterate through chat history in reverse to find the latest file submission
    for message in reversed(chat_history):
        if message.get("role") == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                try:
                    # Try to parse as JSON to check if it's a file submission
                    file_data = json.loads(content)
                    if isinstance(file_data, dict) and "file_uuid" in file_data:
                        return file_data["file_uuid"]
                except (json.JSONDecodeError, TypeError):
                    # Not a JSON string, continue
                    continue
    return None


def format_chat_history_with_audio(chat_history: list[dict]) -> str:
    chat_history = deepcopy(chat_history)

    role_to_label = {
        "user": "Student",
        "assistant": "AI",
    }

    parts = []

    for message in chat_history:
        label = role_to_label[message["role"]]

        if isinstance(message["content"], list):
            for item in message["content"]:
                if item["type"] == "input_audio":
                    item.pop("input_audio")
                    item["content"] = "<audio_message>"

        if message["role"] == "user":
            content = message["content"]
            parts.append(f"**{label}**\n\n```\n{content}\n```\n\n")
        else:
            # Wherever there is a single \n followed by content before and either nothing after or non \n after, replace that \n with 2 \n\n
            import re

            # Replace a single newline between content with double newlines, except when already double or more
            def single_newline_to_double(text):
                # This regex matches single \n (not preceded nor followed by \n) with non-\n after, or end of string
                #  - positive lookbehind: previous char is not \n
                #  - match \n
                #  - negative lookahead: next char is not \n
                #  - next char is not \n or is end of string
                return re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)

            content_str = single_newline_to_double(
                message["content"].replace("```", "\n")
            )
            parts.append(f"**{label}**\n\n{content_str}\n\n")

    return "\n\n---\n\n".join(parts)


@observe(name="rewrite_query")
async def rewrite_query(
    chat_history: list[dict],
    question_details: str,
    user_id: str = None,
    is_root_trace: bool = False,
):
    # rewrite query
    messages = compile_prompt(
        REWRITE_QUERY_SYSTEM_PROMPT,
        REWRITE_QUERY_USER_PROMPT,
        chat_history=convert_chat_history_to_prompt(chat_history),
        reference_material=question_details,
    )

    model = openai_plan_to_model_name["text-mini"]

    class Output(BaseModel):
        rewritten_query: str = Field(
            description="The rewritten query/message of the student"
        )

    messages += chat_history

    pred = await run_llm_with_openai(
        model=model,
        messages=messages,
        response_model=Output,
        max_output_tokens=8192,
    )

    llm_input = f"# Chat History\n\n{convert_chat_history_to_prompt(chat_history)}\n\n# Reference Material\n\n{question_details}"

    if is_root_trace:
        langfuse_update_fn = langfuse.update_current_trace
    else:
        langfuse_update_fn = langfuse.update_current_generation

    output = pred.rewritten_query
    langfuse_update_fn(
        input=llm_input,
        output=output,
        metadata={
            "prompt_name": "rewrite-query",
            "input": llm_input,
            "output": output,
        },
    )

    if user_id is not None and is_root_trace:
        langfuse.update_current_trace(
            user_id=user_id,
        )

    return output


@observe(name="router")
async def get_model_for_task(
    chat_history: list[dict],
    question_details: str,
    user_id: str = None,
    is_root_trace: bool = False,
):
    class Output(BaseModel):
        chain_of_thought: str = Field(
            description="The chain of thought process for the decision to use a reasoning model or a general-purpose model"
        )
        use_reasoning_model: bool = Field(
            description="Whether to use a reasoning model to evaluate the student's response"
        )

    messages = compile_prompt(
        ROUTER_SYSTEM_PROMPT,
        ROUTER_USER_PROMPT,
        task_details=question_details,
    )

    messages += chat_history

    router_output = await run_llm_with_openai(
        model=openai_plan_to_model_name["router"],
        messages=messages,
        response_model=Output,
        max_output_tokens=4096,
    )

    use_reasoning_model = router_output.use_reasoning_model

    if use_reasoning_model:
        model = openai_plan_to_model_name["reasoning"]
    else:
        model = openai_plan_to_model_name["text"]

    llm_input = f"# Chat History\n\n{convert_chat_history_to_prompt(chat_history)}\n\n# Task Details\n\n{question_details}"

    if is_root_trace:
        langfuse_update_fn = langfuse.update_current_trace
    else:
        langfuse_update_fn = langfuse.update_current_generation

    langfuse_update_fn(
        input=llm_input,
        output=use_reasoning_model,
        metadata={
            "prompt_name": "router",
            "input": llm_input,
            "output": use_reasoning_model,
        },
    )

    if user_id is not None and is_root_trace:
        langfuse.update_current_trace(
            user_id=user_id,
        )

    return model


def get_user_audio_message_for_chat_history(uuid: str) -> list[dict]:
    if settings.s3_folder_name:
        audio_data = download_file_from_s3_as_bytes(
            get_media_upload_s3_key_from_uuid(uuid, "wav")
        )
    else:
        with open(os.path.join(settings.local_upload_folder, f"{uuid}.wav"), "rb") as f:
            audio_data = f.read()

    return [
        {
            "type": "input_audio",
            "input_audio": {
                "data": prepare_audio_input_for_ai(audio_data),
                "format": "wav",
            },
        },
    ]


def format_ai_scorecard_report(scorecard: Any) -> str:
    scorecard_as_prompt = []

    # Scorecard can be either:
    # 1) list[{category, score, feedback, ...}] (legacy UI shape)
    # 2) dict[{criterion_name: {score, feedback, ...}}] (raw model output)
    if isinstance(scorecard, dict):
        scorecard = [
            {
                "category": criterion_name,
                **(criterion_data if isinstance(criterion_data, dict) else {}),
            }
            for criterion_name, criterion_data in scorecard.items()
        ]

    if not isinstance(scorecard, list):
        return ""

    for criterion in scorecard:
        if not isinstance(criterion, dict):
            continue

        category = (
            criterion.get("category")
            or criterion.get("criterion_name")
            or "Criterion"
        )
        score = criterion.get("score", "")
        feedback = criterion.get("feedback") or {}

        if not isinstance(feedback, dict):
            feedback = {"wrong": str(feedback)}

        row_as_prompt = []
        row_as_prompt.append(f"**{category}**: {score}")

        if feedback.get("correct"):
            row_as_prompt.append(f"What worked well: {feedback['correct']}")
        if feedback.get("wrong"):
            row_as_prompt.append(f"What needs improvement: {feedback['wrong']}")

        if criterion.get("next_step"):
            row_as_prompt.append(f"Next step: {criterion['next_step']}")

        scorecard_as_prompt.append("\n".join(row_as_prompt))

    return "\n\n".join(scorecard_as_prompt)


def convert_scorecard_to_prompt(scorecard: list[dict]) -> str:
    scoring_criteria_as_prompt = []

    for index, criterion in enumerate(scorecard["criteria"]):
        criterion_name = criterion["name"].replace('"', "")
        scoring_criteria_as_prompt.append(
            f"""Criterion {index + 1}:\n**Name**: **{criterion_name}** [min_score: {criterion['min_score']}, max_score: {criterion['max_score']}, pass_score: {criterion.get('pass_score', criterion['max_score'])}]\n\n{criterion['description']}"""
        )

    return "\n\n".join(scoring_criteria_as_prompt)


def build_evaluation_context(evaluation_criteria: dict) -> str:
    """
    Build evaluation context string with overall scoring info.
    """
    evaluation_context = "**Overall Assignment Scoring:**\n"
    evaluation_context += (
        f"- Minimum Score: {evaluation_criteria.get('min_score', 0)}\n"
    )
    evaluation_context += (
        f"- Maximum Score: {evaluation_criteria.get('max_score', 100)}\n"
    )
    evaluation_context += f"- Pass Score: {evaluation_criteria.get('pass_score', 60)}\n"

    return evaluation_context


async def build_knowledge_base_from_context(context: dict) -> str:
    """
    Build knowledge base description from a context dict that may contain
    blocks and linked material IDs.
    """
    if not context or not context.get("blocks"):
        return ""

    knowledge_blocks = context["blocks"]

    # Add linked learning materials
    linked_ids = context.get("linkedMaterialIds") or []
    for material_id in linked_ids:
        material_task = await get_task(int(material_id))
        if material_task:
            knowledge_blocks += material_task["blocks"]

    return construct_description_from_blocks(knowledge_blocks)


def get_ai_message_for_chat_history(ai_message: dict) -> str:
    try:
        message = json.loads(ai_message) if isinstance(ai_message, str) else ai_message
    except (json.JSONDecodeError, TypeError):
        return str(ai_message)

    if not isinstance(message, dict):
        return str(ai_message)

    feedback = message.get("feedback", "")

    if "scorecard" not in message or not message["scorecard"]:
        return feedback

    scorecard_as_prompt = format_ai_scorecard_report(message["scorecard"])

    if not scorecard_as_prompt:
        return feedback

    return f"""Feedback:\n```\n{feedback}\n```\n\nScorecard:\n```\n{scorecard_as_prompt}\n```"""


async def get_user_details_for_prompt(user_id: str) -> str:
    user_first_name = await get_user_first_name(user_id)

    if not user_first_name:
        return ""

    return f"Name: {user_first_name}"


def _normalise_text(value: Optional[str]) -> str:
    if not value:
        return ""

    return re.sub(r"\s+", " ", value).strip()


def _split_sentences(text: str) -> List[str]:
    text = _normalise_text(text)
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _extract_evidence_from_feedback(feedback_text: str) -> List[Dict]:
    feedback_text = _normalise_text(feedback_text)

    if not feedback_text:
        return []

    evidence: List[Dict] = []

    line_matches = re.findall(r"(?:line\s*\d+|L\d+)", feedback_text, flags=re.IGNORECASE)
    for line_match in line_matches[:2]:
        evidence.append(
            {
                "type": "code_line",
                "reference": line_match,
                "description": feedback_text,
            }
        )

    test_case_matches = re.findall(
        r"(?:test(?:\s*case)?\s*[#:]*\s*[A-Za-z0-9_-]+)",
        feedback_text,
        flags=re.IGNORECASE,
    )
    for test_case_match in test_case_matches[:2]:
        evidence.append(
            {
                "type": "test_case",
                "reference": test_case_match,
                "description": feedback_text,
            }
        )

    quote_matches = re.findall(r'"([^"]{4,120})"|\'([^\']{4,120})\'', feedback_text)
    for quote_match in quote_matches[:2]:
        quote = quote_match[0] or quote_match[1]
        evidence.append(
            {
                "type": "quote",
                "reference": quote,
                "description": feedback_text,
            }
        )

    if not evidence:
        evidence.append(
            {
                "type": "reasoning_gap",
                "reference": "submission",
                "description": feedback_text,
            }
        )

    deduped_evidence = []
    seen_keys = set()

    for item in evidence:
        key = (item["type"], item["reference"].lower())
        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped_evidence.append(item)

    return deduped_evidence[:3]


def _normalise_external_evidence(external_evidence: Optional[List[Dict]]) -> List[Dict]:
    if not external_evidence or not isinstance(external_evidence, list):
        return []

    allowed_types = {"code_line", "test_case", "quote", "reasoning_gap"}
    normalised = []

    for item in external_evidence:
        if not isinstance(item, dict):
            continue

        evidence_type = item.get("type")
        reference = item.get("reference")
        description = _normalise_text(item.get("description"))

        if evidence_type not in allowed_types:
            continue

        if not isinstance(reference, str) or not reference.strip():
            continue

        normalised.append(
            {
                "type": evidence_type,
                "reference": reference.strip(),
                "description": description,
            }
        )

    deduped = []
    seen_keys = set()

    for item in normalised:
        key = (item["type"], item["reference"].lower())
        if key in seen_keys:
            continue

        seen_keys.add(key)
        deduped.append(item)

    return deduped[:5]


def _append_external_evidence_to_submission(
    submission: str,
    external_evidence: List[Dict],
) -> str:
    if not external_evidence:
        return submission

    lines = []
    for item in external_evidence[:3]:
        description = _normalise_text(item.get("description"))
        reference = item.get("reference", "evidence")
        if description:
            lines.append(f"- {reference}: {description}")
        else:
            lines.append(f"- {reference}")

    if not lines:
        return submission

    return (
        f"{submission}\n\n"
        "Execution Evidence (from automated tests/runtime):\n"
        + "\n".join(lines)
    )


def _merge_external_evidence_into_criteria(
    criteria: List[Dict],
    external_evidence: List[Dict],
) -> List[Dict]:
    if not criteria or not external_evidence:
        return criteria

    test_case_evidence = [
        evidence
        for evidence in external_evidence
        if evidence.get("type") == "test_case"
    ]

    if not test_case_evidence:
        return criteria

    keyword_matches = (
        "correct",
        "logic",
        "output",
        "test",
        "function",
        "implementation",
    )

    target_index = 0
    for index, criterion in enumerate(criteria):
        criterion_name = str(criterion.get("criterion_name", "")).lower()
        if any(keyword in criterion_name for keyword in keyword_matches):
            target_index = index
            break

    existing_evidence = criteria[target_index].get("evidence", []) or []
    combined = existing_evidence + test_case_evidence

    deduped = []
    seen = set()
    for item in combined:
        key = (
            item.get("type", ""),
            str(item.get("reference", "")).lower(),
        )
        if key in seen:
            continue

        seen.add(key)
        deduped.append(item)

    criteria[target_index]["evidence"] = deduped[:3]
    return criteria


def _derive_severity(score: float, max_score: float, pass_score: float) -> str:
    if score < pass_score:
        return "high"

    if score < max_score:
        return "medium"

    return "low"


def _derive_next_step(
    wrong_feedback: str,
    criterion_name: str,
    score: float,
    pass_score: float,
) -> str:
    wrong_feedback = _normalise_text(wrong_feedback)

    if wrong_feedback:
        wrong_feedback_sentences = _split_sentences(wrong_feedback)
        if wrong_feedback_sentences:
            return wrong_feedback_sentences[0]

    if score < pass_score:
        return (
            f"Rework {criterion_name} using the evidence above, then resubmit with a clearer explanation."
        )

    return f"Keep strengthening {criterion_name} with one concrete example from your response."


def _normalise_scorecard_feedback(
    feedback_summary: str,
    scorecard: Dict,
    external_evidence: Optional[List[Dict]] = None,
) -> Dict:
    criteria = []

    for criterion_name, criterion_data in scorecard.items():
        feedback = criterion_data.get("feedback", {}) or {}
        score = float(criterion_data.get("score", 0))
        max_score = float(criterion_data.get("max_score", 0))
        pass_score = float(criterion_data.get("pass_score", max_score))

        wrong_feedback = _normalise_text(feedback.get("wrong"))
        correct_feedback = _normalise_text(feedback.get("correct"))

        evidence_source = wrong_feedback or correct_feedback or feedback_summary
        evidence = _extract_evidence_from_feedback(evidence_source)

        next_step = _derive_next_step(
            wrong_feedback,
            criterion_name,
            score,
            pass_score,
        )

        severity = _derive_severity(score, max_score, pass_score)

        criteria.append(
            {
                "criterion_name": criterion_name,
                "score": score,
                "max_score": max_score,
                "pass_score": pass_score,
                "evidence": evidence,
                "next_step": next_step,
                "severity": severity,
            }
        )

    criteria = _merge_external_evidence_into_criteria(
        criteria,
        _normalise_external_evidence(external_evidence),
    )

    total_score = sum(criterion["score"] for criterion in criteria)
    total_max_score = sum(criterion["max_score"] for criterion in criteria)
    overall_score = round((total_score / total_max_score) * 100, 2) if total_max_score else 0

    severity_order = {"high": 0, "medium": 1, "low": 2}
    surfaced_criteria = sorted(
        criteria,
        key=lambda criterion: (
            severity_order.get(criterion["severity"], 3),
            criterion["score"],
        ),
    )[:3]

    return {
        "feedback_summary": _normalise_text(feedback_summary),
        "criteria": criteria,
        "overall_score": overall_score,
        "surfaced_criteria": surfaced_criteria,
    }


def _build_rubric_metadata(rubric: Dict) -> Dict:
    rubric_json = json.dumps(rubric, sort_keys=True)
    rubric_hash = hashlib.sha256(rubric_json.encode("utf-8")).hexdigest()
    rubric_version = f"v1:{rubric_hash[:12]}"

    return {
        "rubric_hash": rubric_hash,
        "rubric_version": rubric_version,
    }


async def _persist_feedback_attempt(
    *,
    user_id: int,
    task_id: int,
    question_id: Optional[int],
    model_used: str,
    rubric: Dict,
    normalized_feedback: Dict,
) -> int:
    rubric_metadata = _build_rubric_metadata(rubric)

    attempt_id = await create_feedback_attempt(
        user_id=user_id,
        task_id=task_id,
        question_id=question_id,
        rubric_version=rubric_metadata["rubric_version"],
        rubric_hash=rubric_metadata["rubric_hash"],
        model_used=model_used,
        overall_score=normalized_feedback["overall_score"],
        feedback_summary=normalized_feedback["feedback_summary"],
        raw_feedback_json=normalized_feedback,
    )

    await create_feedback_items(
        attempt_id,
        normalized_feedback.get("criteria", []),
    )

    return attempt_id


async def _build_diff_from_previous_attempt(
    *,
    user_id: int,
    task_id: int,
    question_id: Optional[int],
    current_attempt_id: int,
    current_criteria: List[Dict],
) -> List[Dict]:
    previous_attempt = await get_previous_feedback_attempt(
        user_id=user_id,
        task_id=task_id,
        question_id=question_id,
        current_attempt_id=current_attempt_id,
    )

    if not previous_attempt:
        return []

    previous_items = await get_feedback_items_for_attempt(previous_attempt["id"])
    previous_score_by_criterion = {
        item["criterion_name"]: item["score"] for item in previous_items
    }

    diff = []

    for criterion in current_criteria:
        criterion_name = criterion["criterion_name"]
        current_score = float(criterion["score"])
        previous_score = previous_score_by_criterion.get(criterion_name)

        if previous_score is None:
            change = "new"
        elif current_score > previous_score:
            change = "improved"
        elif current_score < previous_score:
            change = "regressed"
        else:
            change = "unchanged"

        diff.append(
            {
                "criterion_name": criterion_name,
                "previous_score": previous_score,
                "current_score": current_score,
                "change": change,
            }
        )

    return diff


def _extract_latest_user_submission(chat_history: List[Dict]) -> Optional[str]:
    for message in reversed(chat_history):
        if message.get("role") != "user":
            continue

        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content

    return None


@router.post("/chat")
async def ai_response_for_question(request: AIChatRequest):
    # Define an async generator for streaming
    async def stream_response() -> AsyncGenerator[str, None]:
        with langfuse.start_as_current_span(
            name="ai_chat",
        ) as trace:
            metadata = {
                "task_id": request.task_id,
                "user_id": request.user_id,
                "user_email": request.user_email,
            }

            user_details = await get_user_details_for_prompt(request.user_id)

            if request.task_type == TaskType.QUIZ:
                if request.question_id is None and request.question is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Question ID or question is required for {request.task_type} tasks",
                    )

                if request.question_id is not None and request.user_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail="User ID is required when question ID is provided",
                    )

                if request.question and request.chat_history is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Chat history is required when question is provided",
                    )
                if request.question_id is None:
                    session_id = f"quiz_{request.task_id}_preview_{request.user_id}"
                else:
                    session_id = f"quiz_{request.task_id}_{request.question_id}_{request.user_id}"
            else:
                if request.task_id is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Task ID is required for learning material tasks",
                    )

                if request.chat_history is None:
                    raise HTTPException(
                        status_code=400,
                        detail="Chat history is required for learning material tasks",
                    )
                session_id = f"lm_{request.task_id}_{request.user_id}"

            task = await get_task(request.task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            metadata["task_title"] = task["title"]

            new_user_message = [
                {
                    "role": "user",
                    "content": (
                        get_user_audio_message_for_chat_history(request.user_response)
                        if request.response_type == ChatResponseType.AUDIO
                        else request.user_response
                    ),
                }
            ]

            external_code_evidence = _normalise_external_evidence(request.code_evidence)
            if (
                request.response_type == ChatResponseType.CODE
                and external_code_evidence
                and isinstance(new_user_message[0]["content"], str)
            ):
                new_user_message[0]["content"] = _append_external_evidence_to_submission(
                    new_user_message[0]["content"],
                    external_code_evidence,
                )

            if request.task_type == TaskType.LEARNING_MATERIAL:
                if request.response_type == ChatResponseType.AUDIO:
                    raise HTTPException(
                        status_code=400,
                        detail="Audio response is not supported for learning material tasks",
                    )

                metadata["type"] = "learning_material"

                chat_history = request.chat_history

                chat_history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in chat_history
                ]

                reference_material = construct_description_from_blocks(task["blocks"])

                rewritten_query = await rewrite_query(
                    chat_history + new_user_message, reference_material
                )

                # update the last user message with the rewritten query
                new_user_message[0]["content"] = rewritten_query

                question_details = f"**Reference Material**\n\n{reference_material}\n\n"
            else:
                metadata["type"] = "quiz"

                if request.question_id:
                    question = await get_question(request.question_id)
                    if not question:
                        raise HTTPException(
                            status_code=404, detail="Question not found"
                        )

                    metadata["question_id"] = request.question_id

                    chat_history = await get_question_chat_history_for_user(
                        request.question_id, request.user_id
                    )

                else:
                    question = request.question.model_dump()
                    chat_history = request.chat_history

                    question["scorecard"] = await get_scorecard(
                        question["scorecard_id"]
                    )
                    metadata["question_id"] = None

                chat_history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in chat_history
                ]

                metadata["question_title"] = question["title"]
                metadata["question_type"] = question["type"]
                metadata["question_purpose"] = (
                    "practice" if question["response_type"] == "chat" else "exam"
                )
                metadata["question_input_type"] = question["input_type"]
                metadata["question_has_context"] = bool(question["context"])

                question_description = construct_description_from_blocks(
                    question["blocks"]
                )
                question_details = f"**Task**\n\n{question_description}\n\n"

            task_metadata = await get_task_metadata(request.task_id)
            if task_metadata:
                metadata.update(task_metadata)

            for message in chat_history:
                if message["role"] == "user":
                    if (
                        request.response_type == ChatResponseType.AUDIO
                        and message.get("response_type") == ChatResponseType.AUDIO
                    ):
                        message["content"] = get_user_audio_message_for_chat_history(
                            message["content"]
                        )
                else:
                    if request.task_type == TaskType.LEARNING_MATERIAL:
                        message["content"] = json.dumps(
                            {"feedback": message["content"]}
                        )

                    message["content"] = get_ai_message_for_chat_history(
                        message["content"]
                    )

            if request.task_type == TaskType.QUIZ:
                if question["type"] == QuestionType.OBJECTIVE:
                    answer_as_prompt = construct_description_from_blocks(
                        question["answer"]
                    )
                    question_details += f"---\n\n**Reference Solution (never to be shared with the learner)**\n\n{answer_as_prompt}\n\n"
                else:
                    scorecard_as_prompt = convert_scorecard_to_prompt(
                        question["scorecard"]
                    )
                    question_details += (
                        f"---\n\n**Scoring Criteria**\n\n{scorecard_as_prompt}\n\n"
                    )

            chat_history = chat_history + new_user_message

            # router
            if request.response_type == ChatResponseType.AUDIO:
                model = openai_plan_to_model_name["audio"]
                openai_api_mode = "chat_completions"
            else:
                model = await get_model_for_task(chat_history, question_details)
                openai_api_mode = "responses"

            # response
            llm_input = f"""# Chat History\n\n{format_chat_history_with_audio(chat_history)}\n\n# Task Details\n\n{question_details}"""
            response_metadata = {
                "input": llm_input,
            }

            metadata.update(response_metadata)

            llm_output = ""
            if request.task_type == TaskType.QUIZ:
                if question["type"] == QuestionType.OBJECTIVE:

                    class Output(BaseModel):
                        analysis: str = Field(
                            description="A detailed analysis of the student's response"
                        )
                        feedback: str = Field(
                            description="Feedback on the student's response; add newline characters to the feedback to make it more readable where necessary; address the student by name if their name has been provided."
                        )
                        is_correct: bool = Field(
                            description="Whether the student's response correctly solves the original task that the student is supposed to solve. For this to be true, the original task needs to be completely solved and not just partially solved. Giving the right answer to one step of the task does not count as solving the entire task."
                        )

                else:

                    class Feedback(BaseModel):
                        correct: Optional[str] = Field(
                            description="What worked well in the student's response for this category based on the scoring criteria"
                        )
                        wrong: Optional[str] = Field(
                            description="What needs improvement in the student's response for this category based on the scoring criteria"
                        )

                    class Row(BaseModel):
                        feedback: Feedback = Field(
                            description="Detailed feedback for the student's response for this category"
                        )
                        score: float = Field(
                            description="Score given within the min/max range for this category based on the student's response - the score given should be in alignment with the feedback provided"
                        )
                        max_score: float = Field(
                            description="Maximum score possible for this category as per the scoring criteria"
                        )
                        pass_score: float = Field(
                            description="Pass score possible for this category as per the scoring criteria"
                        )

                    def make_scorecard_model(fields: list[str]) -> type[BaseModel]:
                        """
                        Dynamically create a Pydantic model with fields from a list of strings.
                        Each field defaults to `str`, but you can change that if needed.
                        """
                        # build dictionary for create_model
                        field_definitions: dict[str, tuple[type, any]] = {
                            field: (Row, ...) for field in fields
                        }
                        # ... means "required"
                        return create_model("Scorecard", **field_definitions)

                    Scorecard = make_scorecard_model(
                        [
                            criterion["name"].replace('"', "")
                            for criterion in question["scorecard"]["criteria"]
                        ]
                    )

                    class Output(BaseModel):
                        chain_of_thought: str = Field(
                            description="Concise analysis of the student's response and what the scorecard should be."
                        )
                        feedback: str = Field(
                            description="A single, comprehensive summary based on the scoring criteria; address the student by name if their name has been provided."
                        )
                        scorecard: Optional[Scorecard] = Field(
                            description="Score and feedback for each criterion from the scoring criteria; only include this in the response if the student's response is a valid response to the task"
                        )

            else:

                class Output(BaseModel):
                    response: str = Field(
                        description="Response to the student's query; add proper formatting to the response to make it more readable where necessary; address the student by name if their name has been provided."
                    )

            if request.task_type == TaskType.QUIZ:
                knowledge_base = await build_knowledge_base_from_context(
                    question.get("context")
                )

                if knowledge_base:
                    question_details += (
                        f"---\n\n**Knowledge Base**\n\n{knowledge_base}\n\n"
                    )

                if question["type"] == QuestionType.OBJECTIVE:
                    prompt_name = "objective-question"
                    messages = compile_prompt(
                        OBJECTIVE_QUESTION_SYSTEM_PROMPT,
                        OBJECTIVE_QUESTION_USER_PROMPT,
                        task_details=question_details,
                        user_details=user_details,
                    )
                else:
                    prompt_name = "subjective-question"
                    messages = compile_prompt(
                        SUBJECTIVE_QUESTION_SYSTEM_PROMPT,
                        SUBJECTIVE_QUESTION_USER_PROMPT,
                        task_details=question_details,
                        user_details=user_details,
                    )
            else:
                prompt_name = "doubt_solving"
                messages = compile_prompt(
                    DOUBT_SOLVING_SYSTEM_PROMPT,
                    DOUBT_SOLVING_USER_PROMPT,
                    reference_material=question_details,
                    user_details=user_details,
                )

            messages += chat_history

            with langfuse.start_as_current_observation(
                as_type="generation", name="response"
            ) as observation:
                try:
                    async for chunk in stream_llm_with_openai(
                        model=model,
                        messages=messages,
                        response_model=Output,
                        max_output_tokens=8192,
                        api_mode=openai_api_mode,
                    ):
                        content = json.dumps(chunk.model_dump()) + "\n"
                        llm_output = chunk.model_dump()
                        yield content
                except Exception as e:
                    # Check if it's the specific AsyncStream aclose error
                    if str(e) == "'AsyncStream' object has no attribute 'aclose'":
                        # Silently end partial stream on this specific error
                        pass
                    else:
                        # Re-raise other exceptions
                        raise
                finally:
                    observation.update(
                        input=llm_input,
                        output=llm_output,
                        metadata={
                            "prompt_name": prompt_name,
                            **response_metadata,
                        },
                    )

            if (
                request.task_type == TaskType.QUIZ
                and request.question_id is not None
                and question["type"] != QuestionType.OBJECTIVE
                and isinstance(llm_output, dict)
                and llm_output.get("scorecard")
                and question.get("scorecard")
            ):
                try:
                    normalized_feedback = _normalise_scorecard_feedback(
                        feedback_summary=llm_output.get("feedback", ""),
                        scorecard=llm_output["scorecard"],
                        external_evidence=external_code_evidence,
                    )

                    attempt_id = await _persist_feedback_attempt(
                        user_id=request.user_id,
                        task_id=request.task_id,
                        question_id=request.question_id,
                        model_used=model,
                        rubric=question["scorecard"],
                        normalized_feedback=normalized_feedback,
                    )

                    normalized_feedback["attempt_id"] = attempt_id

                    diff_from_previous = await _build_diff_from_previous_attempt(
                        user_id=request.user_id,
                        task_id=request.task_id,
                        question_id=request.question_id,
                        current_attempt_id=attempt_id,
                        current_criteria=normalized_feedback["criteria"],
                    )

                    llm_output["attempt_id"] = attempt_id
                    llm_output["feedback_output"] = normalized_feedback
                    llm_output["diff_from_previous"] = diff_from_previous

                    yield json.dumps(
                        {
                            "attempt_id": attempt_id,
                            "feedback_output": normalized_feedback,
                            "diff_from_previous": diff_from_previous,
                        }
                    ) + "\n"
                except Exception:
                    logging.exception("Failed to persist normalized feedback for quiz chat")

            metadata["output"] = llm_output
            trace.update_trace(
                user_id=str(request.user_id),
                session_id=session_id,
                metadata=metadata,
                input=llm_input,
                output=llm_output,
            )

    # Return a streaming response
    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )


@router.post("/assignment")
async def ai_response_for_assignment(request: AIChatRequest):
    # Define an async generator for streaming
    async def stream_response() -> AsyncGenerator[str, None]:
        with langfuse.start_as_current_span(
            name="assignment_evaluation",
        ) as trace:
            metadata = {
                "task_id": request.task_id,
                "user_id": request.user_id,
                "user_email": request.user_email,
            }

            user_details = await get_user_details_for_prompt(request.user_id)

            # Validate required fields for assignment
            if request.task_id is None:
                raise HTTPException(
                    status_code=400,
                    detail="Task ID is required for assignment tasks",
                )

            # Get assignment data
            task = await get_task(request.task_id)
            if not task:
                raise HTTPException(status_code=404, detail="Task not found")

            if task["type"] != TaskType.ASSIGNMENT:
                raise HTTPException(status_code=400, detail="Task is not an assignment")

            metadata["task_title"] = task["title"]

            assignment = task["assignment"]
            problem_blocks = assignment["blocks"]
            evaluation_criteria = assignment["evaluation_criteria"]

            if not evaluation_criteria:
                raise HTTPException(
                    status_code=400,
                    detail="Assignment is missing evaluation criteria",
                )

            if not evaluation_criteria.get("scorecard_id"):
                raise HTTPException(
                    status_code=400,
                    detail="Assignment evaluation criteria is missing scorecard_id",
                )

            context = assignment.get("context")

            # Get scorecard for evaluation
            scorecard = await get_scorecard(evaluation_criteria["scorecard_id"])

            if not scorecard:
                raise HTTPException(
                    status_code=400,
                    detail="Scorecard not found for assignment evaluation criteria",
                )

            # Get chat history for this assignment
            # Use request.chat_history if provided (for preview mode), otherwise fetch from database
            if request.chat_history:
                chat_history = request.chat_history

                if chat_history is None:
                    # For first-time submissions (file uploads), chat_history might be empty
                    # We'll initialize it as empty if not provided
                    chat_history = []
            else:
                chat_history = await get_task_chat_history_for_user(
                    request.task_id, request.user_id
                )

            # Convert chat history to the format expected by AI
            formatted_chat_history = [
                {"role": message["role"], "content": message["content"]}
                for message in chat_history
            ]

            # Add new user message
            new_user_message = [
                {
                    "role": "user",
                    "content": (
                        get_user_audio_message_for_chat_history(request.user_response)
                        if request.response_type == ChatResponseType.AUDIO
                        else request.user_response
                    ),
                }
            ]

            external_code_evidence = _normalise_external_evidence(request.code_evidence)
            if (
                request.response_type == ChatResponseType.CODE
                and external_code_evidence
                and isinstance(new_user_message[0]["content"], str)
            ):
                new_user_message[0]["content"] = _append_external_evidence_to_submission(
                    new_user_message[0]["content"],
                    external_code_evidence,
                )

            # Build problem statement from blocks
            problem_statement = construct_description_from_blocks(problem_blocks)

            # Build full chat history
            if request.response_type == ChatResponseType.FILE:
                # For file uploads, include only the new user message with file_uuid
                # This branch is triggered when a learner submits an assignment as a file
                full_chat_history = new_user_message
            else:
                # This branch is triggered when a learner answers questions about the assignment with text or audio
                full_chat_history = formatted_chat_history + new_user_message

            # Handle file submission - extract code
            submission_data = None

            if request.response_type == ChatResponseType.FILE:
                submission_data = extract_submission_file(request.user_response)
            else:
                # Not a file upload, check chat history for latest file submission
                latest_file_uuid = get_latest_file_uuid_from_chat_history(
                    full_chat_history
                )

                if latest_file_uuid:
                    submission_data = extract_submission_file(latest_file_uuid)

            # Build evaluation context
            evaluation_context = build_evaluation_context(evaluation_criteria)

            # Build Key Areas section from scorecard
            key_areas_section = f"\n\n<Key Areas>\n\n{convert_scorecard_to_prompt(scorecard)}\n\n</Key Areas>"

            # Build context with linked materials if available
            knowledge_base = await build_knowledge_base_from_context(context)

            # Build the complete assignment context
            assignment_details = (
                f"<Problem Statement>\n\n{problem_statement}\n\n</Problem Statement>"
            )

            # Add Key Areas from scorecard
            if key_areas_section:
                assignment_details += key_areas_section

            if evaluation_context:
                assignment_details += f"\n\n<Evaluation Criteria>\n\n{evaluation_context}\n\n</Evaluation Criteria>"

            if knowledge_base:
                assignment_details += (
                    f"\n\n<Knowledge Base>\n\n{knowledge_base}\n\n</Knowledge Base>"
                )

            # Add submission data for file uploads
            if submission_data:
                assignment_details += f"\n\n<Student Submission Data>"
                assignment_details += f"\n\n**File Contents:**\n"
                for filename, content in submission_data["file_contents"].items():
                    assignment_details += (
                        f"\n--- {filename} ---\n{content}\n--- End of {filename} ---\n"
                    )
                assignment_details += f"\n\n</Student Submission Data>"

            # Process chat history for audio content if needed
            if request.response_type == ChatResponseType.AUDIO:
                for message in full_chat_history:
                    if (
                        message["role"] == "user"
                        and message.get("response_type") == ChatResponseType.AUDIO
                    ):
                        message["content"] = get_user_audio_message_for_chat_history(
                            message["content"]
                        )

            # Determine model based on input type
            if request.response_type == ChatResponseType.AUDIO:
                model = openai_plan_to_model_name["audio"]
                openai_api_mode = "chat_completions"
            else:
                # For assignments, use reasoning model for better evaluation
                model = openai_plan_to_model_name["reasoning"]
                openai_api_mode = "responses"

            # Enhanced feedback structure for key area scores
            class Feedback(BaseModel):
                correct: Optional[str] = Field(
                    description="What worked well in the student's response for this category based on the scoring criteria"
                )
                wrong: Optional[str] = Field(
                    description="What needs improvement in the student's response for this category based on the scoring criteria"
                )

            class KeyAreaScore(BaseModel):
                feedback: Feedback = Field(
                    description="Detailed feedback for the student's response for this category"
                )
                score: float = Field(
                    description="Score given within the min/max range for this category based on the student's response - the score given should be in alignment with the feedback provided"
                )
                max_score: float = Field(
                    description="Maximum score possible for this category as per the scoring criteria"
                )
                pass_score: float = Field(
                    description="Pass score possible for this category as per the scoring criteria"
                )

            # Base output model for all phases
            class Output(BaseModel):
                chain_of_thought: str = Field(
                    description="Concise analysis of the student's response to the question asked and what the evaluation result should be"
                )
                feedback: Optional[str] = Field(
                    description="A single, comprehensive summary based on the scoring criteria; address the student by name if their name has been provided.",
                )
                evaluation_status: Optional[str] = Field(
                    description="The status of the evaluation; can be `in_progress`, `needs_resubmission`, or `completed`",
                )
                key_area_scores: Optional[Dict[str, KeyAreaScore]] = Field(
                    description="Completed key area scores with detailed feedback",
                    default={},
                )
                current_key_area: Optional[str] = Field(
                    description="Current key area being evaluated"
                )

            # Output model for file submissions that includes project score
            class FileSubmissionOutput(Output):
                chain_of_thought: str = Field(
                    description="Concise analysis of the student's submission to the assignment and what the evaluation result should be"
                )
                assignment_score: Optional[float] = Field(
                    description="Assignment score assigned when evaluating initial file submission"
                )

            messages = compile_prompt(
                ASSIGNMENT_SYSTEM_PROMPT,
                ASSIGNMENT_USER_PROMPT,
                assignment_details=assignment_details,
                user_details=user_details,
            )

            messages += full_chat_history

            # Build input for metadata
            llm_input = f"""`Assignment Details`:\n\n{assignment_details}\n\n`Chat History`:\n\n{format_chat_history_with_audio(full_chat_history)}"""
            response_metadata = {
                "input": llm_input,
            }

            metadata.update(response_metadata)

            llm_output = ""

            # Use FileSubmissionOutput for file submissions, otherwise use base Output
            response_model = (
                FileSubmissionOutput
                if request.response_type == ChatResponseType.FILE
                else Output
            )

            with langfuse.start_as_current_observation(
                as_type="generation", name="response"
            ) as observation:
                try:
                    async for chunk in stream_llm_with_openai(
                        model=model,
                        messages=messages,
                        response_model=response_model,
                        max_output_tokens=8192,
                        api_mode=openai_api_mode,
                    ):
                        content = json.dumps(chunk.model_dump()) + "\n"
                        llm_output = chunk.model_dump()
                        yield content
                except Exception as e:
                    # Check if it's the specific AsyncStream aclose error
                    if str(e) == "'AsyncStream' object has no attribute 'aclose'":
                        # Silently end partial stream on this specific error
                        pass
                    else:
                        # Re-raise other exceptions
                        raise
                finally:
                    observation.update(
                        input=llm_input,
                        output=llm_output,
                        metadata={
                            "prompt_name": "assignment",
                            **response_metadata,
                        },
                    )

            session_id = f"assignment_{request.task_id}_{request.user_id}"

            if (
                isinstance(llm_output, dict)
                and llm_output.get("key_area_scores")
                and isinstance(llm_output["key_area_scores"], dict)
            ):
                try:
                    normalized_feedback = _normalise_scorecard_feedback(
                        feedback_summary=llm_output.get("feedback", ""),
                        scorecard=llm_output["key_area_scores"],
                        external_evidence=external_code_evidence,
                    )

                    attempt_id = await _persist_feedback_attempt(
                        user_id=request.user_id,
                        task_id=request.task_id,
                        question_id=None,
                        model_used=model,
                        rubric=scorecard,
                        normalized_feedback=normalized_feedback,
                    )

                    normalized_feedback["attempt_id"] = attempt_id

                    diff_from_previous = await _build_diff_from_previous_attempt(
                        user_id=request.user_id,
                        task_id=request.task_id,
                        question_id=None,
                        current_attempt_id=attempt_id,
                        current_criteria=normalized_feedback["criteria"],
                    )

                    llm_output["attempt_id"] = attempt_id
                    llm_output["feedback_output"] = normalized_feedback
                    llm_output["diff_from_previous"] = diff_from_previous

                    yield json.dumps(
                        {
                            "attempt_id": attempt_id,
                            "feedback_output": normalized_feedback,
                            "diff_from_previous": diff_from_previous,
                        }
                    ) + "\n"
                except Exception:
                    logging.exception(
                        "Failed to persist normalized feedback for assignment evaluation"
                    )

            metadata["output"] = llm_output
            trace.update_trace(
                user_id=str(request.user_id),
                session_id=session_id,
                metadata=metadata,
                input=llm_input,
                output=llm_output,
            )

    # Return a streaming response
    return StreamingResponse(
        stream_response(),
        media_type="application/x-ndjson",
    )


def _build_subjective_output_model(scorecard: Dict) -> type[BaseModel]:
    class Feedback(BaseModel):
        correct: Optional[str] = Field(
            description="What worked well in the student's response for this category based on the scoring criteria"
        )
        wrong: Optional[str] = Field(
            description="What needs improvement in the student's response for this category based on the scoring criteria"
        )

    class Row(BaseModel):
        feedback: Feedback = Field(
            description="Detailed feedback for the student's response for this category"
        )
        score: float = Field(
            description="Score given within the min/max range for this category based on the student's response"
        )
        max_score: float = Field(
            description="Maximum score possible for this category as per the scoring criteria"
        )
        pass_score: float = Field(
            description="Pass score possible for this category as per the scoring criteria"
        )

    scorecard_model = create_model(
        "Scorecard",
        **{
            criterion["name"].replace('"', ""): (Row, ...)
            for criterion in scorecard["criteria"]
        },
    )

    class Output(BaseModel):
        chain_of_thought: str = Field(
            description="Concise analysis of the student's response and what the scorecard should be."
        )
        feedback: str = Field(
            description="A single, comprehensive summary based on the scoring criteria."
        )
        scorecard: Optional[scorecard_model] = Field(
            description="Score and feedback for each criterion from the scoring criteria"
        )

    return Output


async def _re_evaluate_subjective_quiz(
    request: ReevaluateFeedbackRequest,
) -> Dict:
    if request.question_id is None:
        raise HTTPException(
            status_code=400,
            detail="question_id is required for quiz re-evaluation",
        )

    question = await get_question(request.question_id)
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    if question["type"] == QuestionType.OBJECTIVE:
        raise HTTPException(
            status_code=400,
            detail="Re-evaluation is currently supported for subjective quiz questions only",
        )

    if not question.get("scorecard"):
        raise HTTPException(
            status_code=400,
            detail="Question is missing scorecard configuration",
        )

    chat_history = await get_question_chat_history_for_user(
        request.question_id,
        request.user_id,
    )

    latest_submission = request.latest_submission or _extract_latest_user_submission(
        chat_history
    )

    if not latest_submission:
        raise HTTPException(
            status_code=400,
            detail="No learner submission found for re-evaluation",
        )

    external_code_evidence = _normalise_external_evidence(request.code_evidence)

    prompt_chat_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in chat_history
    ]

    for message in prompt_chat_history:
        if message["role"] != "assistant":
            continue

        message["content"] = get_ai_message_for_chat_history(message["content"])

    prompt_chat_history.append(
        {
            "role": "user",
            "content": _append_external_evidence_to_submission(
                latest_submission,
                external_code_evidence,
            ),
        }
    )

    question_description = construct_description_from_blocks(question["blocks"])
    question_details = f"**Task**\n\n{question_description}\n\n"

    scorecard_as_prompt = convert_scorecard_to_prompt(question["scorecard"])
    question_details += f"---\n\n**Scoring Criteria**\n\n{scorecard_as_prompt}\n\n"

    knowledge_base = await build_knowledge_base_from_context(question.get("context"))
    if knowledge_base:
        question_details += f"---\n\n**Knowledge Base**\n\n{knowledge_base}\n\n"

    user_details = await get_user_details_for_prompt(request.user_id)
    messages = compile_prompt(
        SUBJECTIVE_QUESTION_SYSTEM_PROMPT,
        SUBJECTIVE_QUESTION_USER_PROMPT,
        task_details=question_details,
        user_details=user_details,
    )
    messages += prompt_chat_history

    model = await get_model_for_task(prompt_chat_history, question_details)
    output_model = _build_subjective_output_model(question["scorecard"])

    llm_output = await run_llm_with_openai(
        model=model,
        messages=messages,
        response_model=output_model,
        max_output_tokens=8192,
    )

    llm_output = llm_output.model_dump()

    if not llm_output.get("scorecard"):
        raise HTTPException(
            status_code=422,
            detail="AI response is missing scorecard. Please try re-evaluating again.",
        )

    normalized_feedback = _normalise_scorecard_feedback(
        feedback_summary=llm_output.get("feedback", ""),
        scorecard=llm_output["scorecard"],
        external_evidence=external_code_evidence,
    )

    attempt_id = await _persist_feedback_attempt(
        user_id=request.user_id,
        task_id=request.task_id,
        question_id=request.question_id,
        model_used=model,
        rubric=question["scorecard"],
        normalized_feedback=normalized_feedback,
    )

    normalized_feedback["attempt_id"] = attempt_id

    diff_from_previous = await _build_diff_from_previous_attempt(
        user_id=request.user_id,
        task_id=request.task_id,
        question_id=request.question_id,
        current_attempt_id=attempt_id,
        current_criteria=normalized_feedback["criteria"],
    )

    return {
        "attempt_id": attempt_id,
        "feedback_summary": normalized_feedback["feedback_summary"],
        "overall_score": normalized_feedback["overall_score"],
        "criteria": normalized_feedback["criteria"],
        "diff_from_previous": diff_from_previous,
        "feedback": llm_output.get("feedback"),
        "scorecard": llm_output.get("scorecard"),
    }


def _extract_file_uuid_from_submission(submission: str) -> Optional[str]:
    if not submission:
        return None

    try:
        payload = json.loads(submission)
        if isinstance(payload, dict) and payload.get("file_uuid"):
            return payload["file_uuid"]
    except (json.JSONDecodeError, TypeError):
        return None

    return None


async def _re_evaluate_assignment(
    request: ReevaluateFeedbackRequest,
) -> Dict:
    task = await get_task(request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task["type"] != TaskType.ASSIGNMENT:
        raise HTTPException(status_code=400, detail="Task is not an assignment")

    assignment = task.get("assignment") or {}
    evaluation_criteria = assignment.get("evaluation_criteria")

    if not evaluation_criteria:
        raise HTTPException(
            status_code=400,
            detail="Assignment is missing evaluation criteria",
        )

    scorecard_id = evaluation_criteria.get("scorecard_id")
    if not scorecard_id:
        raise HTTPException(
            status_code=400,
            detail="Assignment evaluation criteria is missing scorecard_id",
        )

    scorecard = await get_scorecard(scorecard_id)
    if not scorecard:
        raise HTTPException(
            status_code=400,
            detail="Scorecard not found for assignment evaluation criteria",
        )

    chat_history = await get_task_chat_history_for_user(request.task_id, request.user_id)

    latest_submission = request.latest_submission or _extract_latest_user_submission(
        chat_history
    )

    if not latest_submission:
        raise HTTPException(
            status_code=400,
            detail="No learner submission found for re-evaluation",
        )

    external_code_evidence = _normalise_external_evidence(request.code_evidence)

    prompt_chat_history = [
        {
            "role": message["role"],
            "content": message["content"],
        }
        for message in chat_history
    ]

    for message in prompt_chat_history:
        if message["role"] != "assistant":
            continue

        try:
            message["content"] = get_ai_message_for_chat_history(message["content"])
        except Exception:
            continue

    if request.latest_submission:
        prompt_chat_history.append(
            {
                "role": "user",
                "content": _append_external_evidence_to_submission(
                    request.latest_submission,
                    external_code_evidence,
                ),
            }
        )

    problem_statement = construct_description_from_blocks(assignment.get("blocks", []))
    evaluation_context = build_evaluation_context(evaluation_criteria)

    key_areas_section = (
        f"\n\n<Key Areas>\n\n{convert_scorecard_to_prompt(scorecard)}\n\n</Key Areas>"
    )

    context = assignment.get("context")
    knowledge_base = await build_knowledge_base_from_context(context)

    assignment_details = (
        f"<Problem Statement>\n\n{problem_statement}\n\n</Problem Statement>"
    )

    if key_areas_section:
        assignment_details += key_areas_section

    if evaluation_context:
        assignment_details += (
            f"\n\n<Evaluation Criteria>\n\n{evaluation_context}\n\n</Evaluation Criteria>"
        )

    if knowledge_base:
        assignment_details += (
            f"\n\n<Knowledge Base>\n\n{knowledge_base}\n\n</Knowledge Base>"
        )

    submission_data = None
    file_uuid = _extract_file_uuid_from_submission(latest_submission)
    if file_uuid:
        submission_data = extract_submission_file(file_uuid)

    if submission_data:
        assignment_details += "\n\n<Student Submission Data>"
        assignment_details += "\n\n**File Contents:**\n"
        for filename, content in submission_data["file_contents"].items():
            assignment_details += (
                f"\n--- {filename} ---\n{content}\n--- End of {filename} ---\n"
            )
        assignment_details += "\n\n</Student Submission Data>"

    class Feedback(BaseModel):
        correct: Optional[str] = Field(
            description="What worked well in the student's response for this category based on the scoring criteria"
        )
        wrong: Optional[str] = Field(
            description="What needs improvement in the student's response for this category based on the scoring criteria"
        )

    class KeyAreaScore(BaseModel):
        feedback: Feedback = Field(
            description="Detailed feedback for the student's response for this category"
        )
        score: float = Field(
            description="Score given within the min/max range for this category based on the student's response"
        )
        max_score: float = Field(
            description="Maximum score possible for this category as per the scoring criteria"
        )
        pass_score: float = Field(
            description="Pass score possible for this category as per the scoring criteria"
        )

    class Output(BaseModel):
        chain_of_thought: str = Field(
            description="Concise analysis of the student's response"
        )
        feedback: Optional[str] = Field(
            description="A single, comprehensive summary based on the scoring criteria.",
        )
        evaluation_status: Optional[str] = Field(
            description="The status of the evaluation",
        )
        key_area_scores: Optional[Dict[str, KeyAreaScore]] = Field(
            description="Completed key area scores with detailed feedback",
            default={},
        )
        current_key_area: Optional[str] = Field(
            description="Current key area being evaluated"
        )

    user_details = await get_user_details_for_prompt(request.user_id)
    messages = compile_prompt(
        ASSIGNMENT_SYSTEM_PROMPT,
        ASSIGNMENT_USER_PROMPT,
        assignment_details=assignment_details,
        user_details=user_details,
    )
    messages += prompt_chat_history

    model = openai_plan_to_model_name["reasoning"]

    llm_output = await run_llm_with_openai(
        model=model,
        messages=messages,
        response_model=Output,
        max_output_tokens=8192,
    )

    llm_output = llm_output.model_dump()

    if not llm_output.get("key_area_scores"):
        raise HTTPException(
            status_code=422,
            detail="AI response is missing key area scores. Please try re-evaluating again.",
        )

    normalized_feedback = _normalise_scorecard_feedback(
        feedback_summary=llm_output.get("feedback", ""),
        scorecard=llm_output["key_area_scores"],
        external_evidence=external_code_evidence,
    )

    attempt_id = await _persist_feedback_attempt(
        user_id=request.user_id,
        task_id=request.task_id,
        question_id=None,
        model_used=model,
        rubric=scorecard,
        normalized_feedback=normalized_feedback,
    )

    normalized_feedback["attempt_id"] = attempt_id

    diff_from_previous = await _build_diff_from_previous_attempt(
        user_id=request.user_id,
        task_id=request.task_id,
        question_id=None,
        current_attempt_id=attempt_id,
        current_criteria=normalized_feedback["criteria"],
    )

    return {
        "attempt_id": attempt_id,
        "feedback_summary": normalized_feedback["feedback_summary"],
        "overall_score": normalized_feedback["overall_score"],
        "criteria": normalized_feedback["criteria"],
        "diff_from_previous": diff_from_previous,
        "feedback": llm_output.get("feedback"),
        "scorecard": llm_output.get("key_area_scores"),
    }


@router.post("/feedback/re-evaluate", response_model=ReevaluateFeedbackResponse)
async def re_evaluate_feedback(
    request: ReevaluateFeedbackRequest,
) -> ReevaluateFeedbackResponse:
    if request.task_type == TaskType.QUIZ:
        return await _re_evaluate_subjective_quiz(request)

    if request.task_type == TaskType.ASSIGNMENT:
        return await _re_evaluate_assignment(request)

    raise HTTPException(
        status_code=400,
        detail="Re-evaluation supports quiz or assignment tasks only",
    )


@router.get(
    "/feedback/attempts",
    response_model=List[FeedbackAttemptSummary],
)
async def get_feedback_attempt_history(
    user_id: int,
    task_id: int,
    question_id: Optional[int] = None,
    limit: int = 20,
):
    attempts = await get_feedback_attempts(
        user_id=user_id,
        task_id=task_id,
        question_id=question_id,
        limit=limit,
    )

    return attempts
