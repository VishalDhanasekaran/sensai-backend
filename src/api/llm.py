from typing import Optional, Type, Literal, Any
import hashlib
import json
import backoff
from langfuse.openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import create_model
from pydantic.fields import FieldInfo
import jiter
from langchain_core.output_parsers import PydanticOutputParser
import openai
import instructor
from api.utils.logging import logger
from api.db.prompt_cache import (
    log_prompt_cache_stat,
    get_cached_response,
    save_response_to_cache,
)


def _stable_hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


MAX_PROMPT_CACHE_KEY_LENGTH = 64


def _default_prompt_cache_key(model: str, api_mode: str, messages: list[dict]) -> str:
    """
    Build a deterministic cache key for each logical request.

    OpenAI enforces a maximum length of 64 characters for `prompt_cache_key`.
    """
    digest = _stable_hash_payload(
        {
            "model": model,
            "api_mode": api_mode,
            "messages": messages,
        }
    )
    # Keep a stable, short prefix for easier log filtering.
    return f"pc:{digest}"[:MAX_PROMPT_CACHE_KEY_LENGTH]


def _normalize_prompt_cache_key(cache_key: str | None) -> str | None:
    if cache_key is None:
        return None

    if len(cache_key) <= MAX_PROMPT_CACHE_KEY_LENGTH:
        return cache_key

    # If caller supplies a longer key, fold it deterministically to 64 chars.
    return f"pcu:{_stable_hash_payload(cache_key)}"[:MAX_PROMPT_CACHE_KEY_LENGTH]


def _log_cached_tokens(response: Any, model: str, prompt_cache_key: str | None = None):
    usage = getattr(response, "usage", None)
    if not usage:
        return

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
    cached_tokens_raw = None
    if prompt_tokens_details:
        cached_tokens_raw = getattr(prompt_tokens_details, "cached_tokens", None)

    cached_tokens = cached_tokens_raw if isinstance(cached_tokens_raw, int) else None
    is_hit = cached_tokens is not None and cached_tokens > 0

    if is_hit:
        logger.info(
            "OpenAI prompt cache hit",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "prompt_cache_key": prompt_cache_key,
            },
        )
    else:
        logger.info(
            "OpenAI prompt cache miss",
            extra={
                "model": model,
                "prompt_tokens": prompt_tokens,
                "prompt_cache_key": prompt_cache_key,
            },
        )

    import asyncio

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    loop.create_task(
        log_prompt_cache_stat(
            cache_key=prompt_cache_key or "",
            model=model,
            is_hit=is_hit,
            prompt_tokens=prompt_tokens,
            cached_tokens=cached_tokens,
        )
    )


def _prepare_api_kwargs(
    model: str,
    api_mode: Literal["responses", "chat_completions"],
    messages: list[dict],
    prompt_cache_retention: Literal["in_memory", "24h"],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """
    Build kwargs for OpenAI calls.

    Notes:
    - Langfuse/OpenAI wrappers can reject `prompt_cache_retention` for
      `responses.parse/stream` even when prompt caching is otherwise enabled.
    - We therefore only pass `prompt_cache_key` for Responses API.
    """
    api_kwargs = dict(kwargs)

    user_cache_key = _normalize_prompt_cache_key(
        api_kwargs.pop("prompt_cache_key", None)
    )
    api_kwargs.pop("prompt_cache_retention", None)

    prompt_cache_key: str | None = None
    if api_mode == "responses":
        prompt_cache_key = user_cache_key or _default_prompt_cache_key(
            model, api_mode, messages
        )
        api_kwargs["prompt_cache_key"] = prompt_cache_key

    return api_kwargs, prompt_cache_key


def is_reasoning_model(model: str) -> bool:
    if not model:
        return False

    for model_family in ["o3", "o1", "o1", "o4", "gpt-5"]:
        if model_family in model:
            return True

    return False


def _get_user_message_cache_key(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                return f"u:{_stable_hash_payload(content)}"[
                    :MAX_PROMPT_CACHE_KEY_LENGTH
                ]
    return None


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def stream_llm_with_instructor(
    model: str,
    messages: list,
    response_model: BaseModel,
    max_completion_tokens: int,
    **kwargs,
):
    client = instructor.from_openai(openai.AsyncOpenAI())

    if not kwargs and not is_reasoning_model(model):
        kwargs["temperature"] = 0

    return client.chat.completions.create_partial(
        model=model,
        messages=messages,
        response_model=response_model,
        stream=True,
        max_completion_tokens=max_completion_tokens,
        store=True,
        **kwargs,
    )


# This function takes any Pydantic model and creates a new one
# where all fields are optional, allowing for partial data.
def create_partial_model(model: Type[BaseModel]) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model where all fields of the original model
    are converted to Optional and have a default value of None.
    """
    new_fields = {}
    for name, field_info in model.model_fields.items():
        # Create a new FieldInfo with Optional type and a default of None
        new_field_info = FieldInfo.from_annotation(Optional[field_info.annotation])
        new_field_info.default = None
        new_fields[name] = (new_field_info.annotation, new_field_info)

    # Create the new model with the same name prefixed by "Partial"
    return create_model(f"Partial{model.__name__}", **new_fields)


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def stream_llm_with_openai(
    model: str,
    messages: list[dict],
    response_model: BaseModel,
    max_output_tokens: int,
    api_mode: Literal["responses", "chat_completions"] = "responses",
    prompt_cache_retention: Literal["in_memory", "24h"] = "in_memory",
    **kwargs,
):
    client = AsyncOpenAI()

    partial_model = create_partial_model(response_model)

    if not kwargs and not is_reasoning_model(model):
        kwargs["temperature"] = 0

    api_kwargs, prompt_cache_key = _prepare_api_kwargs(
        model=model,
        api_mode=api_mode,
        messages=messages,
        prompt_cache_retention=prompt_cache_retention,
        kwargs=kwargs,
    )

    if api_mode == "responses":
        stream = client.responses.stream(
            model=model,
            input=messages,
            text_format=response_model,
            max_output_tokens=max_output_tokens,
            store=True,
            metadata={},
            **api_kwargs,
        )
    else:
        if "-audio-" in model:
            # hack for audio as current audio models do not support response_format
            output_parser = PydanticOutputParser(pydantic_object=response_model)
            format_instructions = output_parser.get_format_instructions()

            messages[0]["content"] = (
                messages[0]["content"] + f"\n\nOutput format:\n{format_instructions}"
            )

            async for stream in await stream_llm_with_instructor(
                model=model,
                messages=messages,
                response_model=response_model,
                max_completion_tokens=max_output_tokens,
                **api_kwargs,
            ):
                yield stream

            return
        else:
            stream = client.chat.completions.stream(
                model=model,
                messages=messages,
                response_format=response_model,
                max_completion_tokens=max_output_tokens,
                store=True,
                n=1,
                **api_kwargs,
            )

    async with stream as stream:
        json_buffer = ""
        async for event in stream:
            if api_mode == "responses":
                if event.type == "response.output_text.delta":
                    # Get the content delta from the chunk
                    content = event.delta or ""
                    if not content:
                        continue

                    json_buffer += content

                    # Use jiter to parse the potentially incomplete JSON string.
                    # We wrap this in a try-except block to handle cases where the buffer
                    # is not yet a parsable JSON fragment (e.g., just whitespace or a comma).
                    try:
                        # 'trailing-strings' mode allows jiter to parse incomplete strings at the end of the JSON.
                        parsed_data = jiter.from_json(
                            json_buffer.encode("utf-8"), partial_mode="trailing-strings"
                        )

                        # Validate the partially parsed data against our dynamic partial model.
                        # `strict=False` allows for some type coercion, which is helpful here.
                        partial_obj = partial_model.model_validate(
                            parsed_data, strict=False
                        )
                        yield partial_obj
                    except:
                        # The buffer isn't a valid partial JSON object yet, so we wait for more chunks.
                        continue
            else:
                if event.type == "chunk":
                    content = event.snapshot.choices[0].message.content
                    if not content:
                        continue

                    # Use jiter to parse the potentially incomplete JSON string.
                    # We wrap this in a try-except block to handle cases where the buffer
                    # is not yet a parsable JSON fragment (e.g., just whitespace or a comma).
                    try:
                        # 'trailing-strings' mode allows jiter to parse incomplete strings at the end of the JSON.
                        parsed_data = jiter.from_json(
                            content.encode("utf-8"), partial_mode="trailing-strings"
                        )

                        # Validate the partially parsed data against our dynamic partial model.
                        # `strict=False` allows for some type coercion, which is helpful here.
                        partial_obj = partial_model.model_validate(
                            parsed_data, strict=False
                        )
                        yield partial_obj
                    except:
                        # The buffer isn't a valid partial JSON object yet, so we wait for more chunks.
                        continue
                elif event.type == "error":
                    raise event.error
                elif event.type == "content.done":
                    yield event.parsed


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def run_llm_with_openai(
    model: str,
    messages: list[dict],
    response_model: BaseModel,
    max_output_tokens: int,
    api_mode: Literal["responses", "chat_completions"] = "responses",
    prompt_cache_retention: Literal["in_memory", "24h"] = "in_memory",
    **kwargs,
):
    user_cache_key = _get_user_message_cache_key(messages)

    if user_cache_key:
        cached_response_text, cached_model = await get_cached_response(user_cache_key)
        if cached_response_text is not None:
            logger.info(
                "LLM response cache hit",
                extra={"cache_key": user_cache_key, "model": cached_model},
            )
            if isinstance(response_model, type) and issubclass(
                response_model, BaseModel
            ):
                return response_model.model_validate_json(cached_response_text)
            return cached_response_text

    client = AsyncOpenAI()

    if not kwargs and not is_reasoning_model(model):
        kwargs["temperature"] = 0

    api_kwargs, prompt_cache_key = _prepare_api_kwargs(
        model=model,
        api_mode=api_mode,
        messages=messages,
        prompt_cache_retention=prompt_cache_retention,
        kwargs=kwargs,
    )

    if api_mode == "responses":
        response = await client.responses.parse(
            model=model,
            input=messages,
            text_format=response_model,
            max_output_tokens=max_output_tokens,
            store=True,
            **api_kwargs,
        )

        _log_cached_tokens(response, model, prompt_cache_key)

        if user_cache_key:
            response_json = response.output_parsed.model_dump_json()
            await save_response_to_cache(user_cache_key, response_json, model)

        return response.output_parsed

    if "-audio-" in model:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=max_output_tokens,
            store=True,
            **api_kwargs,
        )

        _log_cached_tokens(response, model, prompt_cache_key)

        if user_cache_key:
            await save_response_to_cache(
                user_cache_key, response.choices[0].message.content, model
            )

        return response.choices[0].message.content

    response = await client.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_model,
        max_completion_tokens=max_output_tokens,
        store=True,
        **api_kwargs,
    )

    _log_cached_tokens(response, model, prompt_cache_key)

    if user_cache_key:
        response_json = response.choices[0].message.parsed.model_dump_json()
        await save_response_to_cache(user_cache_key, response_json, model)

    return response.choices[0].message.parsed
