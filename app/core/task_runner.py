"""Task runner: orchestrates the full LLM task pipeline.

Flow: validate context → resolve model → call runtime → validate output → retry if needed → log.
"""
import logging
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..config import settings
from ..models.db import TaskLog
from ..models.schemas import TaskRequest, TaskResponse, TaskMeta, TaskError, AttemptError
from ..runtimes.base import LLMRuntime
from . import context_manager, output_validator, model_resolver

logger = logging.getLogger(__name__)


async def run_task(
    request: TaskRequest,
    runtime: LLMRuntime,
    db: Session,
) -> TaskResponse:
    """Execute a task through the full pipeline.

    1. Resolve model from preference
    2. Check context budget
    3. Handle chunking if needed
    4. Call LLM with retry logic
    5. Validate and repair output
    6. Log task
    7. Return result or structured error
    """
    task_id = uuid.uuid4()
    start = time.monotonic()

    # --- Resolve model ---
    runtime_name = request.runtime or settings.default_runtime

    if request.model:
        # Direct model override — skip capability resolution
        resolved = model_resolver.ResolvedModel(
            name=request.model,
            safe_context_limit=request.context_budget or settings.default_context_budget,
            runtime=runtime_name,
            config=model_resolver.get_model_config(request.model),
        )
    else:
        resolved = model_resolver.resolve_model_with_fallback(
            db, request.model_preference, runtime_name,
        )
    if not resolved:
        return _error_response(
            task_id=task_id,
            code="MODEL_UNAVAILABLE",
            message=f"No model available for preference '{request.model_preference}'",
            model="none",
            start=start,
        )

    temperature = request.temperature if request.temperature is not None else settings.default_temperature
    max_tokens = request.max_tokens or settings.default_max_tokens
    context_budget = request.context_budget or settings.default_context_budget
    safe_limit = min(context_budget, resolved.safe_context_limit)

    # --- Apply model-specific config ---
    system_prompt = request.system_prompt
    user_prompt = request.user_prompt

    if request.task_type == "extract_json":
        system_prompt = system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown, no code fences, no explanation — just the JSON object."

    model_config = resolved.config
    if model_config.get("append_to_prompt"):
        user_prompt = user_prompt + " " + model_config["append_to_prompt"]

    # --- Check context budget ---
    budget = context_manager.check_context_budget(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        safe_context_limit=safe_limit,
        max_output_tokens=max_tokens,
    )

    if not budget.fits:
        # If chunking is enabled, handle it
        if request.chunking and request.chunking.enabled:
            return await _run_chunked(
                request=request,
                runtime=runtime,
                db=db,
                task_id=task_id,
                resolved=resolved,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                safe_limit=safe_limit,
                start=start,
            )

        return _error_response(
            task_id=task_id,
            code="CONTEXT_OVERFLOW",
            message=budget.message,
            model=resolved.name,
            start=start,
        )

    # --- Execute with retry logic ---
    return await _run_with_retries(
        task_id=task_id,
        request=request,
        runtime=runtime,
        db=db,
        resolved=resolved,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        safe_limit=safe_limit,
        budget=budget,
        start=start,
    )


async def _run_with_retries(
    task_id: uuid.UUID,
    request: TaskRequest,
    runtime: LLMRuntime,
    db: Session,
    resolved: model_resolver.ResolvedModel,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    safe_limit: int,
    budget: context_manager.ContextBudgetResult,
    start: float,
) -> TaskResponse:
    """Execute LLM call with retry logic on validation failure."""
    attempt_errors: list[AttemptError] = []
    max_retries = min(request.max_retries, settings.max_retries)
    was_repaired = False

    for attempt in range(1, max_retries + 1):
        # Adjust temperature on retries
        attempt_temp = temperature * (settings.retry_temperature_decay ** (attempt - 1))

        # Add format reminder on retries
        if attempt > 1 and request.task_type == "extract_json":
            retry_prompt = user_prompt + "\n\nIMPORTANT: Respond ONLY with the JSON object. Keep it concise."
        else:
            retry_prompt = user_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": retry_prompt},
        ]

        try:
            response = await runtime.generate(
                model=resolved.name,
                messages=messages,
                temperature=attempt_temp,
                max_tokens=max_tokens,
                num_ctx=safe_limit,
            )
        except Exception as e:
            attempt_errors.append(AttemptError(
                attempt=attempt,
                error=f"Runtime error: {type(e).__name__}: {str(e)[:200]}",
                raw_length=0,
            ))
            logger.error(f"Task {task_id} attempt {attempt} runtime error: {e}")
            if attempt == max_retries:
                break
            continue

        raw = response.content

        # --- Validate output ---
        if request.task_type == "extract_json":
            validation = output_validator.validate_json_output(raw, request.output_schema)

            if validation.valid:
                was_repaired = validation.was_repaired
                elapsed_ms = int((time.monotonic() - start) * 1000)

                _log_task(db, task_id, request, resolved, response, "success",
                          attempts=attempt, was_repaired=was_repaired, elapsed_ms=elapsed_ms,
                          temperature=attempt_temp, context_utilization=budget.utilization)

                return TaskResponse(
                    status="success",
                    result=validation.data,
                    meta=TaskMeta(
                        task_id=task_id,
                        model_used=resolved.name,
                        runtime=resolved.runtime,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        context_window=safe_limit,
                        context_utilization=budget.utilization,
                        latency_ms=elapsed_ms,
                        attempts=attempt,
                        temperature=attempt_temp,
                    ),
                )

            attempt_errors.append(AttemptError(
                attempt=attempt,
                error=f"{validation.error_code}: {validation.error_message}",
                raw_length=len(raw),
            ))
            logger.warning(f"Task {task_id} attempt {attempt}: {validation.error_code} ({len(raw)} chars)")

        elif request.task_type == "generate_text":
            if raw.strip():
                elapsed_ms = int((time.monotonic() - start) * 1000)

                _log_task(db, task_id, request, resolved, response, "success",
                          attempts=attempt, was_repaired=False, elapsed_ms=elapsed_ms,
                          temperature=attempt_temp, context_utilization=budget.utilization)

                return TaskResponse(
                    status="success",
                    result=raw,
                    meta=TaskMeta(
                        task_id=task_id,
                        model_used=resolved.name,
                        runtime=resolved.runtime,
                        tokens_in=response.tokens_in,
                        tokens_out=response.tokens_out,
                        context_window=safe_limit,
                        context_utilization=budget.utilization,
                        latency_ms=elapsed_ms,
                        attempts=attempt,
                        temperature=attempt_temp,
                    ),
                )

            attempt_errors.append(AttemptError(
                attempt=attempt,
                error="EMPTY_RESPONSE",
                raw_length=0,
            ))

    # All retries exhausted
    elapsed_ms = int((time.monotonic() - start) * 1000)
    error_code = attempt_errors[-1].error.split(":")[0] if attempt_errors else "UNKNOWN"

    _log_task(db, task_id, request, resolved, None, "error",
              attempts=max_retries, was_repaired=False, elapsed_ms=elapsed_ms,
              temperature=temperature, context_utilization=budget.utilization,
              error_code=error_code, error_detail=str(attempt_errors))

    return TaskResponse(
        status="error",
        error=TaskError(
            code=error_code,
            message=f"Failed after {max_retries} attempts",
            attempts=attempt_errors,
        ),
        meta=TaskMeta(
            task_id=task_id,
            model_used=resolved.name,
            runtime=resolved.runtime,
            tokens_in=0,
            tokens_out=0,
            context_window=safe_limit,
            context_utilization=budget.utilization,
            latency_ms=elapsed_ms,
            attempts=max_retries,
            temperature=temperature,
        ),
    )


async def _run_chunked(
    request: TaskRequest,
    runtime: LLMRuntime,
    db: Session,
    task_id: uuid.UUID,
    resolved: model_resolver.ResolvedModel,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    safe_limit: int,
    start: float,
) -> TaskResponse:
    """Handle chunked processing for inputs that exceed context budget."""
    chunking = request.chunking

    chunks = context_manager.chunk_text(
        user_prompt,
        chunk_tokens=chunking.chunk_tokens,
        overlap_tokens=chunking.overlap_tokens,
    )

    logger.info(f"Task {task_id}: chunked into {len(chunks)} pieces")

    chunk_results = []
    total_tokens_in = 0
    total_tokens_out = 0
    total_attempts = 0

    for chunk in chunks:
        # Create a sub-request for each chunk
        chunk_request = TaskRequest(
            task_type=request.task_type,
            model_preference=request.model_preference,
            system_prompt=request.system_prompt,
            user_prompt=chunk.text,
            output_schema=request.output_schema,
            max_retries=request.max_retries,
            context_budget=request.context_budget,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        chunk_response = await _run_with_retries(
            task_id=task_id,
            request=chunk_request,
            runtime=runtime,
            db=db,
            resolved=resolved,
            system_prompt=system_prompt,
            user_prompt=chunk.text,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_limit=safe_limit,
            budget=context_manager.check_context_budget(
                system_prompt, chunk.text, safe_limit, max_tokens,
            ),
            start=start,
        )

        if chunk_response.status == "success" and chunk_response.result:
            chunk_results.append(chunk_response.result)
            total_tokens_in += chunk_response.meta.tokens_in
            total_tokens_out += chunk_response.meta.tokens_out
            total_attempts += chunk_response.meta.attempts
        else:
            logger.warning(f"Chunk {chunk.chunk_index} failed, skipping")

    if not chunk_results:
        return _error_response(task_id, "ALL_CHUNKS_FAILED",
                               "All chunks failed during processing", resolved.name, start)

    merged = context_manager.merge_json_results(chunk_results, chunking.merge_strategy)
    elapsed_ms = int((time.monotonic() - start) * 1000)

    return TaskResponse(
        status="success",
        result=merged,
        meta=TaskMeta(
            task_id=task_id,
            model_used=resolved.name,
            runtime=resolved.runtime,
            tokens_in=total_tokens_in,
            tokens_out=total_tokens_out,
            context_window=safe_limit,
            context_utilization=0.0,
            latency_ms=elapsed_ms,
            attempts=total_attempts,
            temperature=temperature,
        ),
    )


def _error_response(
    task_id: uuid.UUID,
    code: str,
    message: str,
    model: str,
    start: float,
) -> TaskResponse:
    elapsed_ms = int((time.monotonic() - start) * 1000)
    return TaskResponse(
        status="error",
        error=TaskError(code=code, message=message),
        meta=TaskMeta(
            task_id=task_id,
            model_used=model,
            runtime=settings.default_runtime,
            tokens_in=0,
            tokens_out=0,
            context_window=0,
            context_utilization=0.0,
            latency_ms=elapsed_ms,
            attempts=0,
            temperature=0.0,
        ),
    )


def _log_task(
    db: Session,
    task_id: uuid.UUID,
    request: TaskRequest,
    resolved: model_resolver.ResolvedModel,
    response,
    status: str,
    attempts: int,
    was_repaired: bool,
    elapsed_ms: int,
    temperature: float,
    context_utilization: float,
    error_code: str | None = None,
    error_detail: str | None = None,
):
    """Log the task to the database."""
    try:
        log = TaskLog(
            id=task_id,
            task_type=request.task_type,
            model_requested=request.model_preference,
            model_used=resolved.name,
            runtime=resolved.runtime,
            tokens_in=response.tokens_in if response else 0,
            tokens_out=response.tokens_out if response else 0,
            context_utilization=context_utilization,
            status=status,
            error_code=error_code,
            error_detail=error_detail,
            attempts=attempts,
            was_repaired=was_repaired,
            latency_ms=elapsed_ms,
            temperature=temperature,
        )
        db.add(log)
        db.commit()
    except Exception as e:
        logger.error(f"Failed to log task {task_id}: {e}")
        db.rollback()
