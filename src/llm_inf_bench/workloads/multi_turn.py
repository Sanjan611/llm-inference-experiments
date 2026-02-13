"""Multi-turn conversation workload."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.runners.base import Runner
from llm_inf_bench.workloads.base import Workload

logger = logging.getLogger(__name__)


@dataclass
class ConversationScript:
    """A single conversation: initial messages plus follow-up user messages."""

    initial_messages: list[dict[str, str]]
    follow_up_messages: list[str] = field(default_factory=list)


def load_multi_turn_prompts(
    source: str | Path,
    count: int,
    turns: int,
    system_prompt: str | None = None,
    user_messages: list[str] | None = None,
    shared_prefix: str | None = None,
) -> list[ConversationScript]:
    """Load multi-turn conversation prompts from a JSONL file.

    Supports two formats:

    **Messages format** (uses config follow-ups)::

        {"messages": [{"role": "user", "content": "What is photosynthesis?"}]}

    **Turns format** (fully scripted)::

        {"turns": ["What is Python?", "How does it handle memory?", ...]}

    Args:
        source: Path to the JSONL prompt file.
        count: Number of conversations to generate.
        turns: Number of turns per conversation.
        system_prompt: Optional system prompt prepended to each conversation.
        user_messages: Follow-up messages for turns 2+ (messages format only).
        shared_prefix: Optional shared prefix for the system prompt.

    Returns:
        A list of *count* ``ConversationScript`` objects.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        ValueError: If the file format is invalid or insufficient follow-ups.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    raw_entries: list[dict[str, object]] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {e}") from e
            raw_entries.append(data)

    if not raw_entries:
        raise ValueError(f"No prompts found in {path}")

    # Detect format from first entry
    first = raw_entries[0]
    if "turns" in first:
        format_type = "turns"
    elif "messages" in first:
        format_type = "messages"
    else:
        raise ValueError(
            f"Invalid format in {path}: each line must have 'messages' or 'turns' key"
        )

    # Build conversation scripts
    scripts: list[ConversationScript] = []

    if format_type == "turns":
        for line_num, entry in enumerate(raw_entries, 1):
            if "turns" not in entry:
                raise ValueError(f"Line {line_num} in {path}: expected 'turns' key")
            turn_messages = entry["turns"]
            if not isinstance(turn_messages, list) or len(turn_messages) < turns:
                got = len(turn_messages) if isinstance(turn_messages, list) else 0
                raise ValueError(
                    f"Line {line_num} in {path}: 'turns' must have "
                    f"at least {turns} entries, got {got}"
                )

            initial: list[dict[str, str]] = []
            if system_prompt:
                initial.append({"role": "system", "content": system_prompt})
            initial.append({"role": "user", "content": turn_messages[0]})

            follow_ups = list(turn_messages[1:turns])
            scripts.append(
                ConversationScript(
                    initial_messages=initial,
                    follow_up_messages=follow_ups,
                )
            )

    else:  # messages format
        if user_messages is None:
            raise ValueError(
                "Messages format requires 'user_messages' in conversation config "
                "for follow-up turns"
            )
        if len(user_messages) < turns - 1:
            raise ValueError(
                f"Need at least {turns - 1} user_messages for {turns} turns, "
                f"got {len(user_messages)}"
            )

        for line_num, entry in enumerate(raw_entries, 1):
            if "messages" not in entry:
                raise ValueError(f"Line {line_num} in {path}: expected 'messages' key")

            msgs_initial: list[dict[str, str]] = []
            if system_prompt:
                msgs_initial.append({"role": "system", "content": system_prompt})
            msgs_initial.extend(entry["messages"])  # type: ignore[arg-type]

            follow_ups = list(user_messages[: turns - 1])
            scripts.append(
                ConversationScript(
                    initial_messages=msgs_initial,
                    follow_up_messages=follow_ups,
                )
            )

    if not scripts:
        raise ValueError(f"No valid conversations found in {path}")

    # Cycle to reach desired count
    result: list[ConversationScript] = []
    for i in range(count):
        result.append(scripts[i % len(scripts)])
    return result


class MultiTurnWorkload(Workload):
    """Execute multi-turn conversations sequentially.

    Each conversation runs its turns serially (each turn depends on the
    previous response). Conversations themselves run sequentially.
    """

    def __init__(
        self,
        conversations: list[ConversationScript],
        turns: int,
        model: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        on_request_complete: Callable[[RequestResult], None] | None = None,
        max_consecutive_failed_conversations: int = 3,
        stop_on_turn_error: bool = True,
    ) -> None:
        self._conversations = conversations
        self._turns = turns
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._on_request_complete = on_request_complete
        self._max_consecutive_failed_conversations = max_consecutive_failed_conversations
        self._stop_on_turn_error = stop_on_turn_error

    def total_requests(self) -> int:
        return len(self._conversations) * self._turns

    async def execute(self, runner: Runner) -> list[RequestResult]:
        """Execute all conversations, collecting per-turn results."""
        results: list[RequestResult] = []
        request_idx = 0
        consecutive_failed = 0

        for conv_idx, script in enumerate(self._conversations):
            conversation_failed = False
            messages: list[dict[str, str]] = list(script.initial_messages)

            for turn_idx in range(self._turns):
                if conversation_failed and self._stop_on_turn_error:
                    # Emit skip result for remaining turns
                    result = RequestResult(
                        request_index=request_idx,
                        error="skipped: previous turn failed",
                        turn_index=turn_idx,
                        conversation_index=conv_idx,
                    )
                    results.append(result)
                    if self._on_request_complete:
                        self._on_request_complete(result)
                    request_idx += 1
                    continue

                result = await runner.chat_completion(
                    messages=messages,
                    model=self._model,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                result.request_index = request_idx
                result.turn_index = turn_idx
                result.conversation_index = conv_idx

                results.append(result)
                if self._on_request_complete:
                    self._on_request_complete(result)
                request_idx += 1

                if result.error:
                    conversation_failed = True
                    logger.warning(
                        "Conversation %d turn %d failed: %s",
                        conv_idx,
                        turn_idx,
                        result.error,
                    )
                    continue

                # Append assistant response to history
                assistant_text = result.completion_text or ""
                messages.append({"role": "assistant", "content": assistant_text})

                # Append next user message if there are more turns
                if turn_idx < self._turns - 1 and turn_idx < len(script.follow_up_messages):
                    messages.append(
                        {
                            "role": "user",
                            "content": script.follow_up_messages[turn_idx],
                        }
                    )

            # Track consecutive failed conversations for circuit breaker
            if conversation_failed:
                consecutive_failed += 1
                logger.warning(
                    "Conversation %d failed (%d consecutive)",
                    conv_idx,
                    consecutive_failed,
                )
            else:
                consecutive_failed = 0

            if consecutive_failed >= self._max_consecutive_failed_conversations:
                logger.error(
                    "Circuit breaker: %d consecutive failed conversations, stopping workload",
                    consecutive_failed,
                )
                break

        return results
