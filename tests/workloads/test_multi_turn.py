"""Tests for multi-turn conversation workload."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from llm_inf_bench.metrics.collector import RequestResult
from llm_inf_bench.workloads.multi_turn import (
    ConversationScript,
    MultiTurnWorkload,
    load_multi_turn_prompts,
)


# ---------------------------------------------------------------------------
# load_multi_turn_prompts
# ---------------------------------------------------------------------------


class TestLoadMultiTurnPrompts:
    """Tests for the multi-turn prompt loader."""

    def test_messages_format_basic(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            '{"messages": [{"role": "user", "content": "Hello"}]}\n'
            '{"messages": [{"role": "user", "content": "World"}]}\n'
        )
        scripts = load_multi_turn_prompts(
            path, count=2, turns=3,
            user_messages=["Follow up 1", "Follow up 2"],
        )
        assert len(scripts) == 2
        assert scripts[0].initial_messages == [{"role": "user", "content": "Hello"}]
        assert scripts[0].follow_up_messages == ["Follow up 1", "Follow up 2"]
        assert scripts[1].initial_messages == [{"role": "user", "content": "World"}]

    def test_messages_format_with_system_prompt(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "Hi"}]}\n')
        scripts = load_multi_turn_prompts(
            path, count=1, turns=2,
            system_prompt="You are helpful.",
            user_messages=["More please"],
        )
        assert len(scripts) == 1
        assert scripts[0].initial_messages[0] == {
            "role": "system", "content": "You are helpful."
        }
        assert scripts[0].initial_messages[1] == {
            "role": "user", "content": "Hi"
        }

    def test_messages_format_cycles(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "A"}]}\n')
        scripts = load_multi_turn_prompts(
            path, count=3, turns=2,
            user_messages=["Follow up"],
        )
        assert len(scripts) == 3
        assert all(s.initial_messages[0]["content"] == "A" for s in scripts)

    def test_messages_format_requires_user_messages(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "Hi"}]}\n')
        with pytest.raises(ValueError, match="user_messages"):
            load_multi_turn_prompts(path, count=1, turns=3)

    def test_messages_format_insufficient_user_messages(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text('{"messages": [{"role": "user", "content": "Hi"}]}\n')
        with pytest.raises(ValueError, match="at least 4 user_messages"):
            load_multi_turn_prompts(
                path, count=1, turns=5,
                user_messages=["A", "B"],  # need 4
            )

    def test_turns_format_basic(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        turns_data = ["Q1", "Q2", "Q3"]
        path.write_text(json.dumps({"turns": turns_data}) + "\n")
        scripts = load_multi_turn_prompts(path, count=1, turns=3)
        assert len(scripts) == 1
        assert scripts[0].initial_messages == [{"role": "user", "content": "Q1"}]
        assert scripts[0].follow_up_messages == ["Q2", "Q3"]

    def test_turns_format_with_system_prompt(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(json.dumps({"turns": ["Q1", "Q2"]}) + "\n")
        scripts = load_multi_turn_prompts(
            path, count=1, turns=2,
            system_prompt="Be concise.",
        )
        assert scripts[0].initial_messages[0] == {
            "role": "system", "content": "Be concise."
        }
        assert scripts[0].initial_messages[1] == {
            "role": "user", "content": "Q1"
        }

    def test_turns_format_insufficient_turns(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(json.dumps({"turns": ["Q1", "Q2"]}) + "\n")
        with pytest.raises(ValueError, match="at least 5 entries"):
            load_multi_turn_prompts(path, count=1, turns=5)

    def test_turns_format_cycles(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(json.dumps({"turns": ["A", "B", "C"]}) + "\n")
        scripts = load_multi_turn_prompts(path, count=4, turns=3)
        assert len(scripts) == 4

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_multi_turn_prompts("/nonexistent.jsonl", count=1, turns=2, user_messages=["x"])

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="No prompts found"):
            load_multi_turn_prompts(path, count=1, turns=2, user_messages=["x"])

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_multi_turn_prompts(path, count=1, turns=2, user_messages=["x"])

    def test_invalid_format_no_key(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"prompt": "hello"}\n')
        with pytest.raises(ValueError, match="'messages' or 'turns'"):
            load_multi_turn_prompts(path, count=1, turns=2, user_messages=["x"])

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "prompts.jsonl"
        path.write_text(
            "\n"
            '{"messages": [{"role": "user", "content": "A"}]}\n'
            "\n"
        )
        scripts = load_multi_turn_prompts(
            path, count=1, turns=2, user_messages=["B"],
        )
        assert len(scripts) == 1


# ---------------------------------------------------------------------------
# MultiTurnWorkload
# ---------------------------------------------------------------------------


class TestMultiTurnWorkload:
    """Tests for the MultiTurnWorkload execution."""

    def _make_result(
        self,
        completion_text: str = "Response",
        error: str | None = None,
        ttft_ms: float | None = 50.0,
        e2e_latency_ms: float | None = 300.0,
        prompt_tokens: int | None = 50,
        completion_tokens: int | None = 20,
    ) -> RequestResult:
        return RequestResult(
            request_index=0,
            ttft_ms=ttft_ms if not error else None,
            e2e_latency_ms=e2e_latency_ms if not error else None,
            prompt_tokens=prompt_tokens if not error else None,
            completion_tokens=completion_tokens if not error else None,
            completion_text=completion_text if not error else None,
            error=error,
        )

    def _make_runner(
        self, results: list[RequestResult],
    ) -> tuple[MagicMock, list[list[dict[str, str]]]]:
        """Create a mock runner that records message snapshots.

        Returns the runner mock and a list that will be populated with
        a copy of the messages from each call (to avoid mutation issues).
        """
        captured_messages: list[list[dict[str, str]]] = []
        results_iter = iter(results)

        async def _chat_completion(
            messages: list[dict[str, str]], **kwargs: object,
        ) -> RequestResult:
            captured_messages.append([dict(m) for m in messages])
            return next(results_iter)

        runner = MagicMock()
        runner.chat_completion = AsyncMock(side_effect=_chat_completion)
        return runner, captured_messages

    @pytest.mark.asyncio
    async def test_basic_3_turn(self):
        """3-turn conversation produces 3 results."""
        script = ConversationScript(
            initial_messages=[{"role": "user", "content": "Hi"}],
            follow_up_messages=["Follow 1", "Follow 2"],
        )
        results_data = [
            self._make_result("Response 1"),
            self._make_result("Response 2"),
            self._make_result("Response 3"),
        ]
        runner, _ = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=[script], turns=3, model="test",
        )
        assert workload.total_requests() == 3

        results = await workload.execute(runner)
        assert len(results) == 3
        assert runner.chat_completion.call_count == 3

        # Check indices
        assert results[0].turn_index == 0
        assert results[0].conversation_index == 0
        assert results[1].turn_index == 1
        assert results[2].turn_index == 2

    @pytest.mark.asyncio
    async def test_message_history_grows(self):
        """Each turn should have a longer message list."""
        script = ConversationScript(
            initial_messages=[{"role": "user", "content": "Q1"}],
            follow_up_messages=["Q2", "Q3"],
        )
        results_data = [
            self._make_result("A1"),
            self._make_result("A2"),
            self._make_result("A3"),
        ]
        runner, captured = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=[script], turns=3, model="test",
        )
        await workload.execute(runner)

        # Turn 1: [user Q1]
        assert len(captured[0]) == 1
        assert captured[0][0]["content"] == "Q1"

        # Turn 2: [user Q1, assistant A1, user Q2]
        assert len(captured[1]) == 3
        assert captured[1][1]["role"] == "assistant"
        assert captured[1][1]["content"] == "A1"
        assert captured[1][2]["content"] == "Q2"

        # Turn 3: [user Q1, assistant A1, user Q2, assistant A2, user Q3]
        assert len(captured[2]) == 5

    @pytest.mark.asyncio
    async def test_multiple_conversations(self):
        """Multiple conversations run sequentially."""
        scripts = [
            ConversationScript(
                initial_messages=[{"role": "user", "content": f"Conv{i}"}],
                follow_up_messages=["Follow"],
            )
            for i in range(3)
        ]
        # 3 conversations x 2 turns = 6 results
        results_data = [self._make_result(f"R{i}") for i in range(6)]
        runner, _ = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=scripts, turns=2, model="test",
        )
        assert workload.total_requests() == 6

        results = await workload.execute(runner)
        assert len(results) == 6
        assert results[0].conversation_index == 0
        assert results[2].conversation_index == 1
        assert results[4].conversation_index == 2

    @pytest.mark.asyncio
    async def test_turn_error_skips_remaining(self):
        """When a turn fails, remaining turns are skipped."""
        script = ConversationScript(
            initial_messages=[{"role": "user", "content": "Hi"}],
            follow_up_messages=["Q2", "Q3"],
        )
        results_data = [
            self._make_result(error="server error"),  # turn 0 fails
        ]
        runner, _ = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=[script], turns=3, model="test",
            stop_on_turn_error=True,
        )
        results = await workload.execute(runner)

        assert len(results) == 3
        assert results[0].error == "server error"
        assert results[1].error == "skipped: previous turn failed"
        assert results[2].error == "skipped: previous turn failed"
        # Only one actual API call
        assert runner.chat_completion.call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker(self):
        """Circuit breaker stops after N consecutive failed conversations."""
        scripts = [
            ConversationScript(
                initial_messages=[{"role": "user", "content": f"C{i}"}],
                follow_up_messages=[],
            )
            for i in range(5)
        ]
        # All fail on turn 0
        results_data = [self._make_result(error="fail") for _ in range(5)]
        runner, _ = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=scripts,
            turns=1,
            model="test",
            max_consecutive_failed_conversations=2,
        )
        results = await workload.execute(runner)

        # Should stop after 2 consecutive failures
        assert len(results) == 2
        assert runner.chat_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker counter resets when a conversation succeeds."""
        scripts = [
            ConversationScript(
                initial_messages=[{"role": "user", "content": f"C{i}"}],
                follow_up_messages=[],
            )
            for i in range(5)
        ]
        results_data = [
            self._make_result(error="fail"),     # conv 0 fails
            self._make_result("OK"),             # conv 1 succeeds — reset
            self._make_result(error="fail"),     # conv 2 fails
            self._make_result(error="fail"),     # conv 3 fails — breaker trips
        ]
        runner, _ = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=scripts,
            turns=1,
            model="test",
            max_consecutive_failed_conversations=2,
        )
        results = await workload.execute(runner)

        # 4 conversations attempted: fail, success (reset), fail, fail (trip)
        assert len(results) == 4
        assert runner.chat_completion.call_count == 4

    @pytest.mark.asyncio
    async def test_callback_invoked(self):
        """on_request_complete is called for every result including skipped."""
        script = ConversationScript(
            initial_messages=[{"role": "user", "content": "Hi"}],
            follow_up_messages=["Q2"],
        )
        results_data = [
            self._make_result(error="fail"),  # turn 0 fails
        ]
        runner, _ = self._make_runner(results_data)
        callback = MagicMock()

        workload = MultiTurnWorkload(
            conversations=[script], turns=2, model="test",
            on_request_complete=callback,
        )
        results = await workload.execute(runner)

        # 1 actual + 1 skipped = 2 callbacks
        assert callback.call_count == 2

    def test_total_requests(self):
        scripts = [
            ConversationScript(initial_messages=[{"role": "user", "content": "Hi"}])
            for _ in range(5)
        ]
        workload = MultiTurnWorkload(
            conversations=scripts, turns=10, model="test",
        )
        assert workload.total_requests() == 50

    @pytest.mark.asyncio
    async def test_system_prompt_in_messages(self):
        """System prompt should be preserved in API calls."""
        script = ConversationScript(
            initial_messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Q1"},
            ],
            follow_up_messages=["Q2"],
        )
        results_data = [
            self._make_result("A1"),
            self._make_result("A2"),
        ]
        runner, captured = self._make_runner(results_data)

        workload = MultiTurnWorkload(
            conversations=[script], turns=2, model="test",
        )
        await workload.execute(runner)

        # Turn 1 should have system + user
        assert captured[0][0]["role"] == "system"
        assert captured[0][1]["role"] == "user"

        # Turn 2 should have system + user + assistant + user
        assert len(captured[1]) == 4
        assert captured[1][0]["role"] == "system"
