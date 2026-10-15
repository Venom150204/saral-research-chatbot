"""The conversational agent: tools, limits, streaming, interrupts and saving."""

from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import replace
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from backend.app.agent import AgentServices, build_agent, stream_turn
from backend.app.agent.answer_check import answer_units, needs_check
from backend.app.agent.deck import render_deck, source_ids
from backend.app.agent.runner import AGENT_PLAN, FinalDrafts
from backend.app.agent.tools import summarize
from backend.app.chunking import build_chunks
from backend.app.editing import EditResult, UnitDelta
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationResult, GroundedText, SupportReference,
)
from backend.app.ingestion import extract_summary_bytes
from backend.app.verification import ClaimVerdict, RequirementVerdict, VerificationReport

TEST_DATABASE_URL = os.getenv("SARAL_TEST_DATABASE_URL")


def call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class ScriptedModel(BaseChatModel):
    """Replies from a script, one message per model call; the last one repeats with
    fresh tool-call ids, which is how a looping model behaves."""

    responses: list[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _next(self) -> AIMessage:
        message = self.responses[min(self.calls, len(self.responses) - 1)]
        if self.calls >= len(self.responses) and message.tool_calls:
            message = AIMessage(content=message.content, tool_calls=[
                {**item, "id": f"{item['id']}-{self.calls}"} for item in message.tool_calls
            ])
        self.calls += 1
        return message

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._next()
        chunk_id = f"run-{uuid4()}"
        words = message.content.split(" ") if message.content else []
        for index, word in enumerate(words):
            yield ChatGenerationChunk(message=AIMessageChunk(
                content=word + (" " if index < len(words) - 1 else ""), id=chunk_id))
        if message.tool_calls or not words:
            yield ChatGenerationChunk(message=AIMessageChunk(content="", id=chunk_id, tool_call_chunks=[
                {"name": item["name"], "args": json.dumps(item["args"]), "id": item["id"], "index": index}
                for index, item in enumerate(message.tool_calls)
            ]))


DECK = b"# Results\n\nBERT reached 80.5% on GLUE.\n\n# Method\n\nBERT masks 15% of tokens.\n"


class UnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = build_chunks(extract_summary_bytes(DECK, filename="deck.md")).chunks

    def test_the_deck_is_shown_with_short_source_ids(self) -> None:
        self.assertEqual(list(source_ids(self.chunks)), [f"S{n}" for n in range(1, len(self.chunks) + 1)])
        rendered = render_deck(self.chunks)
        self.assertIn("[S1] BERT reached 80.5% on GLUE.", rendered)

    def test_an_answer_is_split_into_cited_claims_and_explanation(self) -> None:
        units, unknown = answer_units(
            "BERT reached 80.5% on GLUE [S1]. Masking is like a fill-in-the-blank exercise. "
            "It also beat GPT [S9].", self.chunks,
        )
        self.assertEqual([(unit.basis, unit.text) for unit in units], [
            ("source", "BERT reached 80.5% on GLUE."),
            ("background", "Masking is like a fill-in-the-blank exercise."),
            ("background", "It also beat GPT."),
        ])
        self.assertEqual(unknown, ["S9"])
        self.assertTrue(needs_check("BERT reached 80.5% [S1]."))
        self.assertFalse(needs_check("Done, saved as a new draft."))

    def test_tool_results_reach_the_model_as_short_summaries(self) -> None:
        summary = json.loads(summarize({"step_id": "c1", "status": "blocked", "value": {
            "kind": "blocked", "message": "Too long for one draft.", "quality_issues": ["a", "b", "c", "d"],
        }}))
        self.assertEqual(summary, {"status": "not saved", "reason": "Too long for one draft.",
                                   "issues": ["a", "b", "c"]})

    def test_a_turn_shows_only_the_final_version_of_each_draft(self) -> None:
        def event(tool, artifact_id, version, **extra):
            return {"type": "artifact", "tool": tool, **extra, "result": {"status": "succeeded", "value": {
                "artifact": {"artifact_id": artifact_id, "version": version}}}}

        drafts = FinalDrafts()
        drafts.add(event("write_deliverable", "tweets", 1))
        drafts.add(event("write_deliverable", "post", 1))
        drafts.add(event("edit_draft", "post", 2))
        self.assertEqual([(a["value"]["artifact"]["artifact_id"], a["value"]["artifact"]["version"])
                          for a in drafts.results()], [("tweets", 1), ("post", 2)])
        drafts.add(event("write_deliverable", "post-fixed", 1, replaces_previous=True))
        self.assertEqual([a["value"]["artifact"]["artifact_id"] for a in drafts.results()],
                         ["tweets", "post-fixed"])

    def test_requirement_verdicts_reach_the_model(self) -> None:
        summary = json.loads(summarize({"step_id": "w", "status": "succeeded", "value": {
            "verification": {"verdicts": [], "fulfills_request": False,
                             "quality_issues": ["Not met: notes (none)", "Opening is slow."],
                             "requirement_verdicts": [
                                 {"requirement": "7 slides", "met": True, "reason": "ok"},
                                 {"requirement": "notes", "met": False, "reason": "none"}]},
        }}))
        self.assertEqual(summary["requirements"], ["met: 7 slides", "NOT MET: notes (none)"])
        self.assertEqual(summary["reviewer_advice"], ["Opening is slow."])

    def test_an_agent_turn_authorises_exactly_its_saving_tools(self) -> None:
        self.assertEqual(sorted(step["capability"] for step in AGENT_PLAN["steps"]),
                         ["edit", "generate", "redo", "undo"])


class _NoDraft:
    def load_active(self, **_): return None


class LimitTests(unittest.TestCase):
    def test_a_looping_model_is_stopped_by_the_limits(self) -> None:
        model = ScriptedModel(responses=[call("check_claims", {"target": "artifact"}, "loop")])
        services = AgentServices(None, _NoDraft(), None, None, None, None, None)
        agent = build_agent(services=services, model=model, checkpointer=InMemorySaver())
        from backend.app.agent.context import AgentContext
        context = AgentContext("s", "t", "tenant", "user", "d", "v", deck="[S1] x")

        async def run():
            return await agent.ainvoke(
                {"messages": [("user", "check it forever")]},
                {"configurable": {"thread_id": "loop"}, "recursion_limit": 100}, context=context,
            )

        state = asyncio.run(run())
        tool_messages = [m for m in state["messages"] if m.type == "tool"]
        executed = [m for m in tool_messages if "no draft" in m.content]
        self.assertEqual(model.calls, 6)
        self.assertEqual(len(executed), 5)
        self.assertIn("limit", state["messages"][-1].content.lower())

    def test_a_tool_call_written_as_text_is_asked_for_again(self) -> None:
        leaked = AIMessage(content="Writing 2 tweets.call:default_api:write_deliverable{deliverable:2 tweets}")
        model = ScriptedModel(responses=[leaked, AIMessage(content="BERT reached 80.5% [S1].")])
        services = AgentServices(None, _NoDraft(), None, None, None, None, None)
        agent = build_agent(services=services, model=model, checkpointer=InMemorySaver())
        from backend.app.agent.context import AgentContext
        context = AgentContext("s", "t", "tenant", "user", "d", "v", deck="[S1] x")

        async def run():
            events = []
            async for chunk in agent.astream(
                {"messages": [("user", "What did BERT reach on GLUE?")]},
                {"configurable": {"thread_id": "leak"}}, context=context,
                stream_mode=["custom", "values"], version="v2",
            ):
                events.append(chunk)
            return events

        events = asyncio.run(run())
        self.assertEqual(model.calls, 2)
        self.assertIn({"type": "retract"}, [chunk["data"] for chunk in events if chunk["type"] == "custom"])
        final = [chunk["data"] for chunk in events if chunk["type"] == "values"][-1]
        self.assertEqual(final["messages"][-1].content, "BERT reached 80.5% [S1].")
        self.assertFalse(any("default_api" in str(m.content) for m in final["messages"]))

    def test_an_empty_reply_is_asked_for_again(self) -> None:
        model = ScriptedModel(responses=[AIMessage(content=""), AIMessage(content="BERT reached 80.5% [S1].")])
        services = AgentServices(None, _NoDraft(), None, None, None, None, None)
        agent = build_agent(services=services, model=model, checkpointer=InMemorySaver())
        from backend.app.agent.context import AgentContext
        context = AgentContext("s", "t", "tenant", "user", "d", "v", deck="[S1] x")

        async def run():
            return await agent.ainvoke(
                {"messages": [("user", "What did BERT reach on GLUE?")]},
                {"configurable": {"thread_id": "empty"}}, context=context,
            )

        state = asyncio.run(run())
        self.assertEqual(model.calls, 2)
        self.assertEqual(state["messages"][-1].content, "BERT reached 80.5% [S1].")
        self.assertFalse([m for m in state["messages"] if m.type == "ai" and not m.content])


class _Generator:
    def __init__(self): self.calls = 0
    async def generate(self, request, evidence):
        self.calls += 1
        chunk = evidence[0]
        support = (SupportReference(chunk.chunk_id, tuple(ref.evidence_id for ref in chunk.evidence)),)
        blocks = tuple(
            ArtifactBlock(f"tweet-{n}", "tweet", (
                GroundedText(f"t{n}", "post", "BERT reached 80.5% on GLUE.", True, support),
            ))
            for n in (1, 2)
        )
        return GenerationResult(GeneratedArtifact(
            "draft", 1, request, blocks, "fake-writer", "fake-v1"), 1.0)


class _Verifier:
    async def verify(self, artifact, evidence):
        units = [u for b in artifact.blocks for u in b.units]
        return VerificationReport(
            tuple(ClaimVerdict(u.unit_id, "supported", "ok") for u in units if u.factual),
            "fake-checker", "fake-v1", 1.0,
            uncited_verdicts=tuple(ClaimVerdict(u.unit_id, "acceptable", "ok") for u in units if not u.factual),
        )


class _RequirementVerifier(_Verifier):
    """Judges each requirement met only on a revised draft."""

    async def verify(self, artifact, evidence):
        report = await super().verify(artifact, evidence)
        met = "Revise the previous draft" in artifact.request.output.format
        return replace(report, requirement_verdicts=tuple(
            RequirementVerdict(item, met, "present" if met else "missing")
            for item in artifact.request.output.requirements
        ), fulfills_request=met)


class _Editor:
    async def edit(self, *, artifact, target_unit_ids, instruction, evidence):
        old = artifact.blocks[0].units[0]
        new = replace(old, text="BERT scored 80.5% on the GLUE benchmark.")
        blocks = (replace(artifact.blocks[0], units=(new,)), *artifact.blocks[1:])
        return EditResult(replace(artifact, version=artifact.version + 1, blocks=blocks),
                          (UnitDelta(old.unit_id, old.text, new.text),), "Reworded", "fake-editor", "fake-v1")


@unittest.skipUnless(TEST_DATABASE_URL, "SARAL_TEST_DATABASE_URL is not configured")
class AgentTurnTests(unittest.TestCase):
    def setUp(self) -> None:
        from backend.app.storage import (
            ArtifactRepository, ConversationRepository, DocumentRepository, apply_domain_migrations,
        )
        apply_domain_migrations(TEST_DATABASE_URL)
        self.owner = {"tenant_id": "tenant-agent", "user_id": "user-agent"}
        self.session_id = f"agent-{uuid4()}"
        self.conversations = ConversationRepository(TEST_DATABASE_URL)
        self.artifacts = ArtifactRepository(TEST_DATABASE_URL)
        documents = DocumentRepository(TEST_DATABASE_URL)
        self.conversations.create_session(session_id=self.session_id, **self.owner)
        self.document = extract_summary_bytes(DECK, filename="agent-deck.md")
        chunks = build_chunks(self.document)
        documents.save_index(
            document=self.document, chunks=chunks,
            vectors=tuple(tuple([1.0] * 768) for _ in chunks.chunks),
            embedding_model_id="fixture", session_id=self.session_id, **self.owner,
        )
        self.generator = _Generator()
        self.services = AgentServices(
            document_repository=documents, artifact_repository=self.artifacts,
            ledger=self.conversations, embedder=None, generator=self.generator,
            verifier=_Verifier(), editor=_Editor(),
        )
        self.saver = InMemorySaver()

    def turn(self, responses, *, message=None, resume=None, turn_id=None):
        agent = build_agent(services=self.services, model=ScriptedModel(responses=responses),
                            checkpointer=self.saver)
        turn_id = turn_id or f"turn-{uuid4()}"

        async def collect():
            return [event async for event in stream_turn(
                agent=agent, services=self.services, agent_model_id="scripted",
                session_id=self.session_id, turn_id=turn_id, **self.owner,
                document_id=self.document.document_id,
                document_version=self.document.document_version,
                message=message, resume=resume,
            )]
        return asyncio.run(collect()), turn_id

    def test_a_deliverable_turn_streams_progress_saves_once_and_records_the_turn(self) -> None:
        write = call("write_deliverable", {
            "deliverable": "two tweets about the results",
            "limits": [{"kind": "item_count", "value": 2, "quote": "2 tweets"}],
        }, "write-1")
        events, turn_id = self.turn(
            [write, write, AIMessage(content="Saved two tweets. Want a third?")],
            message="Write 2 tweets about the results",
        )
        kinds = [event["type"] for event in events]
        self.assertEqual(kinds[0], "start")
        self.assertIn("token", kinds)
        self.assertEqual(kinds[-1], "done")
        stages = [event.get("stage") for event in events if event["type"] == "step"]
        self.assertEqual(stages[:4], ["start", "writing", "measured", "checking"])
        self.assertIn("saved", stages)
        artifacts = [event for event in events if event["type"] == "artifact"]
        self.assertEqual(artifacts[0]["result"]["status"], "succeeded")
        self.assertEqual(self.generator.calls, 1)
        active = self.artifacts.load_active(session_id=self.session_id, **self.owner)
        self.assertEqual((active["version"], len(active["blocks"])), (1, 2))
        outcome = self.conversations.get_turn_outcome(session_id=self.session_id, turn_id=turn_id, **self.owner)
        self.assertEqual(outcome.payload["kind"], "agent_result")
        self.assertEqual(outcome.payload["message"], "Saved two tweets. Want a third?")
        self.assertEqual(len(outcome.payload["results"]), 1)

    def test_a_draft_missing_a_requirement_is_redone_and_only_the_final_one_is_shown(self) -> None:
        self.services = replace(self.services, verifier=_RequirementVerifier())
        args = {"deliverable": "two tweets", "requirements": ["a note per tweet"]}
        events, turn_id = self.turn([
            call("write_deliverable", args, "w1"),
            call("write_deliverable", {**args, "fix": "add a note per tweet"}, "w2"),
            AIMessage(content="Both tweets now have notes."),
        ], message="Write 2 tweets with a note per tweet")
        kinds = [event["type"] for event in events]
        artifacts = [event for event in events if event["type"] == "artifact"]
        self.assertEqual(len(artifacts), 1)
        self.assertGreater(kinds.index("artifact"), max(i for i, k in enumerate(kinds) if k == "token"))
        review = artifacts[0]["result"]["value"]["verification"]
        self.assertEqual([(r["requirement"], r["met"]) for r in review["requirement_verdicts"]],
                         [("a note per tweet", True)])
        self.assertEqual(self.generator.calls, 2)
        outcome = self.conversations.get_turn_outcome(session_id=self.session_id, turn_id=turn_id, **self.owner)
        self.assertEqual((len(outcome.payload["results"]), len(outcome.payload["all_results"])), (1, 2))

    def test_a_rewrite_keeps_a_limit_stated_in_an_earlier_message(self) -> None:
        self.turn([call("write_deliverable", {"deliverable": "tweets"}, "w1"), AIMessage(content="Done.")],
                  message="Write 2 tweets about the results")
        events, _ = self.turn([
            call("write_deliverable", {"deliverable": "tweets, rewritten",
                                       "limits": [{"kind": "item_count", "value": 2, "quote": "2 tweets"}]}, "w2"),
            AIMessage(content="Rewritten."),
        ], message="Rewrite them more simply")
        result = [e for e in events if e["type"] == "artifact"][0]["result"]
        checked = result["value"]["constraints"]["checked"]
        self.assertEqual([(item["quote"], item["met"]) for item in checked], [("2 tweets", True)])
        self.assertEqual(result["value"]["constraints"]["ignored"], [])

    def test_the_agent_can_ask_and_then_continue_the_same_turn(self) -> None:
        ask = call("ask_user", {"question": "Tweet 2 or slide 2?", "options": ["Tweet 2", "Slide 2"]}, "ask-1")
        first, turn_id = self.turn([ask], message="Make #2 simpler")
        self.assertEqual(first[-1]["type"], "interrupt")
        self.assertEqual((first[-1]["question"], first[-1]["options"]), ("Tweet 2 or slide 2?", ["Tweet 2", "Slide 2"]))
        second, _ = self.turn([AIMessage(content="Got it, tweet 2.")], resume="Tweet 2", turn_id=turn_id)
        self.assertEqual(second[-1]["type"], "done")
        self.assertEqual("".join(e["text"] for e in second if e["type"] == "token"), "Got it, tweet 2.")
        outcome = self.conversations.get_turn_outcome(session_id=self.session_id, turn_id=turn_id, **self.owner)
        self.assertEqual(outcome.payload["kind"], "agent_result")

    def test_edit_and_undo_save_and_restore_versions(self) -> None:
        self.turn([call("write_deliverable", {"deliverable": "two tweets"}, "w"), AIMessage(content="Done.")],
                  message="Write two tweets")
        events, _ = self.turn([
            call("edit_draft", {"target": "tweet-1", "instruction": "reword"}, "e"),
            AIMessage(content="Reworded tweet 1."),
        ], message="Reword the first tweet")
        edited = [e for e in events if e["type"] == "artifact"][0]["result"]
        self.assertEqual((edited["status"], edited["value"]["artifact"]["version"]), ("succeeded", 2))
        self.assertEqual(self.artifacts.load_active(session_id=self.session_id, **self.owner)["version"], 2)
        self.turn([call("history", {"action": "undo"}, "u"), AIMessage(content="Undone.")], message="Undo that")
        self.assertEqual(self.artifacts.load_active(session_id=self.session_id, **self.owner)["version"], 1)

    def test_a_streamed_answer_with_citations_is_checked_after_it_finishes(self) -> None:
        events, _ = self.turn([AIMessage(content="BERT reached 80.5% on GLUE [S1].")],
                              message="What did BERT get on GLUE?")
        verification = [e for e in events if e["type"] == "verification"][0]
        self.assertEqual((verification["claims"], verification["claims_supported"], verification["passed"]),
                         (1, 1, True))


if __name__ == "__main__":
    unittest.main()
