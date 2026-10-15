"""Writing a deliverable: contracts, the Gemini writer, the write-check-save graph, limits and figures."""

from __future__ import annotations

import unittest
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference,
)
import asyncio
from backend.app.chunking import build_chunks
from backend.app.generation import GenerationRequest, OutputSpecification
from backend.app.generation.providers.gemini import GeminiGroundedGenerator, ProviderResponse
from backend.app.ingestion import extract_summary_bytes
import json
from dataclasses import replace
from backend.app.generation import (
    ArtifactBlock, DeliverableConstraint, GeneratedArtifact, GenerationRequest,
    GenerationResult, GroundedText, SupportReference,
)
from backend.app.generation.providers.gemini import _build_prompt
from backend.app.verification import (
    ClaimVerdict, SuggestedAction, UserGuidance, VerificationReport,
)
from backend.app.editing import EditResult, UnitDelta
from backend.app.workflow import build_generation_execution_graph
from dataclasses import asdict
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference, artifact_from_dict,
)
from backend.app.verification import ClaimVerdict, GeminiSupportVerifier, VerificationReport
from pathlib import Path
from backend.app.ingestion import extract_pptx_bytes
from backend.app.verification import (
    GeminiSupportVerifier, complete_numeric_citations, figure_findings,
    validate_deterministic_support,
)
from backend.app.generation import (
    ArtifactBlock, DeliverableConstraint, GeneratedArtifact, GenerationRequest,
    GroundedText, OutputSpecification, artifact_from_dict,
)
from backend.app.verification import (
    describe_constraints, measure_constraints, parse_constraints,
)
from backend.app.verification.constraints import word_band
from backend.app.provider_retry import call_with_retry, is_transient
from backend.app.workflow.generation_graph import WorkflowStageError
from backend.app.prompts import GENERATION_PROMPT, render_prompt


class GenerationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = GenerationRequest(
            document_id="doc_attention",
            document_version="sha256:fixture",
            output=OutputSpecification(
                format="a narrated policy briefing with callout cards",
                audience="local policymakers unfamiliar with machine learning",
                style_guidance="plain English, candid about limitations",
                target_duration_seconds=90,
                requirements=("emphasize implications and limitations",),
            ),
        )
        self.support = SupportReference("chk_results", ("blk_bleu",))

    def test_output_format_and_roles_are_not_enumerated(self) -> None:
        self.assertIn("callout cards", self.request.output.format)

    def test_factual_output_requires_explicit_source_support(self) -> None:
        with self.assertRaisesRegex(ValueError, "require source support"):
            GroundedText("sentence-1", "custom_role", "A factual claim.", True)

    def test_transition_text_can_be_nonfactual_and_uncited(self) -> None:
        transition = GroundedText(
            "sentence-1", "rhetorical_transition", "Why does this matter?", False
        )
        self.assertEqual(transition.support, ())

    def test_artifact_tracks_generic_units_provenance_and_spoken_words(self) -> None:
        bullet = GroundedText(
            "bullet-1", "display_callout", "Attention enables parallel processing.",
            True, (self.support,), delivery="display",
        )
        sentence = GroundedText(
            "sentence-1", "narration", "The Transformer replaces recurrence with attention.",
            True, (self.support,), delivery="spoken",
        )
        artifact = GeneratedArtifact(
            "artifact-1", 1, self.request,
            (ArtifactBlock("block-1", "policy_briefing_segment", (bullet, sentence), "Why it matters"),),
            "fixture-generator", "generation-v2",
        )
        self.assertEqual(artifact.spoken_word_count, 6)
        self.assertEqual(artifact.blocks[0].units[1].support[0].chunk_id, "chk_results")


def _request_and_chunks():
    document = extract_summary_bytes(
        b"# Result\n\nThe Transformer replaces recurrence with attention.\n",
        filename="summary.md",
    )
    request = GenerationRequest(
        document.document_id, document.document_version,
        OutputSpecification(
            format="a narrated briefing with one headline",
            audience="policymakers", style_guidance="plain English",
            target_duration_seconds=90,
        ),
    )
    return request, build_chunks(document).chunks

class FakeTransport:
    def __init__(self, payload):
        self.payload, self.calls = payload, []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResponse(self.payload, 100, 40, 140)

def _payload(chunk, *, evidence_id=None):
    return {"blocks": [{
        "block_id": "opening", "block_type": "briefing_segment",
        "title": "A faster architecture", "units": [
            {"unit_id": "headline-1", "role": "headline",
             "text": "Attention replaces recurrence.", "basis": "source",
             "support": [{"chunk_id": chunk.chunk_id}],
             "contains_math": False, "delivery": "display"},
            {"unit_id": "narration-1", "role": "narration",
             "text": "Why should public leaders care?", "basis": "framing",
             "support": [], "contains_math": False, "delivery": "spoken"},
        ],
    }]}

class GeminiGenerationTests(unittest.TestCase):
    def test_schema_requires_only_allowed_chunk_and_server_expands_evidence(self) -> None:
        from backend.app.generation.providers.gemini import _response_schema
        from dataclasses import replace

        _, chunks = _request_and_chunks()
        second = replace(
            chunks[0], chunk_id="second-chunk",
            evidence=(replace(chunks[0].evidence[0], evidence_id="second-evidence"),),
        )
        schema = _response_schema((chunks[0], second))
        support = schema["properties"]["blocks"]["items"]["properties"]["units"]["items"]["properties"]["support"]["items"]
        self.assertEqual(
            support["properties"]["chunk_id"]["enum"],
            [chunks[0].chunk_id, "second-chunk"],
        )
        self.assertNotIn("evidence_ids", support["properties"])

    def test_generates_generic_artifact_with_allowed_provenance(self) -> None:
        request, chunks = _request_and_chunks()
        transport = FakeTransport(_payload(chunks[0]))
        result = asyncio.run(GeminiGroundedGenerator(transport).generate(request, chunks))
        self.assertEqual(result.artifact.spoken_word_count, 5)
        self.assertEqual(result.input_tokens, 100)
        self.assertIn("narrated briefing with one headline", transport.calls[0]["prompt"])
        self.assertIn("Do not mistake brevity for quality", transport.calls[0]["prompt"])
        self.assertIn("three threads means three distinct", transport.calls[0]["prompt"])
        self.assertIn("actual words the user can deliver", transport.calls[0]["prompt"])
        self.assertIn("Never manufacture or complete", transport.calls[0]["prompt"])
        self.assertIn("blocks", transport.calls[0]["response_schema"]["properties"])

    def test_rejects_hallucinated_chunk_identifier(self) -> None:
        request, chunks = _request_and_chunks()
        with self.assertRaisesRegex(ValueError, "outside the supplied chunks"):
            payload = _payload(chunks[0])
            payload["blocks"][0]["units"][0]["support"][0]["chunk_id"] = "invented"
            asyncio.run(GeminiGroundedGenerator(FakeTransport(payload)).generate(request, chunks))

    def test_rejects_cross_document_evidence(self) -> None:
        request, chunks = _request_and_chunks()
        wrong_request = GenerationRequest("doc_other", request.document_version, request.output)
        with self.assertRaises(PermissionError):
            asyncio.run(GeminiGroundedGenerator(FakeTransport(_payload(chunks[0]))).generate(wrong_request, chunks))

def _sized_request(*limits):
    from dataclasses import replace
    from backend.app.generation import DeliverableConstraint

    request, chunks = _request_and_chunks()
    output = replace(request.output, constraints=tuple(DeliverableConstraint(*item) for item in limits))
    return replace(request, output=output), chunks

class ProfiledTransport(FakeTransport):
    """Reports a model profile and cuts the answer off a given number of times."""

    def __init__(self, payload, *, limit=65_536, thinks=False, cut_offs=0):
        super().__init__(payload)
        self.limit, self.thinks, self.cut_offs = limit, thinks, cut_offs

    async def model_profile(self, model_id):
        from backend.app.generation.providers.gemini import ModelProfile
        return ModelProfile(self.limit, self.thinks)

    async def generate(self, **kwargs):
        from backend.app.generation.providers.gemini import OutputTruncatedError
        self.calls.append(kwargs)
        if self.cut_offs:
            self.cut_offs -= 1
            raise OutputTruncatedError(kwargs["max_output_tokens"])
        return ProviderResponse(self.payload, 100, 40, 140)

class WritingRoomTests(unittest.TestCase):
    def budgets(self, transport, request, chunks):
        asyncio.run(GeminiGroundedGenerator(transport).generate(request, chunks))
        return [(call["max_output_tokens"], call["timeout_seconds"]) for call in transport.calls]

    def test_room_and_time_follow_the_size_of_the_request(self) -> None:
        short, chunks = _sized_request(("duration_seconds", 90, "90-second"))
        long, _ = _sized_request(("duration_seconds", 1200, "20-minute"))
        unsized, _ = _sized_request()
        self.assertEqual(self.budgets(ProfiledTransport(_payload(chunks[0])), short, chunks),
                         [(8_000, 66.5625)])
        self.assertEqual(self.budgets(ProfiledTransport(_payload(chunks[0])), long, chunks),
                         [(29_250, 300.0)])
        self.assertEqual(self.budgets(ProfiledTransport(_payload(chunks[0])), unsized, chunks),
                         [(8_000, 92.5)])

    def test_a_reasoning_model_gets_extra_room_and_the_model_maximum_is_the_ceiling(self) -> None:
        short, chunks = _sized_request(("duration_seconds", 90, "90-second"))
        thinking = ProfiledTransport(_payload(chunks[0]), thinks=True)
        self.assertEqual(self.budgets(thinking, short, chunks)[0][0], 12_000)
        long, _ = _sized_request(("duration_seconds", 1200, "20-minute"))
        small_model = ProfiledTransport(_payload(chunks[0]), limit=16_384)
        self.assertEqual(self.budgets(small_model, long, chunks)[0][0], 16_384)

    def test_a_cut_off_answer_is_retried_once_with_double_the_room(self) -> None:
        request, chunks = _sized_request(("duration_seconds", 90, "90-second"))
        transport = ProfiledTransport(_payload(chunks[0]), cut_offs=1)
        self.assertEqual(self.budgets(transport, request, chunks),
                         [(8_000, 66.5625), (16_000, 133.125)])

    def test_a_request_too_long_for_one_draft_says_so_in_plain_words(self) -> None:
        from backend.app.generation.providers.gemini import RequestTooLongError

        request, chunks = _sized_request(("duration_seconds", 90, "90-second"))
        twice = ProfiledTransport(_payload(chunks[0]), cut_offs=2)
        with self.assertRaises(RequestTooLongError) as raised:
            asyncio.run(GeminiGroundedGenerator(twice).generate(request, chunks))
        self.assertEqual(len(twice.calls), 2)
        self.assertIn("about 2460 words fit in one draft", raised.exception.user_message)
        self.assertFalse(raised.exception.retryable)
        at_ceiling = ProfiledTransport(_payload(chunks[0]), limit=8_000, cut_offs=1)
        with self.assertRaises(RequestTooLongError):
            asyncio.run(GeminiGroundedGenerator(at_ceiling).generate(request, chunks))
        self.assertEqual(len(at_ceiling.calls), 1)

    def test_the_providers_finish_reason_is_read_whatever_its_form(self) -> None:
        from types import SimpleNamespace
        from google.genai import types
        from backend.app.generation.providers.gemini import _finish_reason

        def response(reason):
            return SimpleNamespace(candidates=[SimpleNamespace(finish_reason=reason)])

        self.assertEqual(_finish_reason(response(types.FinishReason.MAX_TOKENS)), "MAX_TOKENS")
        self.assertEqual(_finish_reason(response(types.FinishReason.STOP)), "STOP")
        self.assertIsNone(_finish_reason(SimpleNamespace(candidates=[])))


class DocumentRepo:
    def __init__(self, chunks): self.chunks = chunks
    def load_chunk_set(self, **kwargs):
        return self.chunks, tuple(tuple([1.0] * 768) for _ in self.chunks.chunks), "fixture"

class Generator:
    async def generate(self, request, evidence):
        chunk = evidence[0]
        artifact = GeneratedArtifact(
            "artifact-workflow", 1, request,
            (ArtifactBlock("block", "custom_card", (GroundedText(
                "claim", "claim", "Attention replaces recurrence.", True,
                (SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,)),),
            ),)),), "fixture-generator", "fixture-v1",
        )
        return GenerationResult(artifact, 1.0)

class InvalidQualityRewriteGenerator(Generator):
    def __init__(self): self.calls = 0
    async def generate(self, request, evidence):
        self.calls += 1
        result = await super().generate(request, evidence)
        if self.calls == 1:
            return result
        unit = replace(
            result.artifact.blocks[0].units[0],
            support=(SupportReference("missing-chunk", ("missing-evidence",)),),
        )
        artifact = replace(
            result.artifact,
            blocks=(replace(result.artifact.blocks[0], units=(unit,)),),
        )
        return GenerationResult(artifact, 1.0)

class InvalidFirstDraftGenerator(InvalidQualityRewriteGenerator):
    async def generate(self, request, evidence):
        self.calls += 1
        result = await Generator.generate(self, request, evidence)
        if self.calls > 1:
            return result
        unit = replace(
            result.artifact.blocks[0].units[0],
            support=(SupportReference("missing-chunk", ("missing-evidence",)),),
        )
        artifact = replace(
            result.artifact,
            blocks=(replace(result.artifact.blocks[0], units=(unit,)),),
        )
        return GenerationResult(artifact, 1.0)

class Verifier:
    def __init__(self, verdict="supported"): self.verdict = verdict
    async def verify(self, artifact, evidence):
        return VerificationReport(
            (ClaimVerdict("claim", self.verdict, "fixture", ()),),
            "fixture-verifier", "fixture-v1", 1.0,
        )

class SequenceVerifier:
    def __init__(self, verdicts): self.verdicts, self.calls = list(verdicts), []
    async def verify(self, artifact, evidence):
        self.calls.append(artifact)
        verdict = self.verdicts.pop(0)
        return VerificationReport(
            (ClaimVerdict("claim", verdict, "fixture feedback", ()),),
            "fixture-verifier", "fixture-v1", 1.0,
        )

class QualitySequenceVerifier:
    def __init__(self, accepted_after_rewrite=True):
        self.calls = []
        self.accepted_after_rewrite = accepted_after_rewrite
    async def verify(self, artifact, evidence):
        self.calls.append(artifact)
        if len(self.calls) == 1:
            return VerificationReport(
                (ClaimVerdict("claim", "supported", "grounded", ()),),
                "fixture-verifier", "fixture-v1", 1.0, False,
                "The response is a fact list, not usable prose.",
                ("Rewrite as ready-to-use prose.",),
                UserGuidance(
                    summary="The draft is grounded but not usable in the requested form.",
                    observations=("It reads as a fact list.",),
                    attempts=("Grounding and request fit were checked.",),
                    suggested_actions=(SuggestedAction(
                        "Narrow the format", "Create a concise prose version from the same evidence."
                    ),),
                ),
            )
        return VerificationReport(
            (ClaimVerdict("claim", "supported", "grounded", ()),),
            "fixture-verifier", "fixture-v1", 1.0,
            self.accepted_after_rewrite,
            "Usable prose." if self.accepted_after_rewrite else "Still not usable.",
            () if self.accepted_after_rewrite else ("Still a fact list.",),
        )

class CountingVerifier:
    """Supports every claim and counts how often the paid review ran."""
    def __init__(self): self.calls = 0
    async def verify(self, artifact, evidence):
        self.calls += 1
        return VerificationReport(
            tuple(ClaimVerdict(unit.unit_id, "supported", "fixture", ())
                  for block in artifact.blocks for unit in block.units),
            "fixture-verifier", "fixture-v1", 1.0,
        )

class ShapedGenerator:
    """One draft per call; each draft is a list of spoken word counts, one per block."""
    def __init__(self, *drafts): self.drafts, self.requests = list(drafts), []
    async def generate(self, request, evidence):
        self.requests.append(request)
        chunk = evidence[0]
        support = (SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,)),)
        blocks = tuple(
            ArtifactBlock(f"block-{index}", "item", (GroundedText(
                f"unit-{index}", "body", " ".join(["attention"] * words), True, support,
                delivery="spoken",
            ),))
            for index, words in enumerate(self.drafts.pop(0), start=1)
        )
        return GenerationResult(GeneratedArtifact(
            "artifact-shaped", 1, request, blocks, "fixture-generator", "fixture-v1",
        ), 1.0)

class RepairEditor:
    def __init__(self): self.targets = []
    async def edit(self, *, artifact, target_unit_ids, instruction, evidence):
        self.targets.append(target_unit_ids)
        old = artifact.blocks[0].units[0]
        new = replace(old, text="Attention is used in place of recurrence.")
        repaired = replace(
            artifact, version=2,
            blocks=(replace(artifact.blocks[0], units=(new,)),),
        )
        return EditResult(
            repaired, (UnitDelta("claim", old.text, new.text),),
            "Removed the unsupported wording.", "fixture-editor", "fixture-v1",
        )

class ArtifactRepo:
    def __init__(self): self.saved = []
    def save_initial(self, **kwargs): self.saved.append(kwargs)

class UnusedEmbedder:
    async def embed_query(self, text): raise AssertionError("small document should use full coverage")

class RecordingQueryEmbedder:
    def __init__(self, model_id="fixture"): self.model_id, self.queries = model_id, []
    async def embed_query(self, text):
        from backend.app.embeddings import EmbeddingBatch
        self.queries.append(text)
        return EmbeddingBatch(self.model_id, 768, (tuple([1.0] * 768),), 1.0)

class CapturingGenerator(Generator):
    async def generate(self, request, evidence):
        self.evidence = evidence
        return await super().generate(request, evidence)

class GenerationWorkflowTests(unittest.TestCase):
    def setUp(self):
        document = extract_summary_bytes(
            b"# Finding\n\nAttention replaces recurrence.\n", filename="workflow.md"
        )
        self.chunks = build_chunks(document)
        self.initial = {
            "session_id": "session", "turn_id": "turn", "tenant_id": "tenant",
            "user_id": "user", "document_version": document.document_version,
            "output_specification": "a custom evidence card",
        }

    def test_connected_graph_selects_generates_verifies_and_commits(self):
        artifacts = ArtifactRepo()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=Generator(), verifier=Verifier(), artifact_repository=artifacts,
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertTrue(result["committed"])
        self.assertEqual(result["selection_mode"], "whole_document_coverage")
        self.assertEqual(len(artifacts.saved), 1)

    def _oversized_chunks(self):
        text = "\n\n".join(
            f"Finding {index} " + ("describes a distinct measured result. " * 26)
            for index in range(80)
        )
        return build_chunks(extract_summary_bytes(text.encode(), filename="long.md"))

    def test_whole_source_is_passed_without_searching_when_it_fits(self):
        generator = CapturingGenerator()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=Verifier(), artifact_repository=ArtifactRepo(),
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertEqual(result["selection_mode"], "whole_document_coverage")
        self.assertEqual(generator.evidence, self.chunks.chunks)

    def test_source_above_the_budget_is_searched_by_content_focus_and_stays_bounded(self):
        chunks = self._oversized_chunks()
        self.assertGreater(sum(len(c.text) for c in chunks.chunks), 60_000)
        embedder, generator = RecordingQueryEmbedder(), CapturingGenerator()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(chunks), embedder=embedder,
            generator=generator, verifier=Verifier(), artifact_repository=ArtifactRepo(),
        )
        result = asyncio.run(graph.ainvoke({
            **self.initial, "document_version": chunks.document_version,
            "evidence_focus": "measured results",
        }))
        self.assertEqual(result["selection_mode"], "hybrid_relevance_plus_structural_coverage")
        self.assertEqual(embedder.queries, ["measured results"])
        self.assertLessEqual(sum(len(c.text) for c in generator.evidence), 60_000)
        ordinals = [c.ordinal for c in generator.evidence]
        self.assertEqual(ordinals, sorted(ordinals))
        self.assertIn(0, ordinals)
        self.assertIn(len(chunks.chunks) - 1, ordinals)

    def test_retrieval_failure_is_reported_as_its_own_stage(self):
        chunks = self._oversized_chunks()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(chunks),
            embedder=RecordingQueryEmbedder(model_id="a-different-model"),
            generator=Generator(), verifier=Verifier(), artifact_repository=ArtifactRepo(),
        )
        with self.assertRaisesRegex(RuntimeError, "retrieve stage failed.*different models"):
            asyncio.run(graph.ainvoke({
                **self.initial, "document_version": chunks.document_version,
            }))

    def test_missing_number_citation_is_completed_recorded_and_stored_identically(self):
        document = extract_summary_bytes(
            b"# Results\n\nThe model reached 28.4 BLEU.\n\nTraining took 3.5 days.\n",
            filename="numbers.md",
        )
        chunks = build_chunks(document)
        score, days = chunks.chunks

        class TwoFactGenerator:
            async def generate(self, request, evidence):
                unit = GroundedText(
                    "claim", "claim", "The model reached 28.4 BLEU after 3.5 days.", True,
                    (SupportReference(score.chunk_id, (score.evidence[0].evidence_id,)),),
                )
                return GenerationResult(GeneratedArtifact(
                    "artifact", 1, request, (ArtifactBlock("block", "card", (unit,)),),
                    "fixture-generator", "fixture-v1",
                ), 1.0)

        artifacts = ArtifactRepo()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(chunks), embedder=UnusedEmbedder(),
            generator=TwoFactGenerator(), verifier=Verifier(), artifact_repository=artifacts,
        )
        result = asyncio.run(graph.ainvoke({
            **self.initial, "document_version": document.document_version,
        }))
        self.assertTrue(result["committed"])
        self.assertEqual(result["citation_completions"], [
            {"unit_id": "claim", "chunk_id": days.chunk_id, "figure": "3.5"},
        ])
        saved = artifacts.saved[0]
        cited = [ref.chunk_id for ref in saved["artifact"].blocks[0].units[0].support]
        self.assertEqual(cited, [score.chunk_id, days.chunk_id])
        self.assertEqual(
            saved["operation_result"]["value"]["citation_completions"],
            result["citation_completions"],
        )

    def test_number_absent_from_the_source_gets_the_one_rewrite_with_its_name(self):
        class WrongThenRight(Generator):
            def __init__(self): self.requests = []
            async def generate(self, request, evidence):
                self.requests.append(request)
                result = await super().generate(request, evidence)
                if len(self.requests) > 1:
                    return result
                unit = replace(result.artifact.blocks[0].units[0], text="Attention gives a 40% gain.")
                return GenerationResult(replace(
                    result.artifact, blocks=(replace(result.artifact.blocks[0], units=(unit,)),),
                ), 1.0)

        generator = WrongThenRight()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=Verifier(), artifact_repository=ArtifactRepo(),
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertTrue(result["committed"])
        self.assertEqual(len(generator.requests), 2)
        self.assertIn(
            "unsupported numeric tokens: 40%",
            " ".join(generator.requests[1].revision_feedback),
        )
        self.assertEqual(generator.requests[1].output.requirements, ())

    def test_failed_semantic_support_never_commits(self):
        artifacts = ArtifactRepo()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=Generator(), verifier=Verifier("unsupported"), artifact_repository=artifacts,
        )
        with self.assertRaisesRegex(RuntimeError, "repair stage failed.*support verification failed"):
            asyncio.run(graph.ainvoke(self.initial))
        self.assertEqual(artifacts.saved, [])

    def test_repairs_only_rejected_units_reverifies_once_and_commits_version_one(self):
        artifacts = ArtifactRepo()
        verifier = SequenceVerifier(("unsupported", "supported"))
        editor = RepairEditor()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=Generator(), verifier=verifier, artifact_repository=artifacts,
            editor=editor,
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertTrue(result["committed"])
        self.assertEqual(editor.targets, [("claim",)])
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(result["artifact"].version, 1)
        self.assertTrue(result["repair"]["attempted"])
        self.assertEqual(artifacts.saved[0]["artifact"].version, 1)

    def test_failed_repair_verification_never_commits_or_loops(self):
        artifacts = ArtifactRepo()
        verifier = SequenceVerifier(("unsupported", "partially_supported"))
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=Generator(), verifier=verifier, artifact_repository=artifacts,
            editor=RepairEditor(),
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertEqual(len(verifier.calls), 2)
        self.assertEqual(artifacts.saved, [])
        self.assertEqual(result["outcome_status"], "blocked")
        self.assertEqual(result["blocked"]["category"], "support_rejected")

    def run_shaped(self, generator, constraints, verifier=None):
        artifacts, verifier = ArtifactRepo(), verifier or CountingVerifier()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=verifier, artifact_repository=artifacts,
        )
        result = asyncio.run(graph.ainvoke({
            **self.initial, "user_request": "the user's own words",
            "constraints": tuple(constraints),
        }))
        return result, artifacts, verifier

    def test_reviewer_opinion_is_advice_and_never_triggers_a_rewrite(self):
        artifacts = ArtifactRepo()
        verifier = QualitySequenceVerifier()
        generator = InvalidQualityRewriteGenerator()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=verifier, artifact_repository=artifacts,
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertTrue(result["committed"])
        self.assertEqual((generator.calls, len(verifier.calls)), (1, 1))
        self.assertIsNone(result.get("regeneration"))
        stored = artifacts.saved[0]["operation_result"]["value"]["verification"]
        self.assertTrue(stored["accepted"])
        self.assertFalse(stored["fulfills_request"])
        self.assertEqual(list(stored["quality_issues"]), ["Rewrite as ready-to-use prose."])

    def test_the_writer_receives_the_users_own_words_and_the_exact_targets(self):
        generator = ShapedGenerator([225])
        self.run_shaped(generator, [DeliverableConstraint("duration_seconds", 90, "90-second")])
        output = generator.requests[0].output
        self.assertEqual(output.original_request, "the user's own words")
        self.assertEqual(output.format, "a custom evidence card")
        self.assertEqual(output.requirements, ())
        prompt = _build_prompt(generator.requests[0], self.chunks.chunks)
        self.assertIn("<user_request>\nthe user's own words\n</user_request>", prompt)
        self.assertIn(
            '- About 225 spoken words in total, at least 192 and at most 258 '
            '(90 seconds at 150 words per minute) (from "90-second")', prompt,
        )
        self.assertIn("<previous_draft>\nNone.\n</previous_draft>", prompt)

    def test_a_missed_count_gets_one_rewrite_with_the_measured_numbers(self):
        generator = ShapedGenerator([5, 5], [5, 5, 5])
        result, artifacts, verifier = self.run_shaped(
            generator, [DeliverableConstraint("item_count", 3, "three cards")]
        )
        self.assertTrue(result["committed"])
        self.assertEqual(verifier.calls, 1)
        self.assertEqual(generator.requests[0].revision_feedback, ())
        self.assertEqual(generator.requests[1].revision_feedback, (
            "Required: exactly 3 items, one block per item. The draft has 2 items. "
            "Add 1 item: return exactly 3 blocks.",
        ))
        self.assertIsNone(generator.requests[0].previous_draft)
        previous = json.loads(generator.requests[1].previous_draft)
        self.assertEqual([block["block_id"] for block in previous], ["block-1", "block-2"])
        self.assertEqual(generator.requests[1].output, generator.requests[0].output)
        stored = artifacts.saved[0]["operation_result"]["value"]
        self.assertEqual(
            [(item["kind"], item["measured"], item["met"]) for item in stored["constraints"]["checked"]],
            [("item_count", "3 items", True)],
        )
        self.assertEqual(stored["regeneration"], result["regeneration"])

    def test_a_script_outside_the_spoken_band_is_rewritten_to_length(self):
        generator = ShapedGenerator([156], [225])
        result, _, _ = self.run_shaped(
            generator, [DeliverableConstraint("duration_seconds", 90, "90-second")]
        )
        self.assertTrue(result["committed"])
        self.assertEqual(generator.requests[1].revision_feedback, (
            "Required: about 225 spoken words in total, at least 192 and at most 258 "
            "(90 seconds at 150 words per minute). The draft has 156 spoken words "
            "(about 62 seconds). Add about 69 spoken words in total. Spread the change "
            "through the text, and add only explanation of what the source says, never "
            "new facts.",
        ))
        self.assertEqual(
            result["constraint_report"][0]["measured"], "225 spoken words (about 90 seconds)"
        )

    def test_a_spoken_length_still_short_after_the_rewrite_is_saved_with_the_miss_shown(self):
        generator = ShapedGenerator([141], [170])
        result, artifacts, verifier = self.run_shaped(
            generator, [DeliverableConstraint("duration_seconds", 90, "90-second")]
        )
        self.assertTrue(result["committed"])
        self.assertEqual(verifier.calls, 1)
        checked = artifacts.saved[0]["operation_result"]["value"]["constraints"]["checked"]
        self.assertEqual(
            [(item["met"], item["enforced"], item["measured"]) for item in checked],
            [(False, False, "170 spoken words (about 68 seconds)")],
        )
        self.assertTrue(checked[0]["fix"].startswith("Add about 55 spoken words"))

    def test_a_stated_limit_still_missed_is_blocked_without_paying_for_review(self):
        generator = ShapedGenerator([5, 5], [5, 5, 5, 5])
        result, artifacts, verifier = self.run_shaped(
            generator, [DeliverableConstraint("item_count", 3, "three cards")]
        )
        self.assertEqual(result["outcome_status"], "blocked")
        self.assertEqual(result["blocked"]["category"], "constraint_not_met")
        self.assertEqual(result["blocked"]["quality_issues"], [
            "Required: exactly 3 items, one block per item. The draft has 4 items.",
        ])
        self.assertEqual((len(generator.requests), verifier.calls), (2, 0))
        self.assertEqual(artifacts.saved, [])

    def test_an_implied_limit_still_missed_is_reported_not_enforced(self):
        generator = ShapedGenerator([60], [50])
        result, artifacts, verifier = self.run_shaped(generator, [
            DeliverableConstraint("max_words", 40, "tweet-sized", "each_item", "implied"),
        ])
        self.assertTrue(result["committed"])
        self.assertEqual((len(generator.requests), verifier.calls), (2, 1))
        checked = artifacts.saved[0]["operation_result"]["value"]["constraints"]["checked"]
        self.assertEqual(
            [(item["basis"], item["measured"], item["met"]) for item in checked],
            [("implied", "item 1 with 50 words", False)],
        )

    def test_an_invalid_rewrite_keeps_the_first_draft_and_its_measured_problem(self):
        class ThenInvalid(ShapedGenerator):
            async def generate(self, request, evidence):
                result = await super().generate(request, evidence)
                if len(self.requests) == 1:
                    return result
                unit = replace(
                    result.artifact.blocks[0].units[0],
                    support=(SupportReference("missing-chunk", ("missing-evidence",)),),
                )
                return GenerationResult(replace(
                    result.artifact, blocks=(replace(result.artifact.blocks[0], units=(unit,)),),
                ), 1.0)

        generator = ThenInvalid([5, 5], [5, 5, 5])
        result, artifacts, verifier = self.run_shaped(
            generator, [DeliverableConstraint("item_count", 3, "three cards")]
        )
        self.assertEqual(result["outcome_status"], "blocked")
        self.assertEqual(result["blocked"]["category"], "constraint_not_met")
        self.assertFalse(result["regeneration"]["produced_valid_draft"])
        self.assertEqual(len(result["artifact"].blocks), 2)
        self.assertEqual((verifier.calls, artifacts.saved), (0, []))

    def test_a_rejected_figure_cannot_become_requested_through_the_rewrite(self):
        class AlwaysInvents(Generator):
            def __init__(self): self.requests = []
            async def generate(self, request, evidence):
                self.requests.append(request)
                result = await super().generate(request, evidence)
                unit = replace(result.artifact.blocks[0].units[0], text="Attention gives a 40% gain.")
                return GenerationResult(replace(
                    result.artifact, blocks=(replace(result.artifact.blocks[0], units=(unit,)),),
                ), 1.0)

        generator = AlwaysInvents()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=Verifier(), artifact_repository=ArtifactRepo(),
        )
        with self.assertRaisesRegex(RuntimeError, "regenerate stage failed.*numeric tokens: 40%"):
            asyncio.run(graph.ainvoke(self.initial))
        self.assertEqual(len(generator.requests), 2)

    def test_invalid_first_draft_gets_one_contract_rewrite_before_verification(self):
        artifacts = ArtifactRepo()
        generator = InvalidFirstDraftGenerator()
        verifier = Verifier()
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=verifier, artifact_repository=artifacts,
        )
        result = asyncio.run(graph.ainvoke(self.initial))
        self.assertEqual(generator.calls, 2)
        self.assertTrue(result["regeneration"]["produced_valid_draft"])
        self.assertTrue(result["committed"])

    def test_repair_runs_at_most_once_even_after_a_rewrite(self):
        generator = ShapedGenerator([5, 5], [5, 5, 5])
        artifacts = ArtifactRepo()

        class Editor:
            calls = 0
            async def edit(self, *, artifact, target_unit_ids, instruction, evidence):
                Editor.calls += 1
                old = artifact.blocks[0].units[0]
                new = replace(old, text="attention is used")
                blocks = (replace(artifact.blocks[0], units=(new,)), *artifact.blocks[1:])
                return EditResult(
                    replace(artifact, version=2, blocks=blocks),
                    (UnitDelta(old.unit_id, old.text, new.text),), "Weakened", "editor", "edit-v1",
                )

        class AllUnits(SequenceVerifier):
            async def verify(self, artifact, evidence):
                self.calls.append(artifact)
                return VerificationReport(
                    (ClaimVerdict("unit-1", self.verdicts.pop(0), "fixture feedback", ()),),
                    "fixture-verifier", "fixture-v1", 1.0,
                )

        verifier = AllUnits(("unsupported", "unsupported"))
        graph = build_generation_execution_graph(
            document_repository=DocumentRepo(self.chunks), embedder=UnusedEmbedder(),
            generator=generator, verifier=verifier, artifact_repository=artifacts,
            editor=Editor(),
        )
        result = asyncio.run(graph.ainvoke({
            **self.initial,
            "constraints": (DeliverableConstraint("item_count", 3, "three cards"),),
        }))
        self.assertEqual((len(generator.requests), Editor.calls, len(verifier.calls)), (2, 1, 2))
        self.assertEqual(result["blocked"]["category"], "support_rejected")
        self.assertEqual(artifacts.saved, [])


def deck():
    document = extract_summary_bytes(
        b"# Result\n\nTraining took 3.5 days on eight P100 GPUs.\n", filename="bases.md"
    )
    chunk = build_chunks(document).chunks[0]
    request = GenerationRequest(
        document.document_id, document.document_version, OutputSpecification("a short talk"),
    )
    return chunk, request

def draft(chunk, request, *, background_text="A GPU is a processor built for parallel arithmetic."):
    support = (SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,)),)
    return GeneratedArtifact("artifact", 1, request, (ArtifactBlock("talk", "segment", (
        GroundedText("hook", "hook", "Ever wondered what training a model costs?", False),
        GroundedText("explain", "explanation", background_text, False, basis="background"),
        GroundedText("claim", "claim", "Training took 3.5 days.", True, support),
    )),), "fixture", "fixture-v1")

class Transport:
    def __init__(self, payload): self.payload, self.prompts = payload, []
    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.payload

def review(uncited):
    return {
        "verdicts": [{"unit_id": "claim", "verdict": "supported", "reason": "Stated.",
                      "supporting_quotes": []}],
        "uncited_verdicts": [
            {"unit_id": unit_id, "verdict": verdict, "reason": "fixture"}
            for unit_id, verdict in uncited
        ],
    }

class ThreeBasesTests(unittest.TestCase):
    def test_every_unit_has_one_basis_and_old_revisions_still_load(self) -> None:
        chunk, request = deck()
        units = draft(chunk, request).blocks[0].units
        self.assertEqual([unit.basis for unit in units], ["framing", "background", "source"])
        with self.assertRaisesRegex(ValueError, "factual exactly when its basis is source"):
            GroundedText("x", "claim", "text", False, basis="source")
        with self.assertRaisesRegex(ValueError, "unsupported unit basis"):
            GroundedText("x", "claim", "text", False, basis="opinion")
        stored = json.loads(json.dumps(asdict(draft(chunk, request))))
        for unit in stored["blocks"][0]["units"]:
            del unit["basis"]
        restored = artifact_from_dict(stored)
        self.assertEqual([unit.basis for unit in restored.blocks[0].units],
                         ["framing", "framing", "source"])

    def test_the_writer_labels_units_and_background_carries_no_citation(self) -> None:
        chunk, request = deck()
        payload = {"blocks": [{"block_id": "talk", "block_type": "segment", "title": None, "units": [
            {"unit_id": "explain", "role": "explanation", "basis": "background",
             "text": "A GPU runs many calculations at once.",
             "support": [{"chunk_id": chunk.chunk_id}], "contains_math": False, "delivery": "spoken"},
            {"unit_id": "claim", "role": "claim", "basis": "source",
             "text": "Training took 3.5 days.", "support": [{"chunk_id": chunk.chunk_id}],
             "contains_math": False, "delivery": "spoken"},
        ]}]}

        class Writer:
            async def generate(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return ProviderResponse(payload)

        writer = Writer()
        result = asyncio.run(GeminiGroundedGenerator(writer).generate(request, (chunk,)))
        explain, claim = result.artifact.blocks[0].units
        self.assertEqual((explain.basis, explain.factual, explain.support), ("background", False, ()))
        self.assertEqual((claim.basis, claim.factual, len(claim.support)), ("source", True, 1))
        self.assertIn("Use background generously", writer.prompt)

    def test_the_checker_reviews_uncited_text_as_general_knowledge(self) -> None:
        chunk, request = deck()
        transport = Transport(review([("hook", "acceptable"), ("explain", "acceptable")]))
        report = asyncio.run(GeminiSupportVerifier(transport).verify(draft(chunk, request), (chunk,)))
        self.assertTrue(report.accepted)
        self.assertIn('"unit_id": "explain", "basis": "background"', transport.prompts[0])
        self.assertEqual([item.unit_id for item in report.uncited_verdicts], ["hook", "explain"])

    def test_a_claim_about_the_work_disguised_as_background_is_caught(self) -> None:
        chunk, request = deck()
        disguised = draft(chunk, request, background_text="It needed 8 GPUs, which is a lot.")
        transport = Transport(review([("hook", "acceptable"), ("explain", "needs_source")]))
        report = asyncio.run(GeminiSupportVerifier(transport).verify(disguised, (chunk,)))
        self.assertIn('"figures_found_in_the_source": ["8"]', transport.prompts[0])
        self.assertFalse(report.accepted)
        self.assertEqual([(item.unit_id, item.verdict) for item in report.rejected],
                         [("explain", "needs_source")])

    def test_every_uncited_unit_must_be_reviewed(self) -> None:
        chunk, request = deck()
        transport = Transport(review([("hook", "acceptable")]))
        with self.assertRaisesRegex(ValueError, "every uncited unit"):
            asyncio.run(GeminiSupportVerifier(transport).verify(draft(chunk, request), (chunk,)))

    def test_repair_targets_uncited_text_that_needs_a_source(self) -> None:
        report = VerificationReport(
            (ClaimVerdict("claim", "supported", "ok"),), "m", "v", 1.0,
            uncited_verdicts=(ClaimVerdict("explain", "needs_source", "about this work"),
                              ClaimVerdict("hook", "acceptable", "framing")),
        )
        self.assertFalse(report.accepted)
        self.assertEqual([item.unit_id for item in report.rejected], ["explain"])


DECKS = Path(__file__).resolve().parents[1] / "eval" / "datasets" / "decks"

def deck_number_rule(name: str):
    document = extract_pptx_bytes((DECKS / f"{name}.pptx").read_bytes(), filename=f"{name}.pptx")
    return document, build_chunks(document).chunks

def bullet(chunks, phrase: str):
    return next(chunk for chunk in chunks if phrase in " ".join(chunk.text.split()))

def artifact(document, sentences, *, request="a briefing"):
    """`sentences` is a list of (text, cited chunks)."""
    units = tuple(
        GroundedText(
            f"unit-{index}", "sentence", text, True,
            tuple(
                SupportReference(chunk.chunk_id, tuple(ref.evidence_id for ref in chunk.evidence))
                for chunk in cited
            ),
        )
        for index, (text, cited) in enumerate(sentences, start=1)
    )
    return GeneratedArtifact(
        "artifact", 1,
        GenerationRequest(document.document_id, document.document_version, OutputSpecification(request)),
        (ArtifactBlock("block", "section", units),), "fixture", "fixture-v1",
    )

class NumberRuleTests(unittest.TestCase):
    def test_digits_match_a_source_that_spells_its_numbers_out(self) -> None:
        document, chunks = deck_number_rule("bert")
        draft = artifact(document, [(
            "BERT-large has 340 million parameters and scores 80.5% on GLUE.",
            [bullet(chunks, "three hundred forty million"), bullet(chunks, "eighty point five percent")],
        )])
        self.assertEqual(figure_findings(draft, chunks), ())
        validate_deterministic_support(draft, chunks)

    def test_values_are_compared_not_characters(self) -> None:
        document, chunks = deck_number_rule("attention")
        training = bullet(chunks, "3.5 days on eight P 100 GPUs")
        validate_deterministic_support(
            artifact(document, [("Training took 3.5 days on 8 GPUs.", [training])]), chunks
        )
        with self.assertRaisesRegex(ValueError, "unsupported numeric tokens: 5$"):
            validate_deterministic_support(
                artifact(document, [("Training took 5 days.", [training])]), chunks
            )

    def test_a_missing_citation_is_completed_when_the_figure_is_on_a_cited_slide(self) -> None:
        document, chunks = deck_number_rule("attention")
        german = bullet(chunks, "28.4 BLEU")
        french = bullet(chunks, "41.8 BLEU")
        draft = artifact(document, [(
            "The big Transformer reached 28.4 BLEU on English-to-German and 41.8 BLEU "
            "on English-to-French.",
            [german],
        )])
        self.assertEqual(figure_findings(draft, chunks)[0].cited_elsewhere, ("41.8",))
        completed, added = complete_numeric_citations(draft, chunks)
        self.assertEqual([(item.unit_id, item.chunk_id, item.figure) for item in added],
                         [("unit-1", french.chunk_id, "41.8")])
        cited = [ref.chunk_id for ref in completed.blocks[0].units[0].support]
        self.assertEqual(cited, [german.chunk_id, french.chunk_id])
        self.assertEqual(figure_findings(completed, chunks), ())
        self.assertEqual(complete_numeric_citations(completed, chunks), (completed, ()))

    def test_a_figure_only_one_unit_holds_is_cited_even_from_another_slide(self) -> None:
        document, chunks = deck_number_rule("bert")
        motivation = bullet(chunks, "process text from left to right")
        parameters = bullet(chunks, "three hundred forty million")
        draft = artifact(document, [(
            "Left-to-right models limit context, which the 340 million parameter model addresses.",
            [motivation],
        )])
        completed, added = complete_numeric_citations(draft, chunks)
        self.assertEqual([item.chunk_id for item in added], [parameters.chunk_id])
        validate_deterministic_support(completed, chunks)

    def test_an_ambiguous_figure_is_left_for_the_verifier_and_reported_to_it(self) -> None:
        document, chunks = deck_number_rule("attention")
        faster = bullet(chunks, "enables significantly faster training")
        draft = artifact(document, [("The 2014 benchmarks showed faster training.", [faster])])
        completed, added = complete_numeric_citations(draft, chunks)
        self.assertEqual(added, ())
        self.assertEqual(figure_findings(completed, chunks)[0].cited_elsewhere, ("2014",))
        validate_deterministic_support(completed, chunks)

        class Capture:
            async def generate(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return {"verdicts": [{
                    "unit_id": "unit-1", "verdict": "partially_supported",
                    "reason": "The year is not in the cited evidence.",
                    "supporting_quotes": [],
                }]}

        transport = Capture()
        asyncio.run(GeminiSupportVerifier(transport).verify(completed, chunks))
        self.assertIn('"figures_not_in_cited_evidence": ["2014"]', transport.prompt)

    def test_item_labels_and_figures_from_the_request_are_not_claims(self) -> None:
        document, chunks = deck_number_rule("attention")
        layers = bullet(chunks, "six identical layers")
        draft = artifact(document, [
            ("Note 1: Both the encoder and decoder consist of 6 identical layers.", [layers]),
            ("2. In the next 90 seconds, here is why those layers matter.", [layers]),
            ("(3) The stacks are identical.", [layers]),
        ], request="a 90-second script")
        self.assertEqual(figure_findings(draft, chunks), ())
        with self.assertRaisesRegex(ValueError, "unit-1 contains unsupported numeric tokens: 7"):
            validate_deterministic_support(
                artifact(document, [("Note 7: The stacks are identical.", [layers])]), chunks
            )

    def test_every_figure_absent_from_the_source_is_reported_in_one_message(self) -> None:
        document, chunks = deck_number_rule("attention")
        layers = bullet(chunks, "six identical layers")
        with self.assertRaises(ValueError) as raised:
            validate_deterministic_support(artifact(document, [
                ("The model has 12 layers and 99% accuracy.", [layers]),
                ("It was trained on 4.5 million sentence pairs.", [layers]),
            ]), chunks)
        message = str(raised.exception)
        self.assertIn("unit-1 contains unsupported numeric tokens: 12, 99%", message)
        self.assertIn("unit-2 contains unsupported numeric tokens: 4500000", message)

    def test_number_words_vague_amounts_and_designations_are_not_figures(self) -> None:
        document, chunks = deck_number_rule("bert")
        parameters = bullet(chunks, "three hundred forty million")
        validate_deterministic_support(artifact(document, [
            ("Larger models with hundreds of millions of parameters do better.", [parameters]),
            ("There are three reasons this matters, and it is one of the first.", [parameters]),
            ("It ran on P100 hardware with the v1.5 release, 8x faster.", [parameters]),
        ]), chunks)

    def test_source_locators_are_not_quantities(self) -> None:
        document, chunks = deck_number_rule("attention")
        layers = bullet(chunks, "six identical layers")
        validate_deterministic_support(artifact(document, [
            ("Slide 3 and Figure 2 describe the stacks.", [layers]),
        ]), chunks)

    def test_resnet_digits_read_one_by_one_are_matched(self) -> None:
        document, chunks = deck_number_rule("resnet")
        draft = artifact(document, [(
            "The 152-layer ResNet reaches 4.49% top-5 error and the ensemble 3.57%.",
            [bullet(chunks, "152 layer ResNet"), bullet(chunks, "3 point 5 7 percent")],
        )])
        self.assertEqual(figure_findings(draft, chunks), ())


def limit(kind, value, quote, scope="whole", basis="stated"):
    return {"kind": kind, "value": value, "quote": quote, "scope": scope, "basis": basis}

def draft_constraints(constraints, blocks):
    """`blocks` is a list of blocks; each block is a list of (text, delivery)."""
    return GeneratedArtifact(
        "artifact", 1,
        GenerationRequest("document", "version", OutputSpecification(
            "a deliverable", constraints=tuple(constraints),
        )),
        tuple(
            ArtifactBlock(f"block-{b}", "item", tuple(
                GroundedText(f"unit-{b}-{u}", "body", text, False, delivery=delivery)
                for u, (text, delivery) in enumerate(units, start=1)
            ), title="A title that is not counted")
            for b, units in enumerate(blocks, start=1)
        ),
        "fixture", "fixture-v1",
    )

def words(count):
    return " ".join(["word"] * count)

class TracingLimitsToTheUsersWordsTests(unittest.TestCase):
    def test_limits_the_user_stated_are_kept_in_every_spelling(self) -> None:
        message = (
            "Create a seven-slide talk for a two-minute slot, then ten tweet-sized "
            "abstracts and 3 summaries under 120 words each."
        )
        kept, ignored = parse_constraints([
            limit("item_count", 7, "seven-slide talk"),
            limit("duration_seconds", 120, "two-minute slot"),
            limit("max_words", 120, "under 120 words each", "each_item"),
            limit("max_characters", 280, "tweet-sized", "each_item", "implied"),
        ], message)
        self.assertEqual(ignored, ())
        self.assertEqual(
            [(item.kind, item.value, item.scope, item.basis) for item in kept],
            [("item_count", 7, "whole", "stated"), ("duration_seconds", 120, "whole", "stated"),
             ("max_words", 120, "each_item", "stated"),
             ("max_characters", 280, "each_item", "implied")],
        )

    def test_a_limit_the_user_did_not_state_is_never_enforced(self) -> None:
        message = "Write a short script for policymakers about why this matters."
        kept, ignored = parse_constraints([
            limit("duration_seconds", 90, "short script"),
            limit("item_count", 5, "five takeaways"),
            limit("max_words", 0, "short script"),
            limit("words_per_line", 9, "short script"),
            "not an object",
        ], message)
        self.assertEqual(kept, ())
        self.assertEqual([item["reason"] for item in ignored], [
            "the quoted words do not state this number",
            "the quoted words are not in the user's message",
            "a constraint value must be a positive whole number",
            "unsupported constraint kind",
        ])

    def test_one_limit_per_kind_and_a_count_always_covers_the_whole(self) -> None:
        kept, ignored = parse_constraints([
            limit("item_count", 3, "3 posts", "each_item"),
            limit("item_count", 3, "3 posts"),
        ], "Write 3 posts.")
        self.assertEqual([(item.kind, item.scope) for item in kept], [("item_count", "whole")])
        self.assertEqual(
            ignored[0]["reason"], "another limit of the same kind was already accepted"
        )

    def test_absent_or_malformed_planner_output_means_no_limits(self) -> None:
        for raw in (None, "none", {"kind": "item_count"}, []):
            self.assertEqual(parse_constraints(raw, "Write a talk."), ((), ()))

class RequestedLengthTests(unittest.TestCase):
    def test_the_length_a_request_calls_for_is_read_from_its_limits(self) -> None:
        from backend.app.verification.constraints import requested_words

        def size(*limits):
            return requested_words(DeliverableConstraint(*item) for item in limits)

        self.assertIsNone(size())
        self.assertEqual(size(("duration_seconds", 90, "90-second")), 225)
        self.assertEqual(size(("duration_seconds", 1200, "20-minute")), 3000)
        self.assertEqual(size(("item_count", 5, "five slides")), 750)
        self.assertEqual(size(
            ("item_count", 4, "four scripts"),
            ("duration_seconds", 30, "30-second", "each_item"),
        ), 600)
        self.assertEqual(size(
            ("item_count", 10, "ten abstracts"),
            ("max_characters", 280, "tweet-sized", "each_item", "implied"),
        ), 1500)
        self.assertEqual(size(
            ("item_count", 7, "seven slides"), ("duration_seconds", 600, "10-minute"),
        ), 1500)

class MeasuringTests(unittest.TestCase):
    def test_items_are_blocks(self) -> None:
        count = DeliverableConstraint("item_count", 3, "three posts")
        two = measure_constraints(draft_constraints([count], [[("a", None)], [("b", None)]]))[0]
        self.assertEqual((two.met, two.required, two.measured),
                         (False, "exactly 3 items, one block per item", "2 items"))
        self.assertEqual(two.feedback, (
            "Required: exactly 3 items, one block per item. The draft has 2 items. "
            "Add 1 item: return exactly 3 blocks."
        ))
        three = draft_constraints([count], [[("a", None)], [("b", None)], [("c", None)]])
        self.assertTrue(measure_constraints(three)[0].met)

    def test_size_is_measured_per_item_on_unit_text_only(self) -> None:
        size = DeliverableConstraint("max_characters", 20, "20 characters", "each_item")
        result = measure_constraints(draft_constraints([size], [
            [("exactly ten", None), ("and nine", None)],
            [("this one is far too long", None)],
        ]))[0]
        self.assertFalse(result.met)
        self.assertEqual(result.measured, "item 2 with 24 characters")
        ok = measure_constraints(draft_constraints([size], [[("short", None)], [("also short", None)]]))[0]
        self.assertEqual(
            (ok.met, ok.measured),
            (True, "every item within the limit (largest: 10 characters)"),
        )

    def test_a_whole_limit_adds_every_item_together(self) -> None:
        total = DeliverableConstraint("max_words", 5, "5 words")
        result = measure_constraints(draft_constraints([total], [[(words(3), None)], [(words(3), None)]]))[0]
        self.assertEqual((result.met, result.measured), (False, "6 words"))

    def test_spoken_length_is_a_word_band_around_the_target(self) -> None:
        self.assertEqual(word_band(90), (192, 258))
        length = DeliverableConstraint("duration_seconds", 90, "90-second")
        for count, met in ((191, False), (192, True), (225, True), (258, True), (259, False)):
            result = measure_constraints(draft_constraints([length], [[(words(count), "spoken")]]))[0]
            self.assertEqual(result.met, met, count)
        short = measure_constraints(draft_constraints([length], [[(words(156), "spoken")]]))[0]
        self.assertEqual(short.measured, "156 spoken words (about 62 seconds)")
        self.assertEqual(short.fix, (
            "Add about 69 spoken words in total. Spread the change through the text, "
            "and add only explanation of what the source says, never new facts."
        ))
        long = measure_constraints(draft_constraints([length], [[(words(300), "spoken")]]))[0]
        self.assertTrue(long.fix.startswith("Cut about 75 spoken words in total."))
        within = measure_constraints(draft_constraints([length], [[(words(200), "spoken")]]))[0]
        self.assertEqual(within.fix, "")

    def test_only_text_marked_spoken_counts_when_any_is_marked(self) -> None:
        length = DeliverableConstraint("duration_seconds", 90, "90-second")
        slides_with_notes = draft_constraints([length], [
            [(words(40), None), (words(110), "Spoken")],
            [(words(40), None), (words(115), "spoken")],
        ])
        self.assertEqual(
            measure_constraints(slides_with_notes)[0].measured,
            "225 spoken words (about 90 seconds)",
        )
        unmarked = draft_constraints([length], [[(words(100), None)], [(words(125), None)]])
        self.assertTrue(measure_constraints(unmarked)[0].met)

    def test_spoken_length_can_apply_to_each_item(self) -> None:
        each = DeliverableConstraint("duration_seconds", 30, "30-second", "each_item")
        result = measure_constraints(draft_constraints([each], [
            [(words(75), "spoken")], [(words(40), "spoken")],
        ]))[0]
        self.assertEqual(result.measured, "item 2 with 40 spoken words (about 16 seconds)")

    def test_the_writer_is_told_the_same_words_the_server_measures(self) -> None:
        limits = (
            DeliverableConstraint("item_count", 10, "ten abstracts"),
            DeliverableConstraint("max_characters", 280, "tweet-sized", "each_item", "implied"),
        )
        self.assertEqual(describe_constraints(limits), (
            'Exactly 10 items, one block per item (from "ten abstracts")',
            'At most 280 characters in each item (from "tweet-sized")',
        ))
        self.assertEqual(
            [item.required for item in measure_constraints(draft_constraints(limits, [[("a", None)]]))],
            ["exactly 10 items, one block per item", "at most 280 characters in each item"],
        )

    def test_limits_and_the_original_request_survive_storage(self) -> None:
        from dataclasses import asdict
        import json

        original = GeneratedArtifact(
            "artifact", 1,
            GenerationRequest("document", "version", OutputSpecification(
                "a script", original_request="Write a 90-second script.",
                constraints=(DeliverableConstraint("duration_seconds", 90, "90-second"),),
            ), revision_feedback=("Required: more words.",)),
            (ArtifactBlock("block", "item", (GroundedText("unit", "body", "text", False),)),),
            "fixture", "fixture-v1",
        )
        restored = artifact_from_dict(json.loads(json.dumps(asdict(original))))
        self.assertEqual(restored, original)
        legacy = json.loads(json.dumps(asdict(original)))
        for key in ("original_request", "constraints"):
            del legacy["request"]["output"][key]
        del legacy["request"]["revision_feedback"]
        self.assertEqual(artifact_from_dict(legacy).request.output.constraints, ())


class ProviderError(Exception):
    def __init__(self, code): self.code = code

class Calls:
    """Fails with each given error in turn, then answers."""
    def __init__(self, *errors): self.errors, self.count, self.waits = list(errors), 0, []
    async def __call__(self):
        self.count += 1
        if self.errors:
            raise self.errors.pop(0)
        return "answer"
    async def sleep(self, seconds): self.waits.append(seconds)

class ProviderRetryTests(unittest.TestCase):
    def run_calls(self, calls):
        return asyncio.run(call_with_retry(calls, sleep=calls.sleep))

    def test_a_busy_provider_is_retried_with_growing_waits(self) -> None:
        calls = Calls(ProviderError(503), ProviderError(429))
        self.assertEqual(self.run_calls(calls), "answer")
        self.assertEqual(calls.count, 3)
        first, second = calls.waits
        self.assertTrue(1.0 <= first <= 1.5 and 3.0 <= second <= 4.5, calls.waits)

    def test_it_gives_up_after_three_attempts(self) -> None:
        calls = Calls(ProviderError(503), ProviderError(503), ProviderError(503))
        with self.assertRaises(ProviderError):
            self.run_calls(calls)
        self.assertEqual(calls.count, 3)

    def test_a_bad_request_or_invalid_answer_is_never_sent_again(self) -> None:
        for error in (ProviderError(400), ValueError("invalid model output")):
            calls = Calls(error)
            with self.assertRaises(type(error)):
                self.run_calls(calls)
            self.assertEqual((calls.count, calls.waits), (1, []))

    def test_a_slow_failure_is_not_repeated_but_is_safe_for_the_user_to_retry(self) -> None:
        for error in (TimeoutError(), ProviderError(504)):
            calls = Calls(error)
            with self.assertRaises(type(error)):
                self.run_calls(calls)
            self.assertEqual(calls.count, 1)
            self.assertTrue(is_transient(error))

    def test_a_stage_failure_says_whether_trying_again_can_help(self) -> None:
        self.assertTrue(WorkflowStageError("regenerate", ProviderError(503)).retryable)
        self.assertTrue(WorkflowStageError("generate", ProviderError(429)).retryable)
        self.assertFalse(WorkflowStageError("regenerate", ValueError("invalid draft")).retryable)


class PromptTemplateTests(unittest.TestCase):
    def test_prompt_is_directly_importable(self) -> None:
        self.assertIn("SARAL's research communication author", GENERATION_PROMPT)

    def test_renderer_rejects_missing_and_extra_variables(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing=.*SOURCE_EVIDENCE"):
            render_prompt(
                GENERATION_PROMPT,
                prompt_name="generation",
                OUTPUT_SPECIFICATION="briefing",
            )
        with self.assertRaisesRegex(ValueError, "extra=.*UNUSED"):
            render_prompt(
                GENERATION_PROMPT,
                prompt_name="generation",
                OUTPUT_SPECIFICATION="briefing",
                SOURCE_EVIDENCE="source",
                UNUSED="value",
            )

    def test_inserted_text_is_not_recursively_interpreted(self) -> None:
        user_text = "Explain {{SOURCE_EVIDENCE}} literally."
        rendered = render_prompt(
            GENERATION_PROMPT,
            prompt_name="generation",
            USER_REQUEST=user_text,
            OUTPUT_SPECIFICATION="an explanation",
            MEASURED_REQUIREMENTS="None.",
            REVISION_FEEDBACK="None. This is the first draft.",
            PREVIOUS_DRAFT="None.",
            CURRENT_DRAFT="None.",
            CONVERSATION="None.",
            SOURCE_EVIDENCE="trusted source text",
        )
        self.assertIn(user_text, rendered)
        self.assertIn("trusted source text", rendered)


if __name__ == "__main__":
    unittest.main()
