"""Deterministic and semantic claim-support verification tests."""

import asyncio
import unittest
from dataclasses import replace

from backend.app.chunking import build_chunks
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference,
)
from backend.app.ingestion import extract_summary_bytes
from backend.app.verification import GeminiSupportVerifier, validate_deterministic_support
from backend.app.prompts import VERIFICATION_PROMPT


def fixture():
    document = extract_summary_bytes(
        b"# Result\n\nTraining took 3.5 days on eight P100 GPUs.\n",
        filename="verification.md",
    )
    chunk = build_chunks(document).chunks[0]
    support = SupportReference(chunk.chunk_id, (chunk.evidence[-1].evidence_id,))
    request = GenerationRequest(
        document.document_id, document.document_version,
        OutputSpecification(format="evidence card"),
    )
    return chunk, request, support


def artifact(text, support):
    _, request, _ = fixture()
    return GeneratedArtifact(
        "artifact-test", 1, request,
        (ArtifactBlock("block", "card", (GroundedText("claim", "claim", text, True, (support,)),)),),
        "fixture", "fixture-v1",
    )


class FakeTransport:
    def __init__(self, payload):
        self.payload = payload

    async def generate(self, **kwargs):
        return self.payload


class SequenceTransport:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []
    async def generate(self, **kwargs):
        self.prompts.append(kwargs["prompt"])
        return self.payloads.pop(0)


class VerificationTests(unittest.TestCase):
    def test_quality_prompt_rejects_meta_notes_and_fake_equation_preservation(self):
        self.assertIn("directions to a future writer or speaker", VERIFICATION_PROMPT)
        self.assertIn("incomplete symbol fragment", VERIFICATION_PROMPT)
        self.assertIn("claim/provenance boundaries", VERIFICATION_PROMPT)

    def test_numeric_claim_must_copy_number_from_cited_evidence(self):
        chunk, _, support = fixture()
        validate_deterministic_support(artifact("Training took 3.5 days.", support), (chunk,))
        with self.assertRaisesRegex(ValueError, "4.5"):
            validate_deterministic_support(artifact("Training took 4.5 days.", support), (chunk,))

    def test_source_locator_number_is_not_treated_as_research_quantity(self):
        chunk, _, support = fixture()
        validate_deterministic_support(
            artifact("Slide 3 reports that training took 3.5 days.", support), (chunk,)
        )
        with self.assertRaisesRegex(ValueError, "4"):
            validate_deterministic_support(
                artifact("Slide 3 reports a 4% improvement.", support), (chunk,)
            )

    def test_semantic_verifier_accepts_complete_supported_result(self):
        chunk, _, support = fixture()
        verifier = GeminiSupportVerifier(FakeTransport({"verdicts": [{
            "unit_id": "claim", "verdict": "supported",
            "reason": "The duration is stated directly.",
            "supporting_quotes": ["Training took 3.5 days"],
        }]}))
        report = asyncio.run(verifier.verify(artifact("Training took 3.5 days.", support), (chunk,)))
        self.assertTrue(report.accepted)

    def test_semantic_verifier_rejects_duplicate_ids_and_invented_quotes(self):
        chunk, _, support = fixture()
        bad_quote = GeminiSupportVerifier(FakeTransport({"verdicts": [{
            "unit_id": "claim", "verdict": "supported", "reason": "claimed",
            "supporting_quotes": ["dramatically reduced cost"],
        }]}))
        with self.assertRaisesRegex(ValueError, "not present"):
            asyncio.run(bad_quote.verify(artifact("Training took 3.5 days.", support), (chunk,)))

    def test_partial_support_blocks_acceptance(self):
        chunk, _, support = fixture()
        verifier = GeminiSupportVerifier(FakeTransport({"verdicts": [{
            "unit_id": "claim", "verdict": "partially_supported",
            "reason": "Time is supported but cost reduction is not.",
            "supporting_quotes": ["Training took 3.5 days"],
        }]}))
        report = asyncio.run(verifier.verify(
            artifact("Training took 3.5 days and dramatically reduced cost.", support), (chunk,)
        ))
        self.assertFalse(report.accepted)

    def test_each_requirement_gets_its_own_verdict(self):
        chunk, request, support = fixture()
        draft = artifact("Training took 3.5 days.", support)
        output = replace(request.output, requirements=("a speaker note for each slide", "for graduate students"))
        draft = replace(draft, request=replace(request, output=output))
        transport = SequenceTransport([{
            "verdicts": [{"unit_id": "claim", "verdict": "supported", "reason": "stated",
                          "supporting_quotes": []}],
            "requirement_verdicts": [
                {"requirement_id": "R2", "met": True, "reason": "Technical depth fits."},
                {"requirement_id": "R1", "met": False, "reason": "No slide has a separate note."},
            ],
            "fulfills_request": True,
        }])
        report = asyncio.run(GeminiSupportVerifier(transport).verify(draft, (chunk,)))
        self.assertIn('"requirement": "a speaker note for each slide"', transport.prompts[0])
        self.assertEqual([(item.requirement, item.met) for item in report.requirement_verdicts], [
            ("a speaker note for each slide", False), ("for graduate students", True),
        ])
        self.assertFalse(report.fulfills_request)
        self.assertEqual(report.quality_issues[0],
                         "Not met: a speaker note for each slide (No slide has a separate note.)")
        self.assertTrue(report.accepted)

    def test_a_missing_requirement_verdict_is_asked_for_again(self):
        chunk, request, support = fixture()
        draft = artifact("Training took 3.5 days.", support)
        draft = replace(draft, request=replace(request, output=replace(request.output, requirements=("short",))))
        verdict = {"unit_id": "claim", "verdict": "supported", "reason": "stated", "supporting_quotes": []}
        transport = SequenceTransport([
            {"verdicts": [verdict], "requirement_verdicts": []},
            {"verdicts": [verdict], "requirement_verdicts": [
                {"requirement_id": "R1", "met": True, "reason": "It is short."}]},
        ])
        report = asyncio.run(GeminiSupportVerifier(transport).verify(draft, (chunk,)))
        self.assertIn("every requirement", transport.prompts[1])
        self.assertTrue(report.requirement_verdicts[0].met)

    def test_quality_opinion_is_reported_but_does_not_decide_acceptance(self):
        chunk, _, support = fixture()
        verifier = GeminiSupportVerifier(FakeTransport({
            "verdicts": [{
                "unit_id": "claim", "verdict": "supported",
                "reason": "The duration is stated directly.",
                "supporting_quotes": [],
            }],
            "fulfills_request": False,
            "quality_reason": "The requested three summaries were not delivered.",
            "quality_issues": ["Only one of three requested items was produced."],
        }))
        report = asyncio.run(verifier.verify(
            artifact("Training took 3.5 days.", support), (chunk,)
        ))
        self.assertTrue(report.accepted)
        self.assertFalse(report.fulfills_request)
        self.assertEqual(report.quality_issues, (
            "Only one of three requested items was produced.",
        ))

    def test_open_ended_user_guidance_is_preserved_without_issue_taxonomy(self):
        chunk, _, support = fixture()
        verifier = GeminiSupportVerifier(FakeTransport({
            "verdicts": [{
                "unit_id": "claim", "verdict": "supported",
                "reason": "Directly stated.", "supporting_quotes": [],
            }],
            "fulfills_request": False,
            "quality_reason": "The requested form is not ready to use.",
            "quality_issues": ["The response does not fit the requested situation."],
            "guidance_summary": "The source supports the fact, but this draft does not yet work as the requested briefing.",
            "guidance_observations": ["The central fact is supported."],
            "guidance_attempts": ["The draft was checked for factual support and usability."],
            "suggested_actions": [{
                "label": "Refocus the briefing",
                "instruction": "Rewrite it around the practical decision the audience must make.",
            }],
            "can_retry_automatically": False,
        }))
        report = asyncio.run(verifier.verify(
            artifact("Training took 3.5 days.", support), (chunk,)
        ))
        self.assertIn("does not yet work", report.guidance.summary)
        self.assertEqual(report.guidance.suggested_actions[0].label, "Refocus the briefing")
        self.assertFalse(report.guidance.can_retry_automatically)

    def test_reported_quality_issue_overrides_inconsistent_true_flag(self):
        chunk, _, support = fixture()
        verifier = GeminiSupportVerifier(FakeTransport({
            "verdicts": [{
                "unit_id": "claim", "verdict": "supported",
                "reason": "The duration is stated directly.",
                "supporting_quotes": [],
            }],
            "fulfills_request": True,
            "quality_reason": "Mostly complete, with one remaining problem.",
            "quality_issues": ["The requested audience adaptation is missing."],
        }))
        report = asyncio.run(verifier.verify(
            artifact("Training took 3.5 days.", support), (chunk,)
        ))
        self.assertTrue(report.accepted)
        self.assertFalse(report.fulfills_request)

    def test_invalid_structured_response_gets_one_same_model_correction(self):
        chunk, _, support = fixture()
        transport = SequenceTransport([
            {"verdicts": []},
            {"verdicts": [{
                "unit_id": "claim", "verdict": "supported",
                "reason": "Directly stated.", "supporting_quotes": [],
            }], "fulfills_request": True, "quality_reason": "Complete.",
             "quality_issues": []},
        ])
        report = asyncio.run(GeminiSupportVerifier(transport).verify(
            artifact("Training took 3.5 days.", support), (chunk,)
        ))
        self.assertTrue(report.accepted)
        self.assertEqual(len(transport.prompts), 2)
        self.assertIn("STRUCTURED_OUTPUT_CORRECTION", transport.prompts[1])


if __name__ == "__main__":
    unittest.main()
