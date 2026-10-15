"""Editing a draft: targets, scoped edits, deltas and the Gemini editor."""

from __future__ import annotations

import asyncio
from dataclasses import replace
import unittest
from backend.app.chunking import Chunk, EvidenceReference
from backend.app.editing import EditResult, UnitDelta, VerifiedEditService, resolve_artifact_target
from backend.app.generation import (
    ArtifactBlock, GeneratedArtifact, GenerationRequest, GroundedText,
    OutputSpecification, SupportReference,
)
from backend.app.verification import ClaimVerdict, VerificationReport
from backend.app.editing import GeminiArtifactEditor
from backend.app.generation.providers.gemini import ProviderResponse
from backend.app.editing import artifact_outline
from backend.app.generation import (
    ArtifactBlock, DeliverableConstraint, GeneratedArtifact, GenerationRequest,
    GroundedText, OutputSpecification,
)
from backend.app.editing import EditResult, UnitDelta, resolve_artifact_target
from backend.app.generation import artifact_from_dict


def fixture():
    chunk = Chunk(
        "chunk", "document", "version", "slide-1", 0, "slide", "native",
        "Accuracy was 80 percent. The method uses attention.",
        "accuracy was 80 percent the method uses attention",
        (EvidenceReference("evidence", "source_block", slide_number=1),),
        "fixture", "section_paragraph", True,
    )
    support = (SupportReference("chunk", ("evidence",)),)
    artifact = GeneratedArtifact(
        "artifact", 1,
        GenerationRequest("document", "version", OutputSpecification("brief")),
        (
            ArtifactBlock("opening", "section", (
                GroundedText("point-1", "bullet", "Accuracy was 80 percent.", True, support),
                GroundedText("point-2", "bullet", "The method uses attention.", True, support),
            )),
            ArtifactBlock("closing", "section", (
                GroundedText("point-3", "takeaway", "Attention is the core method.", True, support),
            )),
        ),
        "fixture", "fixture-v1",
    )
    return artifact, (chunk,)

def change_unit(artifact, unit_id, text, *, version=2, change_other=None):
    blocks = []
    for block in artifact.blocks:
        units = []
        for unit in block.units:
            replacement = text if unit.unit_id == unit_id else unit.text
            if change_other and unit.unit_id == change_other[0]: replacement = change_other[1]
            units.append(replace(unit, text=replacement))
        blocks.append(replace(block, units=tuple(units)))
    return replace(artifact, version=version, blocks=tuple(blocks), model_id="editor", prompt_version="edit-v1")

class Editor:
    def __init__(self, factory): self.factory = factory
    async def edit(self, *, artifact, target_unit_ids, instruction, evidence):
        return self.factory(artifact, target_unit_ids)

class Verifier:
    def __init__(self, accepted=True): self.accepted = accepted
    async def verify(self, artifact, evidence):
        verdict = "supported" if self.accepted else "unsupported"
        return VerificationReport(
            tuple(ClaimVerdict(u.unit_id, verdict, "fixture") for b in artifact.blocks for u in b.units if u.factual),
            "fixture", "v1", 1.0,
        )

def valid_result(artifact, target_unit_ids):
    target = target_unit_ids[0]
    edited = change_unit(artifact, target, "The method relies on attention.")
    old = next(u.text for b in artifact.blocks for u in b.units if u.unit_id == target)
    return EditResult(edited, (UnitDelta(target, old, "The method relies on attention."),), "Simplified wording", "editor", "edit-v1")

class EditingReliabilityTests(unittest.TestCase):
    def setUp(self): self.artifact, self.evidence = fixture()

    def run_edit(self, editor, target="point-2", verifier=True):
        return asyncio.run(VerifiedEditService(editor=Editor(editor), verifier=Verifier(verifier)).edit(
            artifact=self.artifact, target_id=target, instruction="Make it clearer", evidence=self.evidence,
        ))

    def test_resolves_unit_block_artifact_and_normalized_ids(self):
        self.assertEqual(resolve_artifact_target(self.artifact, "POINT_2"), ("point-2",))
        self.assertEqual(resolve_artifact_target(self.artifact, "opening"), ("point-1", "point-2"))
        self.assertEqual(len(resolve_artifact_target(self.artifact, "artifact")), 3)
        self.assertEqual(len(resolve_artifact_target(self.artifact, "current artifact")), 3)

    def test_unknown_target_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError, "ambiguous or does not exist"):
            resolve_artifact_target(self.artifact, "the interesting one")

    def test_valid_scoped_edit_returns_exact_delta(self):
        result = self.run_edit(valid_result)
        self.assertEqual(result.artifact.version, 2)
        self.assertEqual(result.deltas[0].unit_id, "point-2")

    def test_out_of_scope_change_is_rejected(self):
        def malicious(artifact, targets):
            edited = change_unit(artifact, targets[0], "The method relies on attention.", change_other=("point-3", "Changed too."))
            return EditResult(edited, (UnitDelta(targets[0], "The method uses attention.", "The method relies on attention."),), "Changed", "editor", "v1")
        with self.assertRaisesRegex(ValueError, "outside the resolved target"):
            self.run_edit(malicious)

    def test_wrong_version_and_noop_are_rejected(self):
        def wrong_version(artifact, targets):
            result = valid_result(artifact, targets)
            return replace(result, artifact=replace(result.artifact, version=4))
        with self.assertRaisesRegex(ValueError, "increment one version"):
            self.run_edit(wrong_version)
        def noop(artifact, targets):
            return EditResult(replace(artifact, version=2), (), "No change", "editor", "v1")
        with self.assertRaisesRegex(ValueError, "no change"):
            self.run_edit(noop)

    def test_incomplete_or_incorrect_delta_is_rejected(self):
        def bad_delta(artifact, targets):
            result = valid_result(artifact, targets)
            return replace(result, deltas=(replace(result.deltas[0], old_text="wrong"),))
        with self.assertRaisesRegex(ValueError, "old text"):
            self.run_edit(bad_delta)

    def test_unsupported_number_is_rejected_before_semantic_verifier(self):
        def invented(artifact, targets):
            edited = change_unit(artifact, targets[0], "The method improves accuracy by 99%.")
            return EditResult(edited, (UnitDelta(targets[0], "The method uses attention.", "The method improves accuracy by 99%."),), "Added claim", "editor", "v1")
        with self.assertRaisesRegex(ValueError, "99%"):
            self.run_edit(invented)

    def test_edit_gains_a_missing_number_citation_only_on_the_unit_it_changed(self):
        def source(name, slide, text):
            return Chunk(
                name, "document", "version", f"slide-{slide}", slide - 1, "bullet", "native",
                text, text, (EvidenceReference(f"evidence-{name}", "source_block", slide_number=slide),),
                "fixture", "section_paragraph", True,
            )

        method = source("method", 1, "The method uses attention.")
        score = source("score", 2, "Accuracy was eighty percent.")
        cites_method = (SupportReference("method", ("evidence-method",)),)
        artifact = GeneratedArtifact(
            "artifact", 1,
            GenerationRequest("document", "version", OutputSpecification("brief")),
            (ArtifactBlock("opening", "section", (
                GroundedText("point-1", "bullet", "Attention reached 80 percent.", True, cites_method),
                GroundedText("point-2", "bullet", "The method uses attention.", True, cites_method),
            )),),
            "fixture", "fixture-v1",
        )

        def adds_a_figure(base, targets):
            edited = change_unit(base, "point-2", "Using attention, accuracy was 80%.")
            return EditResult(
                edited,
                (UnitDelta("point-2", "The method uses attention.", "Using attention, accuracy was 80%."),),
                "Added the result", "editor", "edit-v1",
            )

        result = asyncio.run(VerifiedEditService(editor=Editor(adds_a_figure), verifier=Verifier()).edit(
            artifact=artifact, target_id="point-2", instruction="Add the result",
            evidence=(method, score),
        ))
        units = {unit.unit_id: unit for block in result.artifact.blocks for unit in block.units}
        self.assertEqual([ref.chunk_id for ref in units["point-2"].support], ["method", "score"])
        self.assertEqual(units["point-1"], artifact.blocks[0].units[0])

    def test_semantic_rejection_blocks_edit(self):
        from backend.app.editing import EditRejectedError

        with self.assertRaises(EditRejectedError) as raised:
            self.run_edit(valid_result, verifier=False)
        self.assertIn("previous version is kept", raised.exception.user_message)
        self.assertEqual(raised.exception.details, ("point-2: fixture",))

    def test_only_the_units_an_edit_changed_are_reviewed(self):
        class Recording(Verifier):
            async def verify(self, artifact, evidence):
                self.seen = [u.unit_id for b in artifact.blocks for u in b.units]
                return await super().verify(artifact, evidence)

        verifier = Recording()
        asyncio.run(VerifiedEditService(editor=Editor(valid_result), verifier=verifier).edit(
            artifact=self.artifact, target_id="point-2", instruction="Clearer", evidence=self.evidence,
        ))
        self.assertEqual(verifier.seen, ["point-2"])

    def test_an_unknown_part_is_explained_not_guessed(self):
        from backend.app.editing import EditRejectedError

        with self.assertRaises(EditRejectedError) as raised:
            self.run_edit(valid_result, target="the interesting one")
        self.assertIn("couldn't tell which part", raised.exception.user_message)


class Transport:
    def __init__(self, payload): self.payload = payload; self.calls = []
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResponse(self.payload)

class GeminiEditingContractTests(unittest.TestCase):
    def run_editor(self, payload, targets=("point-2",)):
        artifact, evidence = fixture()
        transport = Transport(payload)
        result = asyncio.run(GeminiArtifactEditor(transport, model_id="fixture").edit(
            artifact=artifact, target_unit_ids=targets,
            instruction="Make the selected text clearer", evidence=evidence,
        ))
        return result, transport

    def test_valid_proposal_changes_only_selected_unit(self):
        result, transport = self.run_editor({
            "reason": "Used simpler wording.",
            "replacements": [{
                "unit_id": "point-2", "text": "The approach relies on attention.",
                "basis": "source",
                "support": [{"chunk_id": "chunk", "evidence_ids": ["evidence"]}],
            }],
        })
        self.assertEqual(result.artifact.version, 2)
        self.assertEqual(result.deltas[0].unit_id, "point-2")
        self.assertIn("point-2", transport.calls[0]["response_schema"]["properties"]["replacements"]["items"]["properties"]["unit_id"]["enum"])

    def test_extra_fields_fail_pydantic_boundary(self):
        with self.assertRaisesRegex(Exception, "extra"):
            self.run_editor({
                "reason": "Changed.", "unexpected": "unsafe",
                "replacements": [{"unit_id": "point-2", "text": "Clear.", "basis": "framing", "support": []}],
            })

    def test_duplicate_and_out_of_scope_units_are_rejected(self):
        duplicate = {"reason": "Changed.", "replacements": [
            {"unit_id": "point-2", "text": "Clear A.", "basis": "framing", "support": []},
            {"unit_id": "point-2", "text": "Clear B.", "basis": "framing", "support": []},
        ]}
        with self.assertRaisesRegex(ValueError, "duplicate or out-of-scope"):
            self.run_editor(duplicate)
        outside = {"reason": "Changed.", "replacements": [
            {"unit_id": "point-1", "text": "Clear.", "basis": "framing", "support": []},
        ]}
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            self.run_editor(outside)

    def test_cross_chunk_evidence_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside the active document"):
            self.run_editor({
                "reason": "Changed.", "replacements": [{
                    "unit_id": "point-2", "text": "Clear.", "basis": "source",
                    "support": [{"chunk_id": "chunk", "evidence_ids": ["invented"]}],
                }],
            })


def talk():
    request = GenerationRequest("document", "version", OutputSpecification(
        "a short talk",
        constraints=(DeliverableConstraint("duration_seconds", 30, "30-second"),),
    ))
    blocks = tuple(
        ArtifactBlock(f"slide-{number}", "slide", (
            GroundedText(f"slide-{number}-note", "speaker_note",
                         f"This is the note for slide {number}, read aloud to the room.", False),
        ), title=f"Slide {number}")
        for number in (1, 2, 3)
    )
    return GeneratedArtifact("artifact-1", 1, request, blocks, "fixture", "fixture-v1")

class EditTargetTests(unittest.TestCase):
    def test_the_outline_numbers_parts_in_reading_order(self) -> None:
        outline = artifact_outline(talk())
        self.assertEqual([(item["number"], item["block_id"], item["title"]) for item in outline],
                         [(1, "slide-1", "Slide 1"), (2, "slide-2", "Slide 2"), (3, "slide-3", "Slide 3")])
        self.assertEqual(outline[1]["units"][0]["starts"],
                         "This is the note for slide 2, read aloud to the room.")


class AuthoringReliabilityMatrixTests(unittest.TestCase):
    def setUp(self):
        self.artifact, _ = fixture()

    def test_target_alias_matrix(self):
        cases = {
            "point-1": ("point-1",),
            "POINT-1": ("point-1",),
            "point_1": ("point-1",),
            "  point-1  ": ("point-1",),
            "opening": ("point-1", "point-2"),
            "OPENING": ("point-1", "point-2"),
            "closing": ("point-3",),
            "artifact": ("point-1", "point-2", "point-3"),
            "current artifact": ("point-1", "point-2", "point-3"),
        }
        for target, expected in cases.items():
            with self.subTest(target=target):
                self.assertEqual(resolve_artifact_target(self.artifact, target), expected)

    def test_ambiguous_and_unknown_target_matrix(self):
        for target in ("", "point", "second one", "slide-2", "missing", "point-99", "artifact_artifact"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                resolve_artifact_target(self.artifact, target)

    def test_persisted_round_trip_matrix_preserves_domain_types(self):
        from dataclasses import asdict
        for version in (1, 2, 17):
            with self.subTest(version=version):
                candidate = replace(self.artifact, version=version)
                restored = artifact_from_dict(asdict(candidate))
                self.assertEqual(restored, candidate)
                self.assertIsInstance(restored.blocks, tuple)
                self.assertIsInstance(restored.blocks[0].units, tuple)

    def test_delta_contract_matrix(self):
        unit = self.artifact.blocks[0].units[0]
        variants = (
            "Accuracy was eighty percent.",
            "In plain English, accuracy was 80 percent.",
            "The reported accuracy was 80 percent.",
            "Accuracy: 80 percent.",
            "The source reports an accuracy of 80 percent.",
        )
        for text in variants:
            with self.subTest(text=text):
                edited = replace(unit, text=text)
                delta = UnitDelta(unit.unit_id, unit.text, edited.text)
                result = EditResult(
                    replace(
                        self.artifact, version=2,
                        blocks=(replace(self.artifact.blocks[0], units=(edited, self.artifact.blocks[0].units[1])), self.artifact.blocks[1]),
                    ),
                    (delta,), "Adapted wording", "fixture", "v1",
                )
                self.assertEqual(result.deltas[0].old_text, unit.text)
                self.assertEqual(result.deltas[0].new_text, text)


if __name__ == "__main__":
    unittest.main()
