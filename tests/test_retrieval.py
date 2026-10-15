"""Offline tests for scoped lexical/dense retrieval and embedding boundaries."""

from __future__ import annotations

import asyncio
import unittest
from dataclasses import replace

from backend.app.chunking import build_chunks
from backend.app.embeddings import EmbeddingBatch, default_query_instruction
from backend.app.embeddings.gemini import GeminiEmbedding2, TransportEmbedding
from backend.app.embeddings.fastembed import FastEmbedder
from backend.app.ingestion import extract_summary_bytes
from backend.app.retrieval import (
    DeckRetriever,
    DenseIndex,
    LabeledScore,
    LexicalIndex,
    calibrate_zero_false_accept_threshold,
    cosine_similarity,
    evaluate_threshold,
    reciprocal_rank_fusion,
    SearchHit,
    select_generation_evidence,
    HybridHit,
    tokenize,
)


def _chunks():
    document = extract_summary_bytes(
        b"# Results\n\nProposed F1 was 0.84.\n\n# Limits\n\nSample size was 48.\n",
        filename="retrieval.md",
    )
    return document, build_chunks(document)


class FakeEmbeddingTransport:
    def __init__(self, *, bad_dimension: bool = False) -> None:
        self.calls = []
        self.bad_dimension = bad_dimension

    async def embed(self, *, model_id, text, dimensions):
        self.calls.append((model_id, text, dimensions))
        size = dimensions - 1 if self.bad_dimension else dimensions
        return TransportEmbedding(tuple([1.0] + [0.0] * (size - 1)), 5)


class SlowEmbeddingTransport:
    async def embed(self, **kwargs):
        await asyncio.sleep(0.05)
        return TransportEmbedding(tuple([1.0] * kwargs["dimensions"]), 1)


class FakeLocalBackend:
    embedding_size = 3

    def __init__(self):
        self.calls = []

    def passage_embed(self, texts, **kwargs):
        self.calls.append(("passage", tuple(texts), kwargs))
        return ([float(index), 0.0, 1.0] for index, _ in enumerate(texts, 1))

    def query_embed(self, query, **kwargs):
        self.calls.append(("query", tuple(query), kwargs))
        return ([9.0, 0.0, 1.0] for _ in query)


class RecordingEmbedder:
    """Returns chosen vectors and records which encoding path was used."""

    def __init__(self, *, query, passage, model_id="fixture-embedding") -> None:
        self._query, self._passage, self._model_id = query, passage, model_id
        self.calls: list[tuple[str, str]] = []

    async def embed_query(self, text):
        self.calls.append(("query", text))
        return EmbeddingBatch(self._model_id, 3, (tuple(self._query),), 1.0)

    async def embed_documents(self, texts):
        self.calls.append(("passage", texts[0]))
        return EmbeddingBatch(self._model_id, 3, (tuple(self._passage),), 1.0)


def _deck_like_chunks():
    document = extract_summary_bytes(
        b"# Results\n\nThe large model achieved an eighty point five percent score on "
        b"the G L U E benchmark.\n\nTraining took three days.\n\n"
        b"# Method\n\nBoth directions of context are used.\n\n"
        b"Sentence pairs teach relationships.\n",
        filename="deck_like.md",
    )
    return document, build_chunks(document)


class DeckRetrieverTests(unittest.TestCase):
    VECTORS = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.6, 0.0, 0.8))

    def _retriever(self, embedder):
        _, chunks = _deck_like_chunks()
        self.assertEqual(len(chunks.chunks), 4)
        return chunks, DeckRetriever(
            chunks, self.VECTORS, embedder=embedder,
            embedding_model_id="fixture-embedding",
        )

    def test_lookup_fuses_both_channels_and_encodes_the_question_as_a_query(self) -> None:
        embedder = RecordingEmbedder(query=(0.0, 1.0, 0.0), passage=(0.0, 0.0, 0.0))
        chunks, retriever = self._retriever(embedder)
        hits = asyncio.run(retriever.lookup("How long did training take?", top_k=2))
        self.assertEqual(hits[0].hit.chunk.text, "Training took three days.")
        self.assertEqual((hits[0].lexical_rank, hits[0].dense_rank), (1, 1))
        self.assertEqual([kind for kind, _ in embedder.calls], ["query"])

    def test_candidates_pool_each_channel_and_encode_the_sentence_as_a_passage(self) -> None:
        embedder = RecordingEmbedder(query=(0.0, 0.0, 0.0), passage=(0.0, 0.0, 1.0))
        chunks, retriever = self._retriever(embedder)
        pooled = asyncio.run(
            retriever.candidates("It scores 80.5% on GLUE.", per_channel=1)
        )
        self.assertEqual([kind for kind, _ in embedder.calls], ["passage"])
        self.assertEqual(embedder.calls[0][1], "It scores 80.5% on GLUE.")
        by_text = {item.chunk.text: item for item in pooled}
        glue = next(text for text in by_text if "eighty point five percent" in text)
        self.assertEqual(
            (by_text[glue].keyword_rank, by_text[glue].embedding_rank), (1, None)
        )
        context = by_text["Both directions of context are used."]
        self.assertEqual((context.keyword_rank, context.embedding_rank), (None, 1))
        self.assertEqual(len(pooled), 2)
        self.assertEqual(pooled[0].chunk.text, glue)

    def test_a_bullet_found_by_both_channels_appears_once(self) -> None:
        embedder = RecordingEmbedder(query=(0.0, 0.0, 0.0), passage=(0.0, 1.0, 0.0))
        _, retriever = self._retriever(embedder)
        pooled = asyncio.run(retriever.candidates("Training took three days.", per_channel=1))
        self.assertEqual(len(pooled), 1)
        self.assertEqual((pooled[0].keyword_rank, pooled[0].embedding_rank), (1, 1))

    def test_vectors_from_another_embedding_model_are_refused(self) -> None:
        embedder = RecordingEmbedder(
            query=(1.0, 0.0, 0.0), passage=(1.0, 0.0, 0.0), model_id="another-model"
        )
        _, retriever = self._retriever(embedder)
        with self.assertRaisesRegex(ValueError, "different models"):
            asyncio.run(retriever.lookup("training"))
        with self.assertRaisesRegex(ValueError, "different models"):
            asyncio.run(retriever.candidates("training"))


class RetrievalTests(unittest.TestCase):
    def test_bge_models_get_their_query_instruction_and_others_get_none(self) -> None:
        self.assertIn("searching relevant passages", default_query_instruction("BAAI/bge-base-en-v1.5"))
        self.assertIsNone(default_query_instruction("nomic-ai/nomic-embed-text-v1.5"))
        backend = FakeLocalBackend()
        embedder = FastEmbedder(
            backend=backend, dimensions=3,
            query_instruction=default_query_instruction("BAAI/bge-small-en-v1.5"),
        )
        asyncio.run(embedder.embed_query("find first"))
        asyncio.run(embedder.embed_documents(("find first",)))
        self.assertTrue(backend.calls[0][1][0].endswith("passages: find first"))
        self.assertEqual(backend.calls[1][1], ("find first",))

    def test_local_embedder_uses_asymmetric_retrieval_paths_and_preserves_order(self) -> None:
        backend = FakeLocalBackend()
        embedder = FastEmbedder(backend=backend, dimensions=3, batch_size=2)
        documents = asyncio.run(embedder.embed_documents(("first", "second")))
        query = asyncio.run(embedder.embed_query("find first"))

        self.assertEqual([vector[0] for vector in documents.vectors], [1.0, 2.0])
        self.assertEqual(query.vectors[0][0], 9.0)
        self.assertEqual(backend.calls[0], ("passage", ("first", "second"), {"batch_size": 2}))
        self.assertEqual(backend.calls[1][0:2], ("query", ("find first",)))
        self.assertEqual(documents.model_id, "BAAI/bge-base-en-v1.5")

    def test_local_embedder_rejects_index_dimension_mismatch(self) -> None:
        embedder = FastEmbedder(backend=FakeLocalBackend(), dimensions=768)
        with self.assertRaisesRegex(ValueError, "index requires 768"):
            asyncio.run(embedder.embed_query("query"))

    def test_generation_selection_uses_full_small_document_and_bounded_large_coverage(self) -> None:
        _, small = _chunks()
        full = select_generation_evidence(small, ())
        self.assertEqual(full.mode, "whole_document_coverage")
        self.assertEqual(full.chunks, small.chunks)

        seed = small.chunks[0]
        expanded_chunks = tuple(
            replace(seed, chunk_id=f"chunk-{index}", ordinal=index, text=f"section {index} " + "x" * 90)
            for index in range(20)
        )
        expanded = replace(small, chunks=expanded_chunks)
        ranked = tuple(
            HybridHit(index + 1, 1 / (61 + index), SearchHit(index + 1, 1.0, chunk), index + 1, index + 1)
            for index, chunk in enumerate(reversed(expanded_chunks[:4]))
        )
        selected = select_generation_evidence(expanded, ranked, max_chunks=6, max_characters=1000)
        self.assertEqual(selected.mode, "hybrid_relevance_plus_structural_coverage")
        self.assertLessEqual(len(selected.chunks), 6)
        self.assertIn("chunk-0", {chunk.chunk_id for chunk in selected.chunks})
        self.assertIn("chunk-19", {chunk.chunk_id for chunk in selected.chunks})

    def test_rrf_combines_complementary_rankings_without_score_comparison(self) -> None:
        _, chunk_set = _chunks()
        first, second = chunk_set.chunks
        lexical = (
            SearchHit(1, 12.4, first),
            SearchHit(2, 4.1, second),
        )
        dense = (
            SearchHit(1, 0.91, second),
            SearchHit(2, 0.84, first),
        )
        fused = reciprocal_rank_fusion(lexical, dense, top_k=2)
        self.assertEqual([hit.hit.chunk.chunk_id for hit in fused], [first.chunk_id, second.chunk_id])
        self.assertEqual((fused[0].lexical_rank, fused[0].dense_rank), (1, 2))
        self.assertAlmostEqual(fused[0].rrf_score, 1 / 61 + 1 / 62)

    def test_tokenizer_stems_keeps_decimals_whole_and_drops_punctuation(self) -> None:
        self.assertEqual(tokenize("F1=0.84 held-out"), ("f1", "0.84", "held", "out"))
        self.assertEqual(len(set(tokenize("replaces replaced replacing"))), 1)
        self.assertNotIn("28", tokenize("The model reached 28.4 BLEU."))

    def test_tokenizer_reads_spoken_and_written_forms_as_the_same_tokens(self) -> None:
        self.assertEqual(
            tokenize("an eighty point five percent score on the G L U E benchmark"),
            tokenize("an 80.5% score on the GLUE benchmark"),
        )
        self.assertEqual(
            tokenize("3 point 5 7 percent top five error"), tokenize("3.57% top 5 error")
        )

    def test_keyword_search_matches_a_written_claim_to_a_spoken_source(self) -> None:
        document = extract_summary_bytes(
            b"# Results\n\nThe large model achieved an eighty point five percent score "
            b"on the G L U E benchmark.\n\nTraining data came from two public corpora.\n",
            filename="spoken.md",
        )
        chunks = build_chunks(document)
        hits = LexicalIndex(chunks).search(
            "It scores 80.5% on GLUE.",
            document_id=document.document_id,
            document_version=document.document_version,
        )
        self.assertIn("eighty point five percent", hits[0].chunk.text)

    def test_lexical_ranks_exact_metric_and_enforces_scope(self) -> None:
        document, chunks = _chunks()
        index = LexicalIndex(chunks)
        hits = index.search(
            "What was the proposed F1 0.84?",
            document_id=document.document_id,
            document_version=document.document_version,
            top_k=2,
        )
        self.assertIn("Proposed F1 was 0.84", hits[0].chunk.text)
        with self.assertRaises(PermissionError):
            index.search(
                "F1",
                document_id="another-document",
                document_version=document.document_version,
            )

    def test_dense_cosine_ranking_and_dimension_checks(self) -> None:
        document, chunks = _chunks()
        vectors = [(1.0, 0.0), (0.0, 1.0)]
        index = DenseIndex(chunks, vectors)
        hits = index.search(
            (0.9, 0.1),
            document_id=document.document_id,
            document_version=document.document_version,
            top_k=1,
        )
        self.assertEqual(hits[0].chunk, chunks.chunks[0])
        self.assertAlmostEqual(cosine_similarity((1, 0), (1, 0)), 1.0)
        with self.assertRaisesRegex(ValueError, "dimension"):
            index.search(
                (1.0,),
                document_id=document.document_id,
                document_version=document.document_version,
            )

    def test_embedding_adapter_formats_document_and_query_instructions(self) -> None:
        transport = FakeEmbeddingTransport()
        embedder = GeminiEmbedding2(transport=transport, dimensions=768)
        documents = asyncio.run(embedder.embed_documents(("alpha", "beta")))
        query = asyncio.run(embedder.embed_query("find alpha"))

        self.assertEqual(len(documents.vectors), 2)
        self.assertEqual(documents.input_tokens, 10)
        self.assertEqual(query.input_tokens, 5)
        self.assertIn("task: retrieval | document: alpha", transport.calls[0][1])
        self.assertIn("task: retrieval | query: find alpha", transport.calls[-1][1])

    def test_embedding_adapter_rejects_wrong_provider_dimension(self) -> None:
        embedder = GeminiEmbedding2(
            transport=FakeEmbeddingTransport(bad_dimension=True), dimensions=768
        )
        with self.assertRaisesRegex(ValueError, "unexpected embedding dimension"):
            asyncio.run(embedder.embed_query("query"))

    def test_embedding_batch_has_an_overall_deadline(self) -> None:
        embedder = GeminiEmbedding2(
            transport=SlowEmbeddingTransport(), dimensions=768,
            batch_timeout_seconds=0.01,
        )
        with self.assertRaisesRegex(TimeoutError, "embedding batch exceeded"):
            asyncio.run(embedder.embed_documents(("one", "two", "three")))

    def test_concurrent_embedding_preserves_input_vector_order(self) -> None:
        class OrderedTransport:
            async def embed(self, *, text, dimensions, **kwargs):
                delay = {"first": 0.03, "second": 0.02, "third": 0.01}
                name = text.rsplit(": ", 1)[-1]
                await asyncio.sleep(delay[name])
                value = float(("first", "second", "third").index(name) + 1)
                return TransportEmbedding(tuple([value] * dimensions), 1)
        embedder = GeminiEmbedding2(
            transport=OrderedTransport(), dimensions=768, max_concurrency=3,
        )
        batch = asyncio.run(embedder.embed_documents(("first", "second", "third")))
        self.assertEqual([vector[0] for vector in batch.vectors], [1.0, 2.0, 3.0])

    def test_ranking_metrics_count_only_answer_bearing_passages(self) -> None:
        from eval.metrics import abstains, numeric_overlap, ranking_scores

        ranked = ["Sample size was 48.", "Proposed F1 was 0.84.", "Unrelated."]
        scores = ranking_scores(ranked, ["F1 was 0.84"], total_relevant=1, ks=(1, 3))
        self.assertEqual((scores["precision@1"], scores["recall@1"]), (0.0, 0.0))
        self.assertEqual(scores["recall@3"], 1.0)
        self.assertAlmostEqual(scores["precision@3"], 1 / 3)
        self.assertEqual(scores["reciprocal_rank"], 0.5)
        self.assertAlmostEqual(scores["ndcg@3"], 1 / 1.584962500721156)
        self.assertTrue(numeric_overlap("GLUE reached 80.5%.", "GLUE reached eighty point five percent."))
        self.assertFalse(numeric_overlap("It took 2 days.", "It took 3.5 days."))
        self.assertIsNone(numeric_overlap("It is faster.", "anything"))
        self.assertTrue(abstains("The deck does not say how many sentence pairs were used."))

    def test_threshold_calibration_uses_only_supplied_development_scores(self) -> None:
        development = (
            LabeledScore("positive", 0.8, True),
            LabeledScore("absent", 0.6, False),
        )
        policy = calibrate_zero_false_accept_threshold(
            development, calibration_id="dev-v1"
        )
        self.assertGreater(policy.threshold, 0.6)
        self.assertLess(policy.threshold, 0.8)
        heldout = (
            LabeledScore("positive-heldout", 0.7, True),
            LabeledScore("absent-heldout", 0.65, False),
        )
        metrics = evaluate_threshold(policy, heldout)
        self.assertEqual(metrics["positive_accept_rate"], 1.0)
        self.assertEqual(metrics["absent_false_accept_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
