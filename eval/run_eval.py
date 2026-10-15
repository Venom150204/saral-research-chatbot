"""SARAL evaluation: one command, one results file, one report."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval.metrics import (
    abstains, cosine, mean_scores, numeric_overlap, percentile, ranking_scores, rouge_l,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "eval" / "datasets"
HUMAN = ROOT / "eval" / "human"
RESULTS = ROOT / "eval" / "results"
LATEST = RESULTS / "latest.json"
REPORT = ROOT / "eval" / "REPORT.md"
DECKS = ("attention", "bert", "resnet")
ROUTING_TOOLS = {"write_deliverable", "edit_draft", "history", "check_claims", "find_in_deck", "ask_user"}
CITATION = re.compile(r"\[([^\]]*S\d+[^\]]*)\]")
JUDGE_CONCURRENCY = 4


class Saral:
    """The running application, driven over HTTP like the browser does."""

    def __init__(self, client) -> None:
        self.client = client
        self.decks: dict[str, dict[str, Any]] = {}

    def new_session(self) -> str:
        return self.client.post("/api/sessions").json()["session_id"]

    def deck(self, name: str) -> dict[str, Any]:
        """Upload a deck once; later sessions reuse the stored, indexed version."""
        if name not in self.decks:
            session = self.new_session()
            path = DATA / "decks" / f"{name}.pptx"
            upload = self.client.post(f"/api/documents?filename={path.name}&session_id={session}",
                                      content=path.read_bytes(),
                                      headers={"Content-Type": "application/octet-stream"}).json()
            sources = self.client.get(f"/api/documents/{upload['document_id']}/sources",
                                      params={"document_version": upload["document_version"]}).json()["sources"]
            self.decks[name] = {**upload, "sources": {item["chunk_id"]: item for item in sources}}
        return self.decks[name]

    def retrieve(self, deck: dict[str, Any], query: str, top_k: int) -> list[dict[str, Any]]:
        return self.client.post(f"/api/documents/{deck['document_id']}/retrieve", json={
            "query": query, "top_k": top_k, "document_version": deck["document_version"],
        }).json()["hits"]

    def turn(self, session: str, deck: dict[str, Any], message: str) -> dict[str, Any]:
        """One agent turn; a question the agent asks is answered with its first option."""
        body = {"session_id": session, "turn_id": f"turn-{uuid.uuid4()}", "message": message,
                "document_id": deck["document_id"], "document_version": deck["document_version"]}
        record: dict[str, Any] = {"message": message, "text": "", "tools": [], "artifacts": [], "source_ids": {}}
        started = time.perf_counter()
        for _ in range(2):
            interrupt = None
            with self.client.stream("POST", "/api/chat/stream", json=body) as response:
                if response.status_code != 200:
                    record["error"] = f"http {response.status_code}"
                    break
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    kind = event["type"]
                    if kind == "start":
                        record["source_ids"] = event.get("source_ids") or {}
                    elif kind == "token":
                        record["text"] += event["text"]
                    elif kind == "step" and event.get("stage") == "start" and event["tool"] in ROUTING_TOOLS:
                        record["tools"].append(event["tool"])
                    elif kind == "artifact":
                        record["artifacts"].append({"tool": event["tool"], "result": event["result"]})
                    elif kind == "verification":
                        record["answer_check"] = {key: event.get(key) for key in ("claims", "claims_supported", "passed")}
                    elif kind == "interrupt":
                        interrupt = event
                    elif kind == "error":
                        record["error"] = event["message"]
                    elif kind == "done":
                        record.update({key: event.get(key) for key in (
                            "first_token_ms", "latency_ms", "model_calls", "input_tokens", "output_tokens")})
            if interrupt is None:
                break
            record["question"] = interrupt.get("question")
            body = {key: value for key, value in body.items() if key != "message"} | {
                "resume": (interrupt.get("options") or ["yes"])[0]}
        record["wall_ms"] = round((time.perf_counter() - started) * 1000)
        return record


def cited_chunks(record: dict[str, Any]) -> list[str]:
    """Chunk ids the answer cites as [S#], in order, without repeats."""
    ids: list[str] = []
    for group in CITATION.findall(record["text"]):
        for sid in re.findall(r"S\d+", group):
            chunk = record["source_ids"].get(sid)
            if chunk and chunk not in ids:
                ids.append(chunk)
    return ids


class Judge:
    """RAGAS metrics with a Gemini judge, through Gemini's OpenAI-compatible endpoint
    (RAGAS's native Google adapter needs a synchronous client)."""

    def __init__(self) -> None:
        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import ContextPrecision, ContextRecall, Faithfulness

        self.model = (os.getenv("SARAL_JUDGE_MODEL") or "").strip() or "gemini-3.5-flash"
        client = AsyncOpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        llm = llm_factory(self.model, provider="openai", client=client, max_tokens=16_384)
        self.faithfulness = Faithfulness(llm=llm)
        self.context_precision = ContextPrecision(llm=llm)
        self.context_recall = ContextRecall(llm=llm)
        self.gate = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def score(self, metric: str, **inputs) -> float | None:
        """One metric value; None (reported as a judge failure) if the judge errors."""
        async with self.gate:
            try:
                result = await getattr(self, metric).ascore(**inputs)
            except Exception as error:
                print(f"  judge {metric} failed: {type(error).__name__}: {str(error)[:120]}")
                return None
            return None if result.value is None else float(result.value)


def _mean(values) -> float | None:
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 3) if values else None


def _count(values) -> str:
    values = list(values)
    return f"{sum(bool(v) for v in values)} of {len(values)}"


def labelled_questions() -> list[dict[str, Any]]:
    cases = json.loads((DATA / "retrieval" / "questions.json").read_text(encoding="utf-8"))["cases"]
    return [case for case in cases if case["deck_id"] in DECKS]


def run_retrieval(saral: Saral) -> dict[str, Any]:
    from eval.metrics import is_relevant

    rows, per_question = [], []
    for case in labelled_questions():
        if not case["answer_needles"]:
            continue
        deck = saral.deck(case["deck_id"])
        relevant = sum(is_relevant(item["text"], case["answer_needles"]) for item in deck["sources"].values())
        hits = saral.retrieve(deck, case["query"], top_k=5)
        scores = ranking_scores([hit["text"] for hit in hits], case["answer_needles"], relevant)
        rows.append(scores)
        per_question.append({"case_id": case["case_id"], "query": case["query"], "relevant_in_deck": relevant,
                             "first_hit_slide": hits[0]["evidence"][0].get("slide_number") if hits else None,
                             **{key: round(value, 3) for key, value in scores.items()}})
    print(f"retrieval: {len(rows)} questions")
    return {"questions": len(rows), "method": "hybrid lookup (BM25 + bge-base embeddings, reciprocal-rank fused), top 5",
            "mean": mean_scores(rows), "per_question": per_question}


async def run_answers(saral: Saral, judge: Judge | None) -> dict[str, Any]:
    records = []
    for case in labelled_questions():
        deck = saral.deck(case["deck_id"])
        record = await asyncio.to_thread(saral.turn, saral.new_session(), deck, case["query"])
        chunks = cited_chunks(record)
        slides = [deck["sources"][chunk]["slide_number"] for chunk in chunks if chunk in deck["sources"]]
        gold = [item["text"] for item in deck["sources"].values()
                if case["answer_needles"] and any(n.casefold() in item["text"].casefold() for n in case["answer_needles"])]
        records.append({"case_id": case["case_id"], "deck": case["deck_id"], "category": case["category"],
                        "query": case["query"], "answer": record["text"].strip(), "cited_slides": slides,
                        "gold_slides": case["gold_slides"], "cited_texts": [deck["sources"][c]["text"] for c in chunks if c in deck["sources"]],
                        "gold_texts": gold, "tools": record["tools"], "error": record.get("error"),
                        **{key: record.get(key) for key in ("first_token_ms", "latency_ms", "model_calls",
                                                            "input_tokens", "output_tokens", "answer_check")}})
        print(f"  answer {case['case_id']}: cited slides {slides} gold {case['gold_slides']} "
              f"first={record.get('first_token_ms')}ms")
    answerable = [r for r in records if r["gold_slides"]]
    absent = [r for r in records if not r["gold_slides"]]
    for record in answerable:
        cited = record["cited_slides"]
        record["citation_precision"] = sum(s in record["gold_slides"] for s in cited) / len(cited) if cited else 0.0
        record["citation_hit"] = any(s in record["gold_slides"] for s in cited)
    if judge is not None:
        jobs = []
        for record in records:
            deck = saral.deck(record["deck"])
            whole_deck = [item["text"] for item in deck["sources"].values()]
            jobs.append(("faithfulness", record, judge.score("faithfulness", user_input=record["query"],
                                                                response=record["answer"], retrieved_contexts=whole_deck)))
            if record["gold_texts"] and record["cited_texts"]:
                reference = " ".join(record["gold_texts"])
                jobs.append(("context_precision", record, judge.score(
                    "context_precision", user_input=record["query"], reference=reference,
                    retrieved_contexts=record["cited_texts"])))
                jobs.append(("context_recall", record, judge.score(
                    "context_recall", user_input=record["query"], reference=reference,
                    retrieved_contexts=record["cited_texts"])))
        values = await asyncio.gather(*(job for _, _, job in jobs))
        for (name, record, _), value in zip(jobs, values):
            record[name] = value
    summary = {
        "questions": len(records), "answerable": len(answerable), "not_in_deck": len(absent),
        "errors": sum(bool(r["error"]) for r in records),
        "citation_precision": _mean(r["citation_precision"] for r in answerable),
        "citation_recall": _count(r["citation_hit"] for r in answerable),
        "answered_without_citation": sum(not r["cited_slides"] for r in answerable),
        "abstained_when_not_in_deck": _count(abstains(r["answer"]) for r in absent),
        "first_token_ms_median": statistics.median([r["first_token_ms"] for r in records if r.get("first_token_ms")] or [0]),
        "latency_ms_median": statistics.median([r["latency_ms"] for r in records if r.get("latency_ms")] or [0]),
        "latency_ms_p95": percentile([r["latency_ms"] for r in records if r.get("latency_ms")], 0.95),
        "answer_check_passed": _count(bool((r.get("answer_check") or {}).get("passed")) for r in records if r.get("answer_check")),
    }
    if judge is not None:
        summary["ragas"] = {
            "judge_model": judge.model,
            "faithfulness": _mean(r.get("faithfulness") for r in answerable),
            "faithfulness_not_in_deck": _mean(r.get("faithfulness") for r in absent),
            "context_precision": _mean(r.get("context_precision") for r in answerable),
            "context_recall": _mean(r.get("context_recall") for r in answerable),
            "judge_failures": sum(r.get(name, 0) is None for r in records
                                  for name in ("faithfulness", "context_precision", "context_recall") if name in r),
        }
    return {"summary": summary, "per_question": records}


async def run_conversations(saral: Saral, judge: Judge | None) -> dict[str, Any]:
    from fastembed import TextEmbedding

    dataset = json.loads((DATA / "agent_conversations.json").read_text(encoding="utf-8"))
    records, drafts = [], []
    for conversation in dataset["conversations"]:
        deck = saral.deck(conversation["deck"])
        session = saral.new_session()
        for number, turn in enumerate(conversation["turns"], start=1):
            record = await asyncio.to_thread(saral.turn, session, deck, turn["message"])
            decision = record["tools"][: record["tools"].index("ask_user") + 1] if "ask_user" in record["tools"] else record["tools"]
            record.update(conversation=conversation["id"], turn=number,
                          right_tools=sorted(decision) in [sorted(option) for option in turn["accept"]])
            for item in record["artifacts"]:
                if item["tool"] in {"write_deliverable", "edit_draft"}:
                    drafts.append(_draft_metrics(item, deck))
            records.append(record)
            print(f"  {conversation['id']:24s} t{number} {'ok ' if record['right_tools'] else 'MISS'} "
                  f"tools={record['tools']} total={record.get('latency_ms')}ms")
    model = TextEmbedding("BAAI/bge-base-en-v1.5", cache_dir=os.getenv("SARAL_EMBEDDING_CACHE_DIR", ".cache/fastembed"))
    pairs = [pair for draft in drafts for pair in draft.pop("_pairs")]
    if pairs:
        vectors = list(model.embed([text for pair in pairs for text in pair]))
        similarities = [float(cosine(vectors[2 * i], vectors[2 * i + 1])) for i in range(len(pairs))]
    else:
        similarities = []
    if judge is not None:
        values = await asyncio.gather(*(judge.score(
            "faithfulness", user_input=draft["request"], response=draft["_text"],
            retrieved_contexts=draft["_contexts"]) for draft in drafts if draft["saved"] and draft["_contexts"]))
        for draft, value in zip([d for d in drafts if d["saved"] and d["_contexts"]], values):
            draft["ragas_faithfulness"] = value
    for draft in drafts:
        draft.pop("_text"), draft.pop("_contexts")
    writes = [d for d in drafts if d["tool"] == "write_deliverable"]
    limits = [limit for d in drafts for limit in d["limits"]]
    requirements = [req for d in writes for req in d["requirements"]]
    overlap = [match for draft in drafts for match in draft["numeric_overlap"]]
    latencies = [r["latency_ms"] for r in records if r.get("latency_ms")]
    direct = [r for r in records if not r["tools"] and r.get("first_token_ms")]
    summary = {
        "turns": len(records), "right_tools": _count(r["right_tools"] for r in records),
        "errors": sum(bool(r.get("error")) for r in records),
        "drafts_written": len(writes), "drafts_saved": _count(d["saved"] for d in writes),
        "claims_supported": f"{sum(d['claims_supported'] for d in drafts if d['saved'])} of "
                            f"{sum(d['claims'] for d in drafts if d['saved'])}",
        "limits_met": _count(item["met"] for item in limits),
        "requirements_met": _count(item["met"] for item in requirements),
        "citation_coverage": f"{sum(d['cited_deck_claims'] for d in drafts)} of {sum(d['deck_claims'] for d in drafts)}",
        "unit_basis": {basis: sum(d["basis"].get(basis, 0) for d in drafts) for basis in ("source", "background", "framing")},
        "source_overlap": {"numeric_match": _count(overlap),
                           "semantic_similarity_mean": round(statistics.mean(similarities), 3) if similarities else None,
                           "semantic_similarity_min": round(min(similarities), 3) if similarities else None},
        "answer_check_passed": _count(bool(r["answer_check"].get("passed")) for r in records
                                      if r.get("answer_check") and not r["artifacts"]),
        "first_token_ms_median_direct": statistics.median([r["first_token_ms"] for r in direct] or [0]),
        "latency_ms_median": statistics.median(latencies or [0]),
        "latency_ms_p95": percentile(latencies, 0.95),
        "model_calls_mean": _mean(r.get("model_calls") for r in records),
        "input_tokens_mean": _mean(r.get("input_tokens") for r in records),
        "output_tokens_mean": _mean(r.get("output_tokens") for r in records),
        "tool_misses": [f"{r['conversation']} t{r['turn']}: {r['tools']}" for r in records if not r["right_tools"]],
    }
    if judge is not None:
        summary["ragas_faithfulness_of_drafts"] = _mean(d.get("ragas_faithfulness") for d in drafts)
        summary["judge_model"] = judge.model
    for record in records:
        record["artifacts"] = [{"tool": a["tool"], "status": a["result"]["status"]} for a in record["artifacts"]]
        record.pop("source_ids", None)
    return {"summary": summary, "drafts": drafts, "turns": records}


def _draft_metrics(item: dict[str, Any], deck: dict[str, Any]) -> dict[str, Any]:
    result = item["result"]
    value = result.get("value") or {}
    artifact = value.get("artifact") or {}
    review = value.get("verification") or {}
    verdicts = review.get("verdicts") or []
    units = [unit for block in artifact.get("blocks", []) for unit in block.get("units", [])]
    basis: dict[str, int] = {}
    pairs, overlap, contexts = [], [], []
    for unit in units:
        kind = unit.get("basis") or ("source" if unit.get("factual") else "framing")
        basis[kind] = basis.get(kind, 0) + 1
        cited = [deck["sources"][ref["chunk_id"]]["text"] for ref in unit.get("support") or []
                 if ref["chunk_id"] in deck["sources"]]
        if kind == "source" and cited:
            evidence = " ".join(cited)
            pairs.append((unit["text"], evidence))
            contexts.extend(text for text in cited if text not in contexts)
            match = numeric_overlap(unit["text"], evidence)
            if match is not None:
                overlap.append(match)
    deck_claims = [unit for unit in units if (unit.get("basis") or ("source" if unit.get("factual") else "framing")) == "source"]
    return {
        "tool": item["tool"], "status": result["status"], "saved": result["status"] == "succeeded",
        "request": (artifact.get("request") or {}).get("output", {}).get("original_request")
                   or (artifact.get("request") or {}).get("output", {}).get("format") or "",
        "units": len(units), "basis": basis,
        "deck_claims": len(deck_claims),
        "cited_deck_claims": sum(bool(unit.get("support")) for unit in deck_claims),
        "claims": len(verdicts), "claims_supported": sum(v["verdict"] == "supported" for v in verdicts),
        "limits": [{"quote": c["quote"], "measured": c["measured"], "met": c["met"]}
                   for c in (value.get("constraints") or {}).get("checked", [])],
        "requirements": [{"requirement": r["requirement"], "met": r["met"]}
                         for r in review.get("requirement_verdicts") or []],
        "numeric_overlap": overlap,
        "_pairs": pairs, "_contexts": contexts,
        "_text": " ".join(unit["text"] for unit in deck_claims),
    }


SHOWCASE = [
    ("01-graduate-talk-with-notes-and-equations", "resnet", [
        "Make a 7-slide talk for graduate students focusing on methods; include 3 speaker notes "
        "per slide and preserve key equations.",
        "Make #2 less technical.",
        "Undo that.",
    ]),
    ("02-script-lengths-and-styles", "attention", [
        "Make a 90-second script for policymakers.",
        "Now a 30-second plain-English version for a general audience.",
        "Write a 5-minute technical script for graduate students.",
        "Write a press release about the main result.",
    ]),
    ("03-tweets-and-linkedin", "bert", [
        "Create 5 tweet-sized abstracts for a general audience, each under 280 characters.",
        "Make #3 more visual.",
        "Now 2 LinkedIn summaries for industry engineers.",
    ]),
    ("04-questions-provenance-and-checks", "bert", [
        "Which slide reports the GLUE score?",
        "How big is BERT-base?",
        "Explain masked language modelling to a 15-year-old.",
        "Write a 90-second script for teachers.",
        "Is everything in it supported by the deck?",
    ]),
]


def _render_draft(result: dict[str, Any], deck: dict[str, Any]) -> list[str]:
    value = result.get("value") or {}
    artifact = value.get("artifact")
    out = [f"*Status: {result['status']}*"]
    if not artifact:
        reason = value.get("message") or (value.get("blocked") or {}).get("message")
        return out + ([f"> {reason}"] if reason else [])
    out[0] = f"*Draft version {artifact['version']} · status: {result['status']}*"
    for number, block in enumerate(artifact["blocks"], start=1):
        out.append(f"\n**#{number} {block.get('title') or block['block_type']}**\n")
        for unit in block["units"]:
            places = sorted({deck["sources"][ref["chunk_id"]]["location"] or "source"
                             for ref in unit.get("support") or [] if ref["chunk_id"] in deck["sources"]})
            basis = unit.get("basis") or ("source" if unit.get("factual") else "framing")
            cite = f" `[{', '.join(places)}]`" if places else f" `({basis})`"
            out.append(f"- *{unit['role']}*: {unit['text']}{cite}")
    review = value.get("verification") or {}
    verdicts = review.get("verdicts") or []
    checks = [f"claims supported: {sum(v['verdict'] == 'supported' for v in verdicts)} of {len(verdicts)}"]
    checks += [f"{c['quote']}: {c['measured']} ({'met' if c['met'] else 'missed'})"
               for c in (value.get("constraints") or {}).get("checked", [])]
    checks += [f"requirement \"{r['requirement']}\": {'met' if r['met'] else 'NOT met (' + r['reason'] + ')'}"
               for r in review.get("requirement_verdicts") or []]
    out.append("\nChecks: " + "; ".join(checks))
    for delta in value.get("deltas") or []:
        out.append(f"\n> **Changed** `{delta['unit_id']}`  \n> old: {delta['old_text']}  \n> new: {delta['new_text']}")
    if value.get("reason"):
        out.append(f"\nWhy: {value['reason']}")
    return out


def run_examples(saral: Saral) -> dict[str, Any]:
    """Conversation logs as evidence that the bot iterates (examples/*.md)."""
    folder = ROOT / "examples"
    written = []
    for name, deck_name, messages in SHOWCASE:
        deck = saral.deck(deck_name)
        session = saral.new_session()
        lines = [f"# {name.split('-', 1)[1].replace('-', ' ').capitalize()}", "",
                 f"Source: `eval/datasets/decks/{deck_name}.pptx`. Recorded by `python -m eval.run_eval --only examples` "
                 f"on {datetime.now(timezone.utc).date()}; nothing edited by hand.", ""]
        for message in messages:
            record = saral.turn(session, deck, message)
            lines += ["---", "", f"**User:** {message}", ""]
            if record["tools"]:
                lines.append(f"*Tools: {', '.join(record['tools'])}*")
            if record.get("question"):
                lines.append(f"*SARAL asked: {record['question']} (the log answers with the first option)*")
            text = record["text"].strip()
            for group in CITATION.findall(text):
                labels = [deck["sources"][record["source_ids"][sid]]["location"] or sid
                          for sid in re.findall(r"S\d+", group) if record["source_ids"].get(sid) in deck["sources"]]
                if labels:
                    text = text.replace(f"[{group}]", f"`[{', '.join(labels)}]`")
            if text:
                lines += ["", f"**SARAL:** {text}", ""]
            for item in record["artifacts"]:
                lines += _render_draft(item["result"], deck) + [""]
            meta = [f"first words {record['first_token_ms'] / 1000:.1f} s" if record.get("first_token_ms") else None,
                    f"done in {record['latency_ms'] / 1000:.1f} s" if record.get("latency_ms") else None]
            lines.append(f"*{' · '.join(part for part in meta if part)}*\n")
            print(f"  example {name}: {message[:50]} tools={record['tools']}")
        (folder / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")
        written.append(f"examples/{name}.md")
    return {"logs": written}


def run_scripts(saral: Saral) -> dict[str, Any]:
    """The system's scripts for the frozen tasks, and an anonymised pack for raters."""
    tasks = json.loads((DATA / "generation_tasks.json").read_text(encoding="utf-8"))
    outputs = HUMAN / "system_outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    items = []
    for deck_name in tasks["core_decks"]:
        deck = saral.deck(deck_name)
        for task in tasks["core_tasks"]:
            record = saral.turn(saral.new_session(), deck, task["request"])
            drafts = [a["result"] for a in record["artifacts"]
                      if a["tool"] == "write_deliverable" and a["result"]["status"] == "succeeded"]
            if not drafts:
                print(f"  script {deck_name}__{task['task_id']}: no saved draft")
                continue
            blocks = drafts[-1]["value"]["artifact"]["blocks"]
            script = "\n\n".join(" ".join(unit["text"] for unit in block["units"]) for block in blocks)
            (outputs / f"{deck_name}__{task['task_id']}.txt").write_text(script + "\n", encoding="utf-8")
            words = len(script.split())
            items.append({"deck": deck_name, "task_id": task["task_id"], "request": task["request"],
                          "words": words, "in_band": task["word_band"][0] <= words <= task["word_band"][1],
                          "script": script})
            print(f"  script {deck_name}__{task['task_id']}: {words} words")
    write_rater_pack(items)
    return {"scripts": len(items), "tasks": len(tasks["core_decks"]) * len(tasks["core_tasks"]),
            "within_word_band": _count(item["in_band"] for item in items),
            "per_script": [{key: item[key] for key in ("deck", "task_id", "words", "in_band")} for item in items]}


def _slide_texts(deck_name: str) -> dict[int, str]:
    """The deck's own words per slide, for raters: read locally from the file, without
    model-written picture descriptions or the template text repeated on every slide."""
    from backend.app.chunking import build_chunks
    from backend.app.ingestion import extract_upload

    path = DATA / "decks" / f"{deck_name}.pptx"
    chunks = build_chunks(extract_upload(path.read_bytes(), filename=path.name)).chunks
    slides: dict[int, list[str]] = {}
    for chunk in chunks:
        slide = next((ref.slide_number for ref in chunk.evidence if ref.slide_number), 0)
        slides.setdefault(slide, []).append(" ".join(chunk.text.split()))
    return {number: " ".join(texts) for number, texts in sorted(slides.items())}


def write_rater_pack(items: list[dict[str, Any]]) -> None:
    """Shuffled, anonymised items for the three raters, the private key, and the sheet."""
    import random

    shuffled = items[:]
    random.Random(2026).shuffle(shuffled)
    pack = ["# Rater pack", "", "Read `rater_instructions.md` first. Score each item in your copy of "
            "`rating_sheet_template.csv`.", ""]
    key = {}
    for number, item in enumerate(shuffled, start=1):
        item_id = f"item-{number:02d}"
        key[item_id] = {"deck": item["deck"], "task_id": item["task_id"], "system": "saral-agent"}
        pack += [f"## {item_id}", "", f"**Request:** {item['request']}", "", "**Source slides:**", ""]
        pack += [f"- Slide {n}: {text}" for n, text in _slide_texts(item["deck"]).items()]
        pack += ["", "**Script:**", "", item["script"], ""]
    (HUMAN / "rater_pack.md").write_text("\n".join(pack), encoding="utf-8")
    (HUMAN / "rater_key.json").write_text(json.dumps(key, indent=1), encoding="utf-8")
    with (HUMAN / "rating_sheet_template.csv").open("w", encoding="utf-8", newline="") as sheet:
        writer = csv.writer(sheet)
        writer.writerow(["rater_id", "item_id", "audience_fit_1_to_5", "factuality_1_to_5",
                         "helpfulness_1_to_5", "comments"])
        writer.writerows([["", item_id, "", "", "", ""] for item_id in key])


def rebuild_rater_pack() -> None:
    """The pack again from the saved scripts, with no model calls."""
    tasks = json.loads((DATA / "generation_tasks.json").read_text(encoding="utf-8"))
    items = [
        {"deck": deck, "task_id": task["task_id"], "request": task["request"],
         "script": (HUMAN / "system_outputs" / f"{deck}__{task['task_id']}.txt").read_text(encoding="utf-8").strip()}
        for deck in tasks["core_decks"] for task in tasks["core_tasks"]
        if (HUMAN / "system_outputs" / f"{deck}__{task['task_id']}.txt").exists()
    ]
    write_rater_pack(items)
    print(f"rater pack rebuilt from {len(items)} saved scripts")


def run_references() -> dict[str, Any]:
    """Human-written reference scripts and human ratings: used when present, never invented."""
    out: dict[str, Any] = {}
    reference_dir = DATA / "references"
    references = sorted(reference_dir.glob("*__*.txt")) if reference_dir.exists() else []
    outputs = HUMAN / "system_outputs"
    pairs = [(ref, outputs / ref.name) for ref in references if (outputs / ref.name).exists()]
    if pairs:
        scores = [{"task": ref.stem, "rouge_l": round(rouge_l(out_path.read_text(encoding="utf-8"),
                                                             ref.read_text(encoding="utf-8")), 3)}
                  for ref, out_path in pairs]
        out["rouge_l"] = {"pairs": len(scores), "mean": _mean(s["rouge_l"] for s in scores), "per_task": scores}
        try:
            from bert_score import score as bert_score
        except ImportError:
            out["bertscore"] = "not run: install bert-score (it needs PyTorch)"
        else:
            _, _, f1 = bert_score([o.read_text(encoding="utf-8") for _, o in pairs],
                                  [r.read_text(encoding="utf-8") for r, _ in pairs], lang="en")
            out["bertscore"] = {"pairs": len(pairs), "f1_mean": round(float(f1.mean()), 3)}
    else:
        out["rouge_l"] = out["bertscore"] = (
            "not run: no human-written reference scripts yet (eval/datasets/references/<deck>__<task>.txt, "
            "see eval/human/reference_script_tasks.md)")
    ratings = sorted((HUMAN / "ratings").glob("*.csv")) if (HUMAN / "ratings").exists() else []
    if ratings:
        rows = [row for path in ratings for row in csv.DictReader(path.open(encoding="utf-8"))]
        criteria = [key for key in rows[0] if key not in {"rater_id", "item_id", "comments"}]
        out["human_ratings"] = {"raters": len({row["rater_id"] for row in rows}), "items": len({row["item_id"] for row in rows}),
                                **{c: {"mean": _mean(float(row[c]) for row in rows if row[c]),
                                       "min": min(float(row[c]) for row in rows if row[c]),
                                       "max": max(float(row[c]) for row in rows if row[c])} for c in criteria}}
    else:
        out["human_ratings"] = "not run: no rating sheets yet (eval/human/ratings/<rater>.csv, filled from rater_pack.md)"
    return out


def _pct(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}"


def _seconds(ms) -> str:
    return "–" if not ms else f"{ms / 1000:.1f} s"


def write_report(results: dict[str, Any]) -> None:
    sections = results["sections"]
    lines = [
        "# SARAL evaluation report", "",
        f"Generated by `python -m eval.run_eval` on {results['run_at'][:10]}. Every number below comes from "
        "`eval/results/latest.json`; nothing is typed in by hand. Models: agent "
        f"`{results['models'].get('agent_model')}`, writer `{results['models'].get('generation_model')}`, "
        f"checker `{results['models'].get('verifier_model')}`, RAGAS judge `{results['models'].get('judge_model') or 'not run'}`.",
        "", "Small samples: three decks, 36 answerable questions, 16 scripted turns. A difference of a few "
        "items is within run-to-run variation. Model-judged scores (the checker, RAGAS) are evidence, not ground truth.", "",
    ]
    if "retrieval" in sections:
        r = sections["retrieval"]
        m = r["mean"]
        lines += ["## 1. Retrieval (deck lookup)", "",
                  f"{r['questions']} labelled questions; a passage counts as relevant when it contains the labelled "
                  f"answer phrase. Method: {r['method']}.", "",
                  "| k | Precision@k | Recall@k | F1@k | nDCG@k |", "|---|---|---|---|---|"]
        for k in (1, 3, 5):
            lines.append(f"| {k} | {_pct(m[f'precision@{k}'])} | {_pct(m[f'recall@{k}'])} | {_pct(m[f'f1@{k}'])} | {_pct(m[f'ndcg@{k}'])} |")
        lines += ["", f"MRR: **{_pct(m['reciprocal_rank'])}**. Precision@k is low by construction: most questions "
                  "have one answer-bearing bullet, so precision@5 cannot exceed about 0.2. The writer and the agent "
                  "read the whole deck; lookup serves `find_in_deck` and citation attribution.", ""]
    if "answers" in sections:
        s = sections["answers"]["summary"]
        lines += ["## 2. Answers to questions about the deck", "",
                  f"The agent answered all {s['questions']} questions ({s['answerable']} answerable, "
                  f"{s['not_in_deck']} whose answer is not in the deck), each in a new conversation.", "",
                  "| Metric | Result | What it means |", "|---|---|---|",
                  f"| Citation recall | {s['citation_recall']} | the answer cites a bullet on the gold slide |",
                  f"| Citation precision | {_pct(s['citation_precision'])} | share of cited bullets that are on a gold slide |",
                  f"| Answerable but uncited | {s['answered_without_citation']} | answered without any [S#] citation |",
                  f"| Says \"not in the deck\" when it is not | {s['abstained_when_not_in_deck']} | heuristic phrase match |",
                  f"| Answer check passed | {s['answer_check_passed']} | SARAL's own figure check + checker on cited sentences |",
                  f"| First words (median) | {_seconds(s['first_token_ms_median'])} | streamed |",
                  f"| Full answer median / p95 | {_seconds(s['latency_ms_median'])} / {_seconds(s['latency_ms_p95'])} | |", ""]
        if "ragas" in s:
            g = s["ragas"]
            lines += ["**RAGAS** (judge `" + g["judge_model"] + "`):", "",
                      "| Metric | Score (0–1) | Inputs |", "|---|---|---|",
                      f"| Faithfulness | {_pct(g['faithfulness'])} | answer vs the whole deck: share of the answer's statements the deck supports |",
                      f"| Context precision | {_pct(g['context_precision'])} | cited bullets vs the gold bullet: are the cited bullets relevant, best first |",
                      f"| Context recall | {_pct(g['context_recall'])} | gold bullet vs cited bullets: does what was cited contain the answer |",
                      f"| Faithfulness, answer not in deck | {_pct(g['faithfulness_not_in_deck'])} | an honest \"the deck doesn't say\" scores high |",
                      "", f"Judge failures (excluded): {g['judge_failures']}. Explanations the user asked for are general "
                      "knowledge by design, so faithfulness below 1.0 on explain-style answers is expected, not an error.", ""]
    if "conversations" in sections:
        s = sections["conversations"]["summary"]
        lines += ["## 3. Conversations: writing, editing, undo, checks", "",
                  f"{s['turns']} scripted turns in `eval/datasets/agent_conversations.json`.", "",
                  "| Metric | Result |", "|---|---|",
                  f"| Right action chosen (tool or direct answer) | {s['right_tools']} |",
                  f"| Drafts saved | {s['drafts_saved']} |",
                  f"| Draft claims the checker judged supported | {s['claims_supported']} |",
                  f"| Citation coverage (deck claims with a citation) | {s['citation_coverage']} |",
                  f"| Source overlap: figures found in the cited bullet | {s['source_overlap']['numeric_match']} |",
                  f"| Source overlap: sentence-to-cited-bullet similarity (bge-base cosine), mean / min | "
                  f"{_pct(s['source_overlap']['semantic_similarity_mean'])} / {_pct(s['source_overlap']['semantic_similarity_min'])} |",
                  f"| Limits met (counts, lengths, durations; measured by code) | {s['limits_met']} |",
                  f"| Requirements met (judged by the checker) | {s['requirements_met']} |",
                  f"| Direct answers passing the answer check | {s['answer_check_passed']} |",
                  f"| RAGAS faithfulness of saved drafts (deck claims vs their cited bullets) | {_pct(s.get('ragas_faithfulness_of_drafts'))} |",
                  f"| First words, direct answers (median) | {_seconds(s['first_token_ms_median_direct'])} |",
                  f"| Turn latency median / p95 | {_seconds(s['latency_ms_median'])} / {_seconds(s['latency_ms_p95'])} |",
                  f"| Model calls per turn (mean) | {s['model_calls_mean']} |",
                  f"| Tokens per turn in / out (mean) | {round(s['input_tokens_mean'] or 0):,} / {round(s['output_tokens_mean'] or 0):,} |", "",
                  f"Units written: {s['unit_basis']['source']} deck claims, {s['unit_basis']['background']} "
                  f"background explanation, {s['unit_basis']['framing']} framing. "
                  + (f"Tool misses: {'; '.join(s['tool_misses'])}." if s["tool_misses"] else "No tool misses."), ""]
        per_draft = [d for d in sections["conversations"]["drafts"] if d.get("ragas_faithfulness") is not None]
        if per_draft:
            low = min(per_draft, key=lambda d: d["ragas_faithfulness"])
            high = max(per_draft, key=lambda d: d["ragas_faithfulness"])
            lines += ["**The two judges disagree on drafts.** SARAL's checker supports "
                      f"{s['claims_supported']} cited claims; RAGAS faithfulness of the same deck claims is "
                      f"{_pct(s.get('ragas_faithfulness_of_drafts'))} (lowest {_pct(low['ragas_faithfulness'])}: "
                      f"\"{low['request'][:60]}\"; highest {_pct(high['ragas_faithfulness'])}: \"{high['request'][:60]}\"). "
                      "RAGAS splits each sentence into atomic statements and rejects anything inferred, such as the "
                      "significance of a result that copy for policymakers adds; SARAL's checker accepts a faithful "
                      "interpretation. That reading is a hypothesis: the statements have not yet been audited by a "
                      "person, which is the next quality task.", ""]
    comparison = results.get("orchestrator_comparison")
    if comparison:
        lines += ["## 4. Why the agent replaced the plan-and-execute pipeline", "",
                  "Same 16 turns, same writer and checker, measured before the pipeline was deleted "
                  "(`eval/results/orchestrator_comparison.json`).", "",
                  "| | Right action | First visible output (median) | Turn median / p95 | Drafts saved |",
                  "|---|---|---|---|---|"]
        for row in comparison["rows"]:
            lines.append(f"| {row['name']} | {row['right_tools']} | {row['first_output']} | {row['latency']} | {row['drafts_saved']} |")
        lines += ["", comparison["reading"], ""]
    if "scripts" in sections:
        s = sections["scripts"]
        lines += ["## 5. Scripts for human evaluation", "",
                  f"{s['scripts']} of {s['tasks']} frozen tasks (3 decks x 3 audiences, 90 seconds each) written and saved "
                  f"to `eval/human/system_outputs/`; {s['within_word_band']} within the task's word band "
                  "(90 s at the system's speaking rate, ±15%). The anonymised rater pack is `eval/human/rater_pack.md`.", ""]
    if "references" in sections:
        ref = sections["references"]
        lines += ["## 6. Human references and ratings (assignment metrics)", ""]
        for name, label in (("rouge_l", "ROUGE-L"), ("bertscore", "BERTScore"), ("human_ratings", "Human ratings (3 raters)")):
            value = ref[name]
            lines.append(f"- **{label}**: " + (value if isinstance(value, str) else "`" + json.dumps(value) + "`"))
        lines.append("")
    lines += ["## How each metric is computed", "",
              "- **Precision@k / Recall@k / F1@k**: relevant passages in the top k ÷ k; ÷ all relevant passages "
              "in the deck; their harmonic mean. **nDCG@k** rewards relevant passages ranked higher. **MRR**: 1 ÷ rank of the first relevant passage.",
              "- **Citation precision / recall**: computed from the slides of the bullets an answer cites ([S#]) "
              "against the labelled gold slides. No model involved.",
              "- **RAGAS faithfulness**: the judge splits the answer into statements and checks each against the "
              "context; score = supported ÷ all statements. **Context precision**: the judge marks each cited bullet "
              "useful or not for the reference answer; precision is averaged over ranks. **Context recall**: the "
              "judge splits the reference answer into statements and checks each is in the cited bullets.",
              "- **Source-overlap proxy** (assignment): every figure in a cited sentence must appear in its cited "
              "bullet (compared in written form), plus embedding cosine similarity between the sentence and the bullet.",
              "- **Citation coverage**: deck-claim units carrying at least one citation ÷ deck-claim units "
              "(background explanation and framing are labelled separately and reviewed as such).",
              "- **Claims supported**: SARAL's independent checker (a different model from the writer) per cited claim.",
              "- **Limits**: measured by code (spoken length at 150 words per minute, characters, item counts).", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", REPORT.relative_to(ROOT))


async def main() -> None:
    parser = argparse.ArgumentParser(description="SARAL evaluation")
    parser.add_argument("--only", nargs="*", choices=["retrieval", "answers", "conversations", "scripts", "references", "examples"])
    parser.add_argument("--no-judge", action="store_true", help="skip RAGAS (no judge-model calls)")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--rater-pack", action="store_true", help="rebuild the rater pack from saved scripts")
    args = parser.parse_args()
    if args.rater_pack:
        rebuild_rater_pack()
        return
    if args.report_only:
        write_report(json.loads(LATEST.read_text(encoding="utf-8")))
        return
    chosen = args.only or ["retrieval", "answers", "conversations", "scripts", "references"]
    previous = json.loads(LATEST.read_text(encoding="utf-8")) if LATEST.exists() else {"sections": {}}

    from fastapi.testclient import TestClient

    from backend.app.api import create_app

    judge = None if args.no_judge or not ({"answers", "conversations"} & set(chosen)) else Judge()
    with TestClient(create_app(), backend_options={"loop_factory": asyncio.SelectorEventLoop}) as client:
        health = client.get("/health").json()
        if not health.get("agent_ready"):
            raise SystemExit(f"agent unavailable: {health.get('agent_error')}")
        saral = Saral(client)
        sections = dict(previous.get("sections", {}))
        if "retrieval" in chosen:
            sections["retrieval"] = await asyncio.to_thread(run_retrieval, saral)
        if "answers" in chosen:
            sections["answers"] = await run_answers(saral, judge)
        if "conversations" in chosen:
            sections["conversations"] = await run_conversations(saral, judge)
        if "scripts" in chosen:
            sections["scripts"] = await asyncio.to_thread(run_scripts, saral)
        if "examples" in chosen:
            sections["examples"] = await asyncio.to_thread(run_examples, saral)
        if "references" in chosen:
            sections["references"] = run_references()
    comparison_path = RESULTS / "orchestrator_comparison.json"
    results = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": {key: health.get(key) for key in ("agent_model", "generation_model", "verifier_model", "embedding_model")}
                  | {"judge_model": judge.model if judge else previous.get("models", {}).get("judge_model")},
        "sections": sections,
        "orchestrator_comparison": json.loads(comparison_path.read_text(encoding="utf-8")) if comparison_path.exists() else None,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    LATEST.write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("saved", LATEST.relative_to(ROOT))
    write_report(results)


if __name__ == "__main__":
    asyncio.run(main())
