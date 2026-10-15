# External resources

## Papers

| Paper | Used for |
|---|---|
| Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020 | retrieval + generation design |
| Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond*, 2009 | lexical retrieval |
| Cormack et al., *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods*, SIGIR 2009 | hybrid ranking |
| Vaswani et al. 2017 (Attention), Devlin et al. 2019 (BERT), He et al. 2016 (ResNet) | evaluation decks in `eval/datasets/decks` |

## Models and services

| Resource | Purpose | Terms |
|---|---|---|
| Google Gemini API (`gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`, `gemini-3.5-flash`) | agent, writer, checker, evaluation judge | Google AI terms |
| `BAAI/bge-base-en-v1.5` via fastembed | local embeddings | MIT |
| PostgreSQL with pgvector; Neon for the hosted demo | storage and vector search | PostgreSQL License; Neon terms |
| GitHub Codespaces, Google Cloud Run | demo hosting | provider terms |

## Python libraries

| Library | Purpose | License |
|---|---|---|
| FastAPI, Uvicorn | API and streaming | MIT, BSD-3 |
| LangChain, LangGraph, langchain-google-genai, langgraph-checkpoint-postgres | agent loop, middleware, checkpoints | MIT |
| google-genai | Gemini client | Apache-2.0 |
| fastembed | embeddings runtime | Apache-2.0 |
| psycopg | PostgreSQL driver | LGPL-3.0 |
| python-pptx, pypdf | PPTX and PDF parsing | MIT, BSD-3 |
| py-rust-stemmers, text2num | BM25 stemming, number normalisation | MIT |
| httpx, python-dotenv | HTTP client, configuration | BSD-3 |
| RAGAS, rouge-score, langchain-community (evaluation only) | evaluation metrics | Apache-2.0 |

## Interface

| Library | Purpose | License |
|---|---|---|
| React, Vite, TypeScript | interface and build | MIT, MIT, Apache-2.0 |
| Tailwind CSS, tw-animate-css | styling | MIT |
| Motion | animation | MIT |
| lucide-react | icons | ISC |
| clsx, tailwind-merge, class-variance-authority | class utilities | MIT |
| Magic UI and motion-primitives components (`web/src/components/ui`) | border beam, shimmer and text effects, adapted | MIT |

## Brand

The SARAL name, logo, colours and typeface belong to SARAL / Democratise Research
(saral.democratiseresearch.in) and are used because this module is built for SARAL.
