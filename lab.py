"""All embedder logic, metrics, and the sample-file benchmark."""

from __future__ import annotations

import hashlib
import math
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from dotenv import load_dotenv
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

SAMPLE_PATH = ROOT / "sample.txt"
TOKEN_RE = re.compile(r"[a-z0-9]+")
_FASTEMBED: dict[str, object] = {}
_ST_MODELS: dict[str, object] = {}
_CATALOG: dict[str, "Embedder"] | None = None

QUERIES = [
    ("q1", "How do plants turn sunlight into sugar?", ["photosynthesis"]),
    ("q2", "What Roman name was used for the city that became Paris?", ["paris-history"]),
    ("q3", "Which language uses indentation instead of braces for code blocks?", ["python-programming"]),
    ("q4", "What is the event horizon of a collapsed star?", ["black-holes"]),
    ("q5", "Diet built around olive oil, fish, and vegetables", ["mediterranean-cooking"]),
    ("q6", "How does RAG find similar document chunks?", ["vector-databases"]),
    ("q7", "When did the modern Olympics start in Athens?", ["olympic-games"]),
    ("q8", "Why do corals turn white when the ocean gets too warm?", ["coral-reefs"]),
    ("q9", "libraries used for machine learning in a popular scripting language", ["python-programming"]),
    ("q10", "HNSW indexes for nearest neighbor search", ["vector-databases"]),
]

PAIRS = [
    ("Plants convert light into chemical energy using chlorophyll.", "Photosynthesis stores sunlight as sugars in the Calvin cycle.", 0.92),
    ("Paris was rebuilt with wide boulevards in the 19th century.", "Haussmann transformed the French capital with grand avenues.", 0.88),
    ("Python is popular for data science and machine learning.", "NumPy and PyTorch are common libraries in that language.", 0.8),
    ("Nothing can escape from inside a black hole's event horizon.", "Light cannot leave the region beyond that gravitational boundary.", 0.9),
    ("Vector databases retrieve the nearest embedding vectors.", "RAG systems embed a query and fetch similar document chunks.", 0.84),
    ("Coral bleaching happens when sea temperatures rise.", "The first modern Olympic Games were held in Athens in 1896.", 0.05),
    ("Mediterranean meals often include olive oil and grilled fish.", "A black hole sits at the center of the Milky Way.", 0.04),
    ("The Eiffel Tower was finished for a world's fair in 1889.", "Python uses indentation to define code blocks.", 0.06),
    ("Reefs are built by animals that secrete calcium carbonate.", "Coral colonies live with symbiotic algae in warm shallow water.", 0.86),
    ("Athletes compete in swimming, gymnastics, and athletics.", "Winter Games include skiing, skating, and ice hockey.", 0.72),
]

METRIC_GUIDE = [
    {"name": "Cosine similarity", "role": "Core comparison metric", "summary": "Angle between two vectors. Almost every embedding search ranking uses cosine."},
    {"name": "Recall@k", "role": "Retrieval quality", "summary": "Did the correct document chunk appear in the top-k results?"},
    {"name": "MRR", "role": "Retrieval quality", "summary": "1/rank of the first correct chunk. Rank 1 = 1.0, rank 2 = 0.5."},
    {"name": "nDCG@k", "role": "Ranking quality", "summary": "Correct items count more when they appear higher in the list."},
    {"name": "STS Spearman / Pearson", "role": "Semantic similarity", "summary": "Correlation between model cosine scores and labeled sentence-pair relatedness."},
    {"name": "Anisotropy", "role": "Geometry health", "summary": "Average cosine between different chunks. Very high values mean collapsed vectors."},
    {"name": "Latency / dimensions", "role": "Engineering", "summary": "Speed and vector size. Not quality scores by themselves."},
]

KEY_FIELDS = [
    {"name": "OPENAI_API_KEY", "label": "OpenAI", "unlocks": "text-embedding-3-small and 3-large", "docs": "https://platform.openai.com/api-keys"},
    {"name": "COHERE_API_KEY", "label": "Cohere", "unlocks": "embed-english-v3.0", "docs": "https://dashboard.cohere.com/api-keys"},
    {"name": "GOOGLE_API_KEY", "label": "Google Gemini", "unlocks": "text-embedding-004", "docs": "https://aistudio.google.com/apikey"},
    {"name": "VOYAGE_API_KEY", "label": "Voyage AI", "unlocks": "voyage-3", "docs": "https://dash.voyageai.com/"},
    {"name": "HF_TOKEN", "label": "Hugging Face", "unlocks": "Inference API feature-extraction", "docs": "https://huggingface.co/settings/tokens"},
    {"name": "MISTRAL_API_KEY", "label": "Mistral", "unlocks": "mistral-embed", "docs": "https://console.mistral.ai/api-keys"},
    {"name": "OLLAMA_BASE_URL", "label": "Ollama URL", "unlocks": "nomic-embed-text on a local Ollama server", "docs": "https://ollama.com/"},
]


def env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Chunk:
    id: str
    title: str
    text: str


@dataclass
class QueryCase:
    id: str
    query: str
    relevant_ids: list[str]


@dataclass
class SimilarityPair:
    text_a: str
    text_b: str
    score: float


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    current_id = ""
    current_lines: list[str] = []

    def flush() -> None:
        if current_id:
            chunks.append(
                Chunk(current_id, current_id.replace("-", " ").title(), "\n".join(current_lines).strip())
            )

    for line in SAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            current_id = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return chunks


def load_dataset() -> dict:
    chunks = load_chunks()
    return {
        "file": "sample.txt",
        "summary": (
            "A short homemade article with eight unrelated topics. Each heading is one chunk, "
            "so a good embedder should retrieve the matching section for each labeled question."
        ),
        "how_it_is_used": (
            "The benchmark embeds all chunks plus 10 labeled questions, ranks by cosine, "
            "and scores Recall@k / MRR / nDCG. Sentence pairs are used for STS correlation."
        ),
        "chunks": chunks,
        "queries": [QueryCase(qid, query, rel) for qid, query, rel in QUERIES],
        "pairs": [SimilarityPair(a, b, score) for a, b, score in PAIRS],
    }


class Embedder(ABC):
    id: str
    name: str
    provider: str
    model: str
    kind: str = "api"
    dimensions: int | None = None
    notes: str = ""
    explanation: str = ""
    env_var: str | None = None

    @abstractmethod
    def is_configured(self) -> bool:
        raise NotImplementedError

    def missing_reason(self) -> str | None:
        if self.is_configured():
            return None
        return f"Set {self.env_var} in .env" if self.env_var else "Missing dependency"

    @abstractmethod
    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        ok = self.is_configured()
        return {
            "id": self.id,
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "kind": self.kind,
            "dimensions": self.dimensions,
            "configured": ok,
            "missing_reason": None if ok else self.missing_reason(),
            "notes": self.notes,
            "explanation": self.explanation or self.notes,
            "env_var": self.env_var,
        }


class HashBaseline(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "hash-baseline", "Hash bag-of-words", "local-baseline"
        self.model, self.kind, self.dimensions = "char-ngrams-256d", "local", 256
        self.env_var = None
        self.explanation = (
            "A hashed bag-of-words control, not a neural embedder. It only notices overlapping words. "
            "Use it as the floor real models should beat."
        )

    def is_configured(self) -> bool:
        return True

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        matrix = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = TOKEN_RE.findall(text.lower())
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for gram in grams:
                digest = hashlib.md5(gram.encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % 256
                matrix[i, idx] += 1.0 if digest[4] % 2 == 0 else -1.0
        return matrix


class FastEmbedLocal(Embedder):
    def __init__(self, model: str, embedder_id: str, name: str, explanation: str) -> None:
        self.id, self.name, self.provider, self.model = embedder_id, name, "fastembed", model
        self.kind, self.dimensions, self.env_var = "local", None, None
        self.explanation = explanation
        self._q = "search_query: " if "nomic-embed-text" in model else ""
        self._d = "search_document: " if "nomic-embed-text" in model else ""

    def is_configured(self) -> bool:
        try:
            import fastembed  # noqa: F401
        except ImportError:
            return False
        return True

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        from fastembed import TextEmbedding

        if self.model not in _FASTEMBED:
            _FASTEMBED[self.model] = TextEmbedding(model_name=self.model)
        prefix = self._q if input_type == "query" else self._d
        matrix = np.asarray(list(_FASTEMBED[self.model].embed([prefix + t for t in texts])), dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class SentenceTransformersLocal(Embedder):
    def __init__(self) -> None:
        self.id, self.name = "mpnet", "MPNet base v2"
        self.provider, self.model = "sentence-transformers", "sentence-transformers/all-mpnet-base-v2"
        self.kind, self.dimensions, self.env_var = "local", None, None
        self.explanation = "Optional local 768-d model. Needs `pip install sentence-transformers` (torch)."

    def is_configured(self) -> bool:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def missing_reason(self) -> str | None:
        return None if self.is_configured() else "Needs the sentence-transformers package (and torch)."

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        from sentence_transformers import SentenceTransformer

        if self.model not in _ST_MODELS:
            _ST_MODELS[self.model] = SentenceTransformer(self.model)
        matrix = np.asarray(_ST_MODELS[self.model].encode(texts, normalize_embeddings=True), dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class OpenAIEmbedder(Embedder):
    def __init__(self, model: str, embedder_id: str, name: str, explanation: str) -> None:
        self.id, self.name, self.provider, self.model = embedder_id, name, "openai", model
        self.kind, self.env_var = "api", "OPENAI_API_KEY"
        self.dimensions = 3072 if "large" in model else 1536
        self.explanation = explanation

    def is_configured(self) -> bool:
        return bool(env("OPENAI_API_KEY"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        from openai import OpenAI

        response = OpenAI(api_key=env("OPENAI_API_KEY")).embeddings.create(model=self.model, input=texts)
        matrix = np.asarray([item.embedding for item in sorted(response.data, key=lambda row: row.index)], dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class CohereEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "cohere-v3", "Cohere embed-english-v3", "cohere"
        self.model, self.kind, self.dimensions = env("COHERE_EMBEDDING_MODEL", "embed-english-v3.0"), "api", 1024
        self.env_var = "COHERE_API_KEY"
        self.explanation = "Cohere v3 (1024-d). Retrieval-oriented cloud model. Uses query vs document input types."

    def is_configured(self) -> bool:
        return bool(env("COHERE_API_KEY"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        import cohere

        response = cohere.ClientV2(api_key=env("COHERE_API_KEY")).embed(
            texts=texts,
            model=self.model,
            input_type="search_query" if input_type == "query" else "search_document",
            embedding_types=["float"],
        )
        vectors = getattr(response.embeddings, "float_", None) or getattr(response.embeddings, "float", None)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class GeminiEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "gemini-embedding", "Google Gemini embedding", "google"
        self.model = env("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.kind, self.dimensions, self.env_var = "api", 768, "GOOGLE_API_KEY"
        self.explanation = "Google Gemini embeddings. Needs GOOGLE_API_KEY from AI Studio."

    def is_configured(self) -> bool:
        return bool(env("GOOGLE_API_KEY"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        from google import genai
        from google.genai import types

        task = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
        response = genai.Client(api_key=env("GOOGLE_API_KEY")).models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        matrix = np.asarray([item.values for item in response.embeddings], dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class VoyageEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "voyage-3", "Voyage AI voyage-3", "voyage"
        self.model = env("VOYAGE_EMBEDDING_MODEL", "voyage-3")
        self.kind, self.dimensions, self.env_var = "api", 1024, "VOYAGE_API_KEY"
        self.explanation = "Voyage voyage-3. Retrieval-focused cloud embeddings."

    def is_configured(self) -> bool:
        return bool(env("VOYAGE_API_KEY"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        import voyageai

        response = voyageai.Client(api_key=env("VOYAGE_API_KEY")).embed(
            texts, model=self.model, input_type="query" if input_type == "query" else "document"
        )
        matrix = np.asarray(response.embeddings, dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class HuggingFaceEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "huggingface-api", "Hugging Face Inference", "huggingface"
        self.model = env("HF_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self.kind, self.dimensions, self.env_var = "api", None, "HF_TOKEN"
        self.explanation = "Runs a Hub embedding model through Hugging Face Inference. Needs HF_TOKEN."

    def is_configured(self) -> bool:
        return bool(env("HF_TOKEN"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        from huggingface_hub import InferenceClient

        try:
            client = InferenceClient(api_key=env("HF_TOKEN"))
        except TypeError:
            client = InferenceClient(token=env("HF_TOKEN"))
        vectors = []
        for text in texts:
            array = np.asarray(client.feature_extraction(text, model=self.model), dtype=np.float32)
            vectors.append(array.mean(axis=0) if array.ndim > 1 else array)
        matrix = np.vstack(vectors)
        self.dimensions = int(matrix.shape[1])
        return matrix


class MistralEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "mistral-embed", "Mistral embed", "mistral"
        self.model = env("MISTRAL_EMBEDDING_MODEL", "mistral-embed")
        self.kind, self.dimensions, self.env_var = "api", 1024, "MISTRAL_API_KEY"
        self.explanation = "Mistral cloud embeddings. Needs MISTRAL_API_KEY."

    def is_configured(self) -> bool:
        return bool(env("MISTRAL_API_KEY"))

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        response = httpx.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={"Authorization": f"Bearer {env('MISTRAL_API_KEY')}"},
            json={"model": self.model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        ordered = sorted(response.json()["data"], key=lambda item: item["index"])
        matrix = np.asarray([item["embedding"] for item in ordered], dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


class OllamaEmbedder(Embedder):
    def __init__(self) -> None:
        self.id, self.name, self.provider = "ollama", "Ollama local server", "ollama"
        self.model = env("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        self.kind, self.dimensions, self.env_var = "local", None, "OLLAMA_BASE_URL"
        self.explanation = "Talks to a local Ollama daemon. Pull nomic-embed-text first."

    def _base(self) -> str:
        return env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    def is_configured(self) -> bool:
        try:
            return httpx.get(f"{self._base()}/api/tags", timeout=1.5).status_code == 200
        except httpx.HTTPError:
            return False

    def missing_reason(self) -> str | None:
        return None if self.is_configured() else f"Ollama not reachable at {self._base()}"

    def embed(self, texts: list[str], *, input_type: str = "document") -> np.ndarray:
        del input_type
        payload = httpx.post(f"{self._base()}/api/embed", json={"model": self.model, "input": texts}, timeout=60.0)
        payload.raise_for_status()
        vectors = payload.json().get("embeddings") or []
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimensions = int(matrix.shape[1])
        return matrix


def build_embedders() -> list[Embedder]:
    return [
        HashBaseline(),
        FastEmbedLocal("sentence-transformers/all-MiniLM-L6-v2", "minilm", "MiniLM-L6-v2", "Small fast 384-d local model. No API key. Good first test."),
        FastEmbedLocal("BAAI/bge-small-en-v1.5", "bge-small", "BGE small English", "Local 384-d retrieval model from BAAI. Often beats MiniLM on RAG questions."),
        FastEmbedLocal("nomic-ai/nomic-embed-text-v1.5", "nomic", "Nomic embed text v1.5", "Local 768-d retrieval model. Adds search_query / search_document prefixes."),
        FastEmbedLocal("jinaai/jina-embeddings-v2-small-en", "jina-small", "Jina v2 small English", "Small local 512-d Jina model. No prefixes needed."),
        SentenceTransformersLocal(),
        OpenAIEmbedder(env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), "openai-small", "OpenAI 3-small", "OpenAI 3-small (1536-d). Strong cloud default. Needs OPENAI_API_KEY."),
        OpenAIEmbedder(env("OPENAI_EMBEDDING_MODEL_LARGE", "text-embedding-3-large"), "openai-large", "OpenAI 3-large", "OpenAI 3-large (3072-d). Usually stronger and more expensive."),
        CohereEmbedder(),
        GeminiEmbedder(),
        VoyageEmbedder(),
        HuggingFaceEmbedder(),
        MistralEmbedder(),
        OllamaEmbedder(),
    ]


def reset_embedders() -> None:
    global _CATALOG
    _CATALOG = None


def install_sentence_transformers() -> tuple[bool, str]:
    torch = subprocess.run(
        [sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"],
        capture_output=True,
        text=True,
    )
    st_pkg = subprocess.run(
        [sys.executable, "-m", "pip", "install", "sentence-transformers"],
        capture_output=True,
        text=True,
    )
    log = (torch.stdout or "") + (torch.stderr or "") + "\n" + (st_pkg.stdout or "") + (st_pkg.stderr or "")
    ok = torch.returncode == 0 and st_pkg.returncode == 0
    if ok:
        reset_embedders()
    return ok, log[-2500:]


def all_embedders() -> list[Embedder]:
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = {item.id: item for item in build_embedders()}
    return list(_CATALOG.values())


def get_embedder(embedder_id: str) -> Embedder:
    all_embedders()
    assert _CATALOG is not None
    return _CATALOG[embedder_id]


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    return vectors / norms


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return l2_normalize(a) @ l2_normalize(b).T


def pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=1)


def recall_at_k(ranked_ids: list[list[str]], relevant_ids: list[list[str]], k: int) -> float:
    hits = total = 0
    for ranked, relevant in zip(ranked_ids, relevant_ids, strict=True):
        rel = set(relevant)
        total += len(rel)
        hits += len(rel.intersection(ranked[:k]))
    return hits / total if total else 0.0


def ndcg_at_k(ranked_ids: list[list[str]], relevant_ids: list[list[str]], k: int) -> float:
    values = []
    for ranked, relevant in zip(ranked_ids, relevant_ids, strict=True):
        rel = set(relevant)
        dcg = sum(1.0 / math.log2(i + 1) for i, chunk_id in enumerate(ranked[:k], start=1) if chunk_id in rel)
        ideal = min(len(rel), k)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
        values.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(values)) if values else 0.0


def evaluate_embedder(embedder: Embedder, chunks: list[Chunk], queries: list[QueryCase], pairs: list[SimilarityPair]) -> dict:
    chunk_ids = [chunk.id for chunk in chunks]
    started = time.perf_counter()
    chunk_vectors = embedder.embed([chunk.text for chunk in chunks], input_type="document")
    chunk_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    query_vectors = embedder.embed([item.query for item in queries], input_type="query")
    query_ms = (time.perf_counter() - started) * 1000
    similarities = cosine_similarity_matrix(query_vectors, chunk_vectors)
    ranked_ids, per_query, ranks = [], [], []
    for i, query in enumerate(queries):
        order = np.argsort(-similarities[i])
        ranked = [chunk_ids[j] for j in order]
        ranked_ids.append(ranked)
        rank = next((n for n, chunk_id in enumerate(ranked, start=1) if chunk_id in query.relevant_ids), None)
        ranks.append(rank)
        per_query.append(
            {
                "id": query.id,
                "query": query.query,
                "relevant_ids": query.relevant_ids,
                "rank": rank,
                "hit_at_1": rank == 1,
                "top": [{"chunk_id": chunk_ids[j], "score": float(similarities[i, j])} for j in order[:5]],
            }
        )
    relevant = [query.relevant_ids for query in queries]
    predicted = pairwise_cosine(
        embedder.embed([pair.text_a for pair in pairs]),
        embedder.embed([pair.text_b for pair in pairs]),
    )
    gold = np.asarray([pair.score for pair in pairs], dtype=np.float32)
    spearman = pearson = None
    if len(predicted) >= 3:
        spearman_v = spearmanr(predicted, gold).statistic
        pearson_v = pearsonr(predicted, gold).statistic
        spearman = None if np.isnan(spearman_v) else float(spearman_v)
        pearson = None if np.isnan(pearson_v) else float(pearson_v)
    sims = cosine_similarity_matrix(chunk_vectors, chunk_vectors)
    n = sims.shape[0]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mrr_scores = [1.0 / rank for rank in ranks if rank is not None]
    return {
        "embedder": embedder.info(),
        "ok": True,
        "error": None,
        "dimensions": int(chunk_vectors.shape[1]),
        "latency_ms_per_chunk": round(chunk_ms / max(len(chunks), 1), 2),
        "latency_ms_per_query": round(query_ms / max(len(queries), 1), 2),
        "recall_at_1": round(recall_at_k(ranked_ids, relevant, 1), 4),
        "recall_at_3": round(recall_at_k(ranked_ids, relevant, 3), 4),
        "recall_at_5": round(recall_at_k(ranked_ids, relevant, 5), 4),
        "mrr": round(float(np.mean(mrr_scores)) if mrr_scores else 0.0, 4),
        "ndcg_at_5": round(ndcg_at_k(ranked_ids, relevant, 5), 4),
        "sts_spearman": None if spearman is None else round(spearman, 4),
        "sts_pearson": None if pearson is None else round(pearson, 4),
        "anisotropy": round(float(sims[mask].mean()) if n > 1 else 0.0, 4),
        "queries": per_query,
    }


def search_chunks(embedder: Embedder, chunks: list[Chunk], query: str, k: int = 5) -> list[dict]:
    scores = cosine_similarity_matrix(
        embedder.embed([query], input_type="query"),
        embedder.embed([chunk.text for chunk in chunks], input_type="document"),
    )[0]
    return [
        {"chunk_id": chunks[i].id, "title": chunks[i].title, "text": chunks[i].text, "score": round(float(scores[i]), 4)}
        for i in np.argsort(-scores)[:k]
    ]


def compare_texts(embedder: Embedder, text_a: str, text_b: str) -> dict:
    vectors = embedder.embed([text_a, text_b])
    return {
        "embedder": embedder.info(),
        "cosine": round(float(pairwise_cosine(vectors[0:1], vectors[1:2])[0]), 4),
        "dimensions": int(vectors.shape[1]),
        "preview_a": [round(float(x), 5) for x in vectors[0][:12]],
        "preview_b": [round(float(x), 5) for x in vectors[1][:12]],
    }


def _mask(value: str) -> str:
    if not value:
        return ""
    return "••••" if len(value) <= 8 else f"{value[:3]}…{value[-4:]}"


def public_key_status() -> list[dict]:
    rows = []
    for field in KEY_FIELDS:
        current = env(field["name"], "http://127.0.0.1:11434" if field["name"] == "OLLAMA_BASE_URL" else "")
        if field["name"] == "OLLAMA_BASE_URL":
            current = env("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        rows.append({**field, "set": bool(env(field["name"])) if field["name"] != "OLLAMA_BASE_URL" else bool(current), "masked": _mask(current)})
    return rows


def persist_and_apply(updates: dict[str, str]) -> list[str]:
    cleaned = {key: value.strip() for key, value in updates.items() if value and value.strip()}
    if not cleaned:
        return []
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            out.append(line)
            continue
        name = line.partition("=")[0].strip()
        if name in cleaned:
            out.append(f"{name}={cleaned[name]}")
            seen.add(name)
        else:
            out.append(line)
    for name, value in cleaned.items():
        os.environ[name] = value
        if name not in seen:
            out.append(f"{name}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return list(cleaned)


if __name__ == "__main__":
    import json
    import sys

    wanted = sys.argv[1:]
    if not wanted:
        for item in all_embedders():
            print(f"{item.id:18} {'ready' if item.is_configured() else 'need-key':8} {item.name}")
        raise SystemExit(0)
    data = load_dataset()
    for embedder_id in wanted:
        embedder = get_embedder(embedder_id)
        if not embedder.is_configured():
            print(f"{embedder_id}: {embedder.missing_reason()}")
            continue
        result = evaluate_embedder(embedder, data["chunks"], data["queries"], data["pairs"])
        print(json.dumps({k: result[k] for k in result if k != "queries"}, indent=2))
