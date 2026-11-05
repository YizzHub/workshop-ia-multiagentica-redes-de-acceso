"""
knowledge_base.py
=================
Base de conocimiento vectorial de *workflows* (diseño Zero-Shot del paper).

  1. Un archivo de configuración (workflows.json) define los workflows
     soportados: ``name``, ``description`` y una secuencia ordenada de ``steps``.
  2. Cada workflow se convierte en *embedding* y se almacena en ChromaDB.
  3. El prompt del operador se convierte en embedding y se recupera el workflow
     más similar (similitud coseno). El prompt NO lleva ejemplos (zero-shot).

Backends de embeddings (``EMBEDDING_BACKEND``):
  * "ollama"         -> modelo de embeddings local en Ollama (p.ej. nomic-embed-text)
  * "chroma-default" -> all-MiniLM-L6-v2 (ONNX) de ChromaDB (descarga ~80 MB)
  * "hashing"        -> embedding léxico determinista, 100 % offline (respaldo)
  * "auto"           -> ollama si el modelo está disponible; si no, hashing
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

import requests

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:  # pragma: no cover
    CHROMADB_AVAILABLE = False

STOPWORDS = {"a", "an", "the", "of", "to", "from", "with", "and", "or", "for", "on", "in", "into",
             "between", "by", "please", "new", "existing", "network", "nodes", "node", "any",
             "without", "anything", "two", "is", "are", "be", "that", "this", "it", "me", "i"}

CANONICAL = {
    "establish": "establish", "create": "establish", "provision": "establish", "setup": "establish",
    "set": "establish", "bring": "establish", "connect": "establish", "open": "establish",
    "remove": "remove", "delete": "remove", "tear": "remove", "teardown": "remove",
    "release": "remove", "terminate": "remove", "destroy": "remove",
    "route": "route", "path": "route", "routing": "route", "shortest": "route", "find": "route",
    "status": "status", "state": "status", "show": "status", "list": "status", "check": "status",
    "connection": "connection", "connectivity": "connection", "circuit": "connection", "service": "connection",
    "bandwidth": "bandwidth", "bitrate": "bandwidth", "capacity": "bandwidth", "gb": "bandwidth",
    "gbps": "bandwidth", "mbps": "bandwidth", "1gb": "bandwidth",
}


def _tokens(text: str) -> list[str]:
    out = []
    for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9]*", text.lower()):
        if re.fullmatch(r"[a-z]\d+", tok) or tok in STOPWORDS:
            continue  # descarta identificadores de nodos (h1, r2...) y palabras vacías
        if tok.endswith("ing") and len(tok) > 5:
            tok = tok[:-3]
        elif tok.endswith("s") and len(tok) > 4:
            tok = tok[:-1]
        out.append(CANONICAL.get(tok, tok))
    return out


class HashingEmbedder:
    """Embedding léxico-semántico determinista (bolsa de palabras normalizadas + trigramas)."""

    name = "hashing-bow-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = int(hashlib.md5(feature.encode()).hexdigest(), 16)
        return h % self.dim, (1.0 if (h >> 64) & 1 else -1.0)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in _tokens(text):
                idx, sign = self._bucket("w:" + tok)
                vec[idx] += sign
                for i in range(len(tok) - 2):
                    idx, sign = self._bucket("c:" + tok[i:i + 3])
                    vec[idx] += 0.2 * sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host, self.model = host.rstrip("/"), model
        self.name = f"ollama:{model}"

    def __call__(self, texts: list[str]) -> list[list[float]]:
        r = requests.post(f"{self.host}/api/embed", json={"model": self.model, "input": texts}, timeout=120)
        r.raise_for_status()
        return r.json()["embeddings"]


class ChromaDefaultEmbedder:
    name = "chroma-default:all-MiniLM-L6-v2"

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        self._ef = DefaultEmbeddingFunction()

    def __call__(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._ef(texts)]


def _ollama_has_model(host: str, model: str) -> bool:
    try:
        tags = requests.get(f"{host.rstrip('/')}/api/tags", timeout=2).json().get("models", [])
    except Exception:
        return False
    base = model.split(":")[0]
    return any(m["name"] == model or m["name"].split(":")[0] == base for m in tags)


def make_embedder(backend: str = "auto", ollama_host: str = "http://127.0.0.1:11434",
                  ollama_embed_model: str = "nomic-embed-text"):
    if backend == "ollama" or (backend == "auto" and _ollama_has_model(ollama_host, ollama_embed_model)):
        return OllamaEmbedder(ollama_host, ollama_embed_model)
    if backend == "chroma-default":
        return ChromaDefaultEmbedder()
    return HashingEmbedder()


class _InMemoryCollection:
    """Respaldo mínimo con la misma interfaz usada de ChromaDB (si no está instalado)."""

    def __init__(self):
        self.items: dict[str, tuple[list[float], str, dict]] = {}

    def upsert(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.items[i] = (e, d, m)

    def count(self) -> int:
        return len(self.items)

    def query(self, query_embeddings, n_results):
        q = query_embeddings[0]
        scored = sorted(((1 - sum(a * b for a, b in zip(q, e)), i, d, m)
                         for i, (e, d, m) in self.items.items()))[:n_results]
        return {"ids": [[s[1] for s in scored]], "distances": [[s[0] for s in scored]],
                "documents": [[s[2] for s in scored]], "metadatas": [[s[3] for s in scored]]}


class WorkflowKnowledgeBase:
    def __init__(self, workflows: list[dict], embedder=None, collection_name: str = "telecom_workflows"):
        self.embedder = embedder or HashingEmbedder()
        if CHROMADB_AVAILABLE:
            self.client = chromadb.EphemeralClient()
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass
            self.collection = self.client.create_collection(
                collection_name, embedding_function=None, metadata={"hnsw:space": "cosine"})
            self.backend = f"ChromaDB {chromadb.__version__}"
        else:
            self.collection = _InMemoryCollection()
            self.backend = "in-memory (ChromaDB no instalado)"
        for wf in workflows:
            self.add_workflow(wf)

    @classmethod
    def from_file(cls, path: str = "workflows.json", embedder=None) -> "WorkflowKnowledgeBase":
        with open(path, encoding="utf-8") as fh:
            return cls(json.load(fh)["workflows"], embedder)

    @staticmethod
    def document(wf: dict) -> str:
        intents = "; ".join(s["intent"] for s in wf["steps"])
        return f"{wf['name'].replace('_', ' ')}: {wf['description']} Steps: {intents}"

    def add_workflow(self, wf: dict) -> None:
        doc = self.document(wf)
        self.collection.upsert(
            ids=[wf["name"]], embeddings=self.embedder([doc]), documents=[doc],
            metadatas=[{"name": wf["name"], "description": wf["description"],
                        "steps": json.dumps(wf["steps"]), "defaults": json.dumps(wf.get("defaults", {}))}])

    def search(self, prompt: str, k: int = 3) -> list[dict[str, Any]]:
        k = min(k, self.collection.count())
        res = self.collection.query(query_embeddings=self.embedder([prompt]), n_results=k)
        hits = []
        for dist, meta in zip(res["distances"][0], res["metadatas"][0]):
            hits.append({"name": meta["name"], "description": meta["description"],
                         "steps": json.loads(meta["steps"]), "defaults": json.loads(meta["defaults"]),
                         "similarity": round(1.0 - float(dist), 4)})
        return hits
