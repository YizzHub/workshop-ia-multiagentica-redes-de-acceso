"""
llm_planner.py
==============
Servicio "LLM" del Orquestador (Mensaje 2 de la Fig. 3: ``POST /plan``).

    prompt ──► ChromaDB (workflow más similar) ──► Mistral 7B @ Ollama
           (extrae SOLO los argumentos) ──► validación determinista ──► plan JSON

Si Ollama no está disponible se usa ``RuleBasedExtractor`` (LLM emulado) con
exactamente la misma interfaz, de modo que el notebook funciona en cualquier PC.
"""
import json
import re
from typing import Any

import requests

import workshop_common as wc

try:
    import ollama as ollama_lib
except ImportError:  # pragma: no cover
    ollama_lib = None

# El prompt de sistema vive en workshop_common para que el precalentamiento use los mismos tokens
EXTRACTION_SYSTEM_PROMPT = wc.EXTRACTION_SYSTEM_PROMPT
PARAM_MEANINGS = wc.PARAM_MEANINGS


def _normalize_number(value: Any) -> int | float:
    if isinstance(value, str):
        m = re.search(r"(\d+(?:\.\d+)?)\s*([tgmk])?", value.lower())
        if not m:
            raise ValueError(f"Valor numérico inválido: {value!r}")
        scale = {"t": 1000, "g": 1, "m": 1e-3, "k": 1e-6, None: 1}[m.group(2)]
        value = float(m.group(1)) * scale
    value = float(value)
    return int(value) if value.is_integer() else value


class RuleBasedExtractor:
    """'LLM emulado': extracción por expresiones regulares (determinista)."""

    NODE = r"([A-Za-z]+\d+)"
    PATTERNS = {
        "src_node": [rf"\bfrom\s+(?:node\s+)?{NODE}", rf"\bbetween\s+(?:node\s+)?{NODE}"],
        "dst_node": [rf"\bfrom\s+(?:node\s+)?[A-Za-z]+\d+\s+to\s+(?:node\s+)?{NODE}",
                     rf"\bbetween\s+(?:node\s+)?[A-Za-z]+\d+\s+and\s+(?:node\s+)?{NODE}"],
        "bitrate_gbps": [r"(\d+(?:\.\d+)?\s*[tgm])(?:b|bit|bits|bps|b/s)?\b"],
        "leaf_node": [r"\b((?:ONU|LEAF|L)\d+)\b"],
        "num_sc": [r"(\d+)\s*(?:digital\s+)?(?:sub-?carriers?|SCs?)\b"],
    }

    def extract(self, prompt: str, keys: list[str]) -> dict:
        out = {}
        for key in keys:
            for pattern in self.PATTERNS.get(key, []):
                m = re.search(pattern, prompt, flags=re.IGNORECASE)
                if m:
                    out[key] = m.group(1)
                    break
        return out


class LocalLLM:
    """Cliente del LLM on-premises (Mistral 7B servido por Ollama)."""

    def __init__(self, host: str = "http://127.0.0.1:11434", model: str = "mistral:7b",
                 enabled: Any = "auto"):
        self.host, self.requested_model = host.rstrip("/"), model
        self.model = model
        self.available = False
        if enabled is False:
            self.reason = "deshabilitado por configuración (USE_OLLAMA=False)"
            return
        try:
            tags = requests.get(f"{self.host}/api/tags", timeout=3).json().get("models", [])
        except Exception as exc:
            self.reason = f"Ollama no responde en {self.host} ({type(exc).__name__})"
            return
        names = [m["name"] for m in tags]
        base = model.split(":")[0]
        if model in names:
            self.model = model
        else:
            # Acepta otras etiquetas del mismo modelo, p.ej. mistral:latest (= Mistral 7B)
            alt = [n for n in names if n.split(":")[0] == base]
            if not alt:
                self.reason = f"modelo '{model}' no descargado (ejecute: ollama pull {model})"
                return
            self.model = alt[0]
        self.available = True
        self.reason = f"Ollama OK - modelo {self.model}"

    def chat_json(self, system: str, user: str, timeout: float = 600.0) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        options = dict(wc.LLM_OPTIONS)
        if ollama_lib is not None:
            client = ollama_lib.Client(host=self.host, timeout=timeout)
            resp = client.chat(model=self.model, messages=messages, format="json", options=options,
                               keep_alive=wc.LLM_KEEP_ALIVE)
            content = resp["message"]["content"]
        else:
            r = requests.post(f"{self.host}/api/chat", timeout=timeout,
                              json={"model": self.model, "messages": messages, "format": "json",
                                    "stream": False, "options": options, "keep_alive": wc.LLM_KEEP_ALIVE})
            r.raise_for_status()
            content = r.json()["message"]["content"]
        return json.loads(content)


class Planner:
    """Convierte un prompt en lenguaje natural en el plan estructurado (Fig. 5)."""

    def __init__(self, kb, llm: LocalLLM, known_nodes: set[str] | None = None, min_similarity: float = 0.15):
        self.kb, self.llm = kb, llm
        self.rules = RuleBasedExtractor()
        self.known_nodes = set(known_nodes or [])
        self.min_similarity = min_similarity
        self.last_retrieval: list[dict] = []
        self.last_backend = ""
        self.last_raw: Any = None

    def _validate(self, raw: dict, keys: list[str], defaults: dict) -> dict:
        params = {}
        for key in keys:
            value = raw.get(key)
            if value is None or (isinstance(value, str) and value.strip().lower() in ("", "null", "none")):
                value = defaults.get(key)   # el prompt no lo menciona -> valor por defecto del workflow
            if value is None or value == "":
                raise ValueError(f"Falta el parámetro '{key}'")
            if key.endswith("_node"):
                value = str(value).strip().upper().replace(" ", "")
                if self.known_nodes and value not in self.known_nodes:
                    raise ValueError(f"Nodo desconocido '{value}'")
            elif key == "num_sc":
                value = int(_normalize_number(value))
            elif key.endswith("_gbps"):
                value = _normalize_number(value)
                if value <= 0:
                    if key not in defaults:
                        raise ValueError(f"'{key}' debe ser > 0")
                    value = defaults[key]
            params[key] = value
        return params

    def extract(self, prompt: str, keys: list[str], defaults: dict) -> dict:
        if not keys:
            self.last_backend = "sin argumentos"
            return {}
        if self.llm.available:
            system = wc.extraction_system_prompt(keys)
            try:
                self.last_raw = self.llm.chat_json(system, prompt)
                params = self._validate(self.last_raw, keys, defaults)
                self.last_backend = f"ollama:{self.llm.model}"
                return params
            except Exception as exc:  # guardarraíl: salida del LLM inválida -> respaldo
                print(f"[Planner] Salida del LLM descartada ({exc}); usando extractor determinista")
        self.last_raw = self.rules.extract(prompt, keys)
        self.last_backend = "LLM emulado (reglas)"
        return self._validate(self.last_raw, keys, defaults)

    def plan(self, prompt: str) -> dict:
        self.last_retrieval = self.kb.search(prompt, k=3)
        best = self.last_retrieval[0]
        if best["similarity"] < self.min_similarity:
            raise ValueError(f"Ningún workflow soportado coincide con el prompt (sim={best['similarity']})")
        keys: list[str] = []
        for step in best["steps"]:
            keys += [a for a in step["args"] if a not in keys]
        params = self.extract(prompt, keys, best.get("defaults", {}))
        return {
            "steps": [{"intent": s["intent"]} for s in best["steps"]],
            "plans": [{"tool": s["tool"], "args": {a: params[a] for a in s["args"]}, "source": "llm"}
                      for s in best["steps"]],
        }


def create_llm_app(planner: Planner):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class PromptRequest(BaseModel):
        prompt: str

    app = FastAPI(title="LLM planner (ChromaDB + Ollama)")

    @app.post("/plan")
    def plan(req: PromptRequest):
        try:
            return planner.plan(req.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/health")
    def health():
        return {"llm": planner.llm.reason, "kb": planner.kb.backend,
                "embedder": planner.kb.embedder.name}

    return app
