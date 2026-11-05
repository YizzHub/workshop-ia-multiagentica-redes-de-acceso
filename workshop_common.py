"""
workshop_common.py
==================
Utilidades compartidas por TODOS los componentes del workshop:

  * Configuración global persistida en ``workshop_config.json``.
  * ``TrafficCapture``: captura de tráfico "estilo Wireshark" (Figura 3 del paper).
  * ``http_call``: cliente HTTP instrumentado que registra cada request/response.
  * ``start_server`` / ``stop_server``: lanzan apps FastAPI con uvicorn en hilos
    *daemon*, de modo que ninguna celda bloquee el kernel de Jupyter.
"""
from __future__ import annotations

import copy
import json
import os
import socket
import threading
import time
from typing import Any, Callable

import requests

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workshop_config.json")

DEFAULT_CONFIG: dict[str, Any] = {
    "SIMULATION_MODE": True,          # True: Mininet/Ryu emulados | False: Mininet + Ryu reales
    "HOST": "127.0.0.1",
    "PORTS": {
        "orchestrator": 8000,         # Orquestador (POST /prompt)
        "llm": 8001,                  # Servicio LLM + ChromaDB (POST /plan)
        "agent": 8002,                # IPConnectionAgent (A2A, POST /execute_plan)
        "pce": 8003,                  # Path Computation Element (POST /compute_route)
        "optical_agent": 8004,        # OpticalP2MPAgent (Ejercicio 2)
        "mcp_http": 8005,             # Servidor MCP standalone (opcional, transporte HTTP)
        "sdn": 8080,                  # NBI REST del controlador SDN (Ryu wsapi o emulador)
        "mininet_ctl": 9000,          # Canal de control del script Mininet (modo real)
    },
    "USE_OLLAMA": "auto",             # "auto" | True | False
    "OLLAMA_HOST": "http://127.0.0.1:11434",
    "OLLAMA_MODEL": "mistral:7b",
    "EMBEDDING_BACKEND": "auto",      # "auto" | "ollama" | "chroma-default" | "hashing"
    "OLLAMA_EMBED_MODEL": "nomic-embed-text",
    "SDN_BACKEND": "ryu",             # "ryu" (Ryu/emulador) | "tfs" (ETSI TeraFlowSDN)
    "TFS_URL": "http://127.0.0.1:80",
    "TFS_CONTEXT": "admin",
    "TFS_DRY_RUN": True,
}

CONFIG: dict[str, Any] = {}


def load_config() -> dict[str, Any]:
    """Carga la configuración por defecto y la sobreescribe con workshop_config.json."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as fh:
            user_cfg = json.load(fh)
        ports = {**cfg["PORTS"], **user_cfg.get("PORTS", {})}
        cfg.update(user_cfg)
        cfg["PORTS"] = ports
    CONFIG.clear()
    CONFIG.update(cfg)
    return CONFIG


def save_config(**overrides: Any) -> dict[str, Any]:
    """Actualiza la configuración en memoria y la persiste en disco."""
    if not CONFIG:
        load_config()
    for key, value in overrides.items():
        if key == "PORTS":
            CONFIG["PORTS"].update(value)
        else:
            CONFIG[key] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        json.dump(CONFIG, fh, indent=2)
    return CONFIG


def url(service: str) -> str:
    """URL base de un servicio del workshop, p.ej. url('pce') -> http://127.0.0.1:8003"""
    return f"http://{CONFIG['HOST']}:{CONFIG['PORTS'][service]}"


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


# =============================================================================
# Prompt de sistema del planificador (compartido con el precalentamiento del LLM)
# =============================================================================
EXTRACTION_SYSTEM_PROMPT = (
    "You are the planning component of a telecom network operations system. "
    "Your ONLY task is to extract parameter values from the operator request. "
    "Do not compute routes and do not invent values. "
    "Network node identifiers are short uppercase codes such as H1, H2, R1, R3, ONU2. "
    "Bandwidths must be expressed in Gb/s as a plain number (e.g. '1GB' -> 1, '500 Mbps' -> 0.5). "
    "Reply ONLY with a JSON object whose keys are exactly: {keys}. "
    "Parameter meanings: {meanings}"
)
PARAM_MEANINGS = {
    "src_node": "source node of the connection",
    "dst_node": "destination node of the connection",
    "bitrate_gbps": "requested bandwidth in Gb/s (number)",
    "leaf_node": "leaf node (ONU / access node) of the point-to-multipoint tree",
    "num_sc": "number of digital subcarriers (integer)",
}
LLM_OPTIONS = {"temperature": 0, "seed": 42}
LLM_KEEP_ALIVE = "60m"


def extraction_system_prompt(keys) -> str:
    return EXTRACTION_SYSTEM_PROMPT.format(
        keys=", ".join(keys), meanings="; ".join(f"{k}: {PARAM_MEANINGS.get(k, k)}" for k in keys))


def warmup_llm(host: str, model: str, prompt: str,
               keys=("src_node", "dst_node", "bitrate_gbps")) -> threading.Thread:
    """Carga el modelo en Ollama y deja en su caché el prompt de sistema del planificador.

    En CPU, la primera evaluación de ese prompt puede tardar minutos; hacerlo en segundo
    plano al inicio evita esperas cuando el planificador lo use (Estación 3).
    """
    def _run() -> None:
        try:
            requests.post(f"{host.rstrip('/')}/api/chat", timeout=900, json={
                "model": model, "stream": False, "format": "json", "keep_alive": LLM_KEEP_ALIVE,
                "options": LLM_OPTIONS,
                "messages": [{"role": "system", "content": extraction_system_prompt(keys)},
                             {"role": "user", "content": prompt}]})
        except Exception:
            pass
    thread = threading.Thread(target=_run, name="llm-warmup", daemon=True)
    thread.start()
    return thread


load_config()


# =============================================================================
# Captura de tráfico estilo Wireshark
# =============================================================================
class TrafficCapture:
    """Registro ordenado de los mensajes intercambiados entre componentes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frames: list[dict[str, Any]] = []
        self._t0 = time.perf_counter()

    def reset(self) -> None:
        with self._lock:
            self.frames = []
            self._t0 = time.perf_counter()

    def record(self, src: str, dst: str, protocol: str, info: str,
               payload: Any = None, length: int | None = None) -> dict[str, Any]:
        with self._lock:
            if payload is None:
                body = ""
            elif isinstance(payload, str):
                body = payload
            else:
                body = json.dumps(payload)
            frame = {
                "no": len(self.frames) + 1,
                "time": time.perf_counter() - self._t0,
                "src": src,
                "dst": dst,
                "protocol": protocol,
                "length": length if length is not None else 66 + len(body.encode("utf-8")),
                "info": info,
                "payload": payload,
            }
            self.frames.append(frame)
            return frame

    def filter(self, protocols: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        with self._lock:
            frames = list(self.frames)
        if protocols is None:
            return frames
        return [f for f in frames if f["protocol"] in protocols]

    def to_table(self, protocols: tuple[str, ...] | None = ("HTTP", "OpenFlow"),
                 annotate: Callable[[dict[str, Any]], str] | None = None) -> str:
        frames = self.filter(protocols)
        header = f"{'No.':>4}  {'Time':>10}  {'Source':<13} {'Destination':<13} {'Protocol':<9} {'Length':>6}  "
        if annotate:
            header += f"{'Paso':<5} "
        header += "Info"
        lines = [header, "-" * max(len(header), 110)]
        for i, f in enumerate(frames, 1):
            line = (f"{i:>4}  {f['time']:>10.6f}  {f['src']:<13.13} {f['dst']:<13.13} "
                    f"{f['protocol']:<9.9} {f['length']:>6}  ")
            if annotate:
                line += f"{annotate(f):<5} "
            lines.append(line + f["info"])
        return "\n".join(lines)

    def show(self, protocols: tuple[str, ...] | None = ("HTTP", "OpenFlow"),
             annotate: Callable[[dict[str, Any]], str] | None = None) -> None:
        print(self.to_table(protocols, annotate))


CAPTURE = TrafficCapture()


class HTTPCallError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def http_call(src: str, dst: str, method: str, full_url: str, payload: Any = None,
              timeout: float = 600.0) -> Any:
    """Petición HTTP instrumentada: registra request y response en CAPTURE."""
    rest = full_url.split("://", 1)[-1]
    path = rest[rest.find("/"):] if "/" in rest else "/"
    info = f"{method} {path} HTTP/1.1"
    if payload is not None:
        info += "  (application/json)"
    CAPTURE.record(src, dst, "HTTP", info, payload)
    resp = requests.request(method, full_url, json=payload, timeout=timeout)
    try:
        data: Any = resp.json()
    except ValueError:
        data = resp.text
    CAPTURE.record(dst, src, "HTTP", f"HTTP/1.1 {resp.status_code} {resp.reason}  (application/json)", data)
    if resp.status_code >= 400:
        raise HTTPCallError(resp.status_code, data)
    return data


# =============================================================================
# Servidores HTTP en segundo plano (no bloquean el kernel de Jupyter)
# =============================================================================
_SERVERS: dict[str, tuple[Any, threading.Thread, int]] = {}


def port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_port(host: str, port: int, timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(host, port):
            return True
        time.sleep(0.1)
    return False


def start_server(name: str, app: Any, port_key: str) -> str:
    """Lanza ``app`` (FastAPI) en un hilo daemon y devuelve su URL base.

    Si el puerto configurado está ocupado por otro proceso, se elige uno libre
    y se actualiza ``workshop_config.json`` para que el resto de componentes lo use.
    """
    import uvicorn

    class _ThreadedServer(uvicorn.Server):
        def install_signal_handlers(self) -> None:  # compatibilidad uvicorn antiguo
            pass

    stop_server(name)
    host = CONFIG["HOST"]
    port = int(CONFIG["PORTS"][port_key])
    if port_in_use(host, port):
        new_port = find_free_port(host)
        print(f"⚠️  Puerto {port} ocupado: '{name}' usará el puerto {new_port}")
        port = new_port
        save_config(PORTS={port_key: port})
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", lifespan="off")
    server = _ThreadedServer(config)
    thread = threading.Thread(target=server.run, name=f"srv-{name}", daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"No se pudo iniciar el servidor '{name}' en {host}:{port}")
    _SERVERS[name] = (server, thread, port)
    return f"http://{host}:{port}"


def stop_server(name: str) -> None:
    entry = _SERVERS.pop(name, None)
    if entry is None:
        return
    server, thread, _ = entry
    server.should_exit = True
    thread.join(timeout=5)


def stop_all_servers() -> None:
    for name in list(_SERVERS):
        stop_server(name)


def running_servers() -> dict[str, str]:
    return {name: f"http://{CONFIG['HOST']}:{port}" for name, (_, _, port) in _SERVERS.items()}
