"""
workshop_dashboard.py
=====================
Visualización interactiva final del workshop (Módulo 8).

Ejecuta un escenario REAL sobre la arquitectura levantada en el notebook
(Orquestador -> LLM/ChromaDB -> Agente A2A -> MCP -> PCE/SDN -> OVS) y permite
recorrer, mensaje a mensaje, lo que ocurrió:

  * Arquitectura animada: qué componente habla con cuál en cada instante.
  * Captura estilo Wireshark sincronizada con la animación.
  * Plano de datos: ruta calculada, conexión provisionada, estado de enlaces,
    tablas de flujo OpenFlow por switch y ping antes/después.
  * Razonamiento: similitud de ChromaDB, argumentos extraídos, plan, traza y
    selección del agente (Handoff A2A).
  * Espectro P2MP del Hub (Extensión 2).

Solo usa ipywidgets + HTML/SVG (sin JavaScript propio), por lo que funciona en
JupyterLab, VS Code y Colab. Sin ipywidgets muestra una vista estática.
"""
import copy
import html
import json
import math
import time
from contextlib import contextmanager

import requests

import llm_planner

try:
    import ipywidgets as widgets
    from IPython.display import HTML, display
    WIDGETS_AVAILABLE = True
except ImportError:  # pragma: no cover
    widgets = None
    WIDGETS_AVAILABLE = False
    from IPython.display import HTML, display

STYLE = """
<style>
.wsd{background:#0b1220;color:#e6edf7;font-family:'Segoe UI',Roboto,Arial,sans-serif;border-radius:10px;padding:10px 12px;font-size:13px;line-height:1.4}
.wsd h4{margin:2px 0 8px;color:#4fc3f7;font-size:15px}
.wsd .mut{color:#9fb0cc}.wsd .ok{color:#81c784}.wsd .bad{color:#e57373}.wsd .acc{color:#4fc3f7}.wsd .acc2{color:#ffb74d}
.wsd pre{background:#0a1322;border:1px solid #2a3a5c;border-radius:8px;padding:8px 10px;font-family:Consolas,Menlo,monospace;font-size:11.5px;color:#d6e4f5;white-space:pre-wrap;word-break:break-word;margin:4px 0;max-height:300px;overflow:auto}
.wsd table{border-collapse:collapse;width:100%;font-size:12px;color:#e6edf7}
.wsd th,.wsd td{border-bottom:1px solid #22314f;padding:3px 6px;text-align:left;vertical-align:top;color:#e6edf7}
.wsd b{color:#ffffff}
.wsd th{color:#4fc3f7;background:#0f1a2e;position:sticky;top:0}
.wsd tr.cur td{background:#123a52;color:#fff}
.wsd tr.past td{color:#c9d6ea}.wsd tr.fut td{color:#56698c}
.wsd .card{background:#121b2e;border:1px solid #2a3a5c;border-radius:8px;padding:8px 10px;margin:4px 0}
.wsd .grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.wsd .pill{display:inline-block;border:1px solid #2a3a5c;border-radius:999px;padding:1px 9px;margin:2px 4px 2px 0;font-size:12px;background:#18233b}
.wsd .bar{height:14px;border-radius:4px;background:#2a3a5c}
.wsd .bar>div{height:14px;border-radius:4px}
</style>
"""

# --------------------------------------------------------------------- geometría del diagrama
NODES = {
    "User":         (15, 25, 135, 54, "👤 Operador", "intención"),
    "Orchestrator": (205, 25, 175, 54, "🧭 Orquestador", "POST /prompt"),
    "LLM":          (450, 18, 230, 68, "🧠 LLM + ChromaDB", "POST /plan · Mistral 7B"),
    "Agent":        (205, 165, 175, 54, "🤖 IPConnectionAgent", "A2A /execute_plan"),
    "OpticalAgent": (205, 265, 175, 54, "🔆 OpticalP2MPAgent", "A2A /execute_plan"),
    "MCP Server":   (450, 205, 175, 54, "🧰 Servidor MCP", "tools/list · tools/call"),
    "PCE":          (750, 110, 200, 54, "📐 PCE + Inventario", "POST /compute_route"),
    "SDN":          (750, 215, 200, 54, "🎛️ Controlador SDN", "NBI /service/ip/*"),
    "TFS (dry)":    (750, 18, 200, 54, "☁️ ETSI TFS (dry-run)", "/tfs-api/…/services"),
    "OVS r1":       (575, 380, 80, 44, "OVS R1", ""),
    "OVS r2":       (705, 380, 80, 44, "OVS R2", ""),
    "OVS r3":       (835, 380, 80, 44, "OVS R3", ""),
}
PROTO_COLORS = {"req": "#4fc3f7", "resp": "#81c784", "MCP": "#ffb74d", "OpenFlow": "#f06292"}


def _esc(text):
    return html.escape(str(text))


def _center(name):
    x, y, w, h = NODES[name][:4]
    return x + w / 2, y + h / 2


def _clip(name, dx, dy):
    """Distancia desde el centro hasta el borde del rectángulo en la dirección (dx, dy)."""
    _, _, w, h = NODES[name][:4]
    tx = (w / 2) / abs(dx) if dx else math.inf
    ty = (h / 2) / abs(dy) if dy else math.inf
    return min(tx, ty)


def _arrow_coords(src, dst):
    (x1, y1), (x2, y2) = _center(src), _center(dst)
    dx, dy = x2 - x1, y2 - y1
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    off = 5 if src < dst else -5                     # separa request/response
    px, py = -uy * off, ux * off
    a = _clip(src, ux, uy) + 2
    b = _clip(dst, -ux, -uy) + 4
    return (x1 + ux * a + px, y1 + uy * a + py, x2 - ux * b + px, y2 - uy * b + py)


def frame_color(frame):
    if frame["protocol"] == "HTTP":
        return PROTO_COLORS["resp"] if frame["info"].startswith("HTTP/1.1") else PROTO_COLORS["req"]
    return PROTO_COLORS.get(frame["protocol"], "#9fb0cc")


def explain(frame):
    """(paso, explicación) en lenguaje claro para cada trama capturada."""
    s, d, info, proto = frame["src"], frame["dst"], frame["info"], frame["protocol"]
    payload = frame.get("payload")
    is_resp = info.startswith("HTTP/1.1")
    if proto == "OpenFlow":
        if "PACKET_IN" in info:
            return "", f"El switch {d if s == 'SDN' else s} no tiene regla para el paquete y lo sube al controlador (PACKET_IN)."
        if "DELETE" in info:
            return "6", "El controlador BORRA las reglas del servicio (FLOW_MOD DELETE por cookie): vuelve el aislamiento."
        return "6", f"El controlador Ryu programa {d} con un OFPT_FLOW_MOD (OpenFlow 1.3, interfaz Sur)."
    if proto == "MCP":
        if "initialize" in info:
            return "MCP", f"{s} abre una sesión MCP con el servidor de herramientas."
        if "tools/list" in info:
            return "MCP", "El agente descubre las herramientas: nombre, descripción y JSON Schema de argumentos."
        if "result:" in info and "tools" in info:
            return "MCP", "El servidor MCP responde con el catálogo de herramientas disponibles."
        if "tools/call" in info:
            return "MCP", f"El agente invoca una herramienta determinista: {info.split('name=')[-1]} con {json.dumps(payload)}."
        return "MCP", "El servidor MCP devuelve el resultado determinista de la herramienta."
    if s == "User" and "/prompt" in info:
        return "1", f"El operador envía su intención en lenguaje natural: {payload.get('prompt') if isinstance(payload, dict) else ''!r}."
    if s == "User" and "/approve" in info:
        decision = "APRUEBA" if isinstance(payload, dict) and payload.get("approved") else "RECHAZA"
        return "HITL", f"El operador {decision} el paso pendiente (Human-in-the-Loop)."
    if s == "Orchestrator" and d == "LLM":
        return "2", "El Orquestador pide un plan: ChromaDB recupera el workflow más similar y el LLM extrae los argumentos."
    if s == "LLM" and is_resp:
        return "2", "El servicio LLM devuelve el plan estructurado {steps, plans} (Figura 5, mensaje 2)."
    if s == "Orchestrator" and "/.well-known" in info:
        return "A2A", f"El Orquestador lee la Agent Card de {d} (descubrimiento A2A)."
    if s == "Orchestrator" and "/execute_plan" in info:
        return "3", f"Handoff A2A: el Orquestador delega el plan a {d}, cuyas skills cubren las herramientas pedidas."
    if s == "Orchestrator" and "/a2a" in info:
        return "HITL", f"El Orquestador reenvía la decisión al agente: JSON-RPC SendMessage sobre la misma taskId."
    if s == "Agent" and d == "PCE":
        return "4", "Vía MCP, la herramienta compute_route consulta al PCE (cálculo determinista de ruta)."
    if s == "PCE" and is_resp:
        path = payload.get("path") if isinstance(payload, dict) else None
        return "4", f"El PCE responde la ruta: {' → '.join(path) if path else payload}."
    if d == "SDN" and "/service/ip/create" in info:
        return "5", "Vía MCP, establish_ip_connection invoca la NBI del controlador (POST /service/ip/create)."
    if d == "SDN" and "/service/ip/delete" in info:
        return "5", "Vía MCP, remove_ip_connection invoca la NBI del controlador (DELETE /service/ip/delete)."
    if d == "SDN" and "/service/ip/status" in info:
        return "5", "Vía MCP, get_connection_status consulta la NBI del controlador (GET /service/ip/status)."
    if s == "SDN" and is_resp:
        return "5", "El controlador SDN confirma la operación (HTTP 200 OK)."
    if d.startswith("TFS"):
        return "5", "La misma herramienta MCP genera el descriptor para ETSI TeraFlowSDN (modo dry-run)."
    if d == "Orchestrator" and is_resp:
        state = (payload or {}).get("status", {}).get("state", "") if isinstance(payload, dict) else ""
        if isinstance(payload, dict) and "task" in (payload.get("result") or {}):
            state = payload["result"]["task"]["status"]["state"]
        if state == "TASK_STATE_INPUT_REQUIRED":
            return "7", f"{s} pausa la Task en {state}: espera la aprobación del operador (HITL)."
        return "7", f"{s} devuelve la Task A2A en estado {state or 'final'} con la traza como artifact."
    if s == "Orchestrator" and d == "User" and is_resp:
        if isinstance(payload, dict) and payload.get("status") == "TASK_STATE_INPUT_REQUIRED":
            return "HITL", "El Orquestador muestra al operador la ruta propuesta y le pide aprobación."
        return "8", "El operador recibe la respuesta final: {\"trace\": [...]} (Figura 5, mensaje 8)."
    return "", info


# ===================================================================== dashboard
class WorkshopDashboard:
    def __init__(self, *, wc, orchestrator, orch_url, sdn_url, planner, kb, inventory, ip_agent,
                 prompt, ping_fn, net=None, optical_agent=None, p2mp=None, twin=None,
                 simulation_mode=True):
        self.wc, self.orch, self.orch_url, self.sdn_url = wc, orchestrator, orch_url, sdn_url
        self.planner, self.kb, self.inventory, self.ip_agent = planner, kb, inventory, ip_agent
        self.prompt, self.ping_fn, self.net = prompt, ping_fn, net
        self.optical_agent, self.p2mp, self.twin = optical_agent, p2mp, twin
        self.simulation_mode = simulation_mode
        self.real_llm = planner.llm
        self._no_llm = llm_planner.LocalLLM(enabled=False)
        self.result = None
        self.scenarios = self._build_scenarios()

    # ------------------------------------------------------------------ escenarios
    def _conn_exists(self):
        st = requests.get(f"{self.sdn_url}/service/ip/status", timeout=10).json()
        return "H1-H2" in st.get("connections", {})

    def _remove_conn(self):
        if self._conn_exists():
            requests.delete(f"{self.sdn_url}/service/ip/delete", params={"connection_id": "H1-H2"}, timeout=10)

    def _ensure_conn(self):
        requests.post(f"{self.sdn_url}/service/ip/create", timeout=10, json={
            "src_node": "H1", "dst_node": "H2", "bitrate_gbps": 1, "path": ["H1", "R1", "R2", "R3", "H2"]})

    def _degrade(self):
        self._ensure_conn()
        self.twin.inject_degradation("R2", "R3", extra_ms=2.0, ber=6e-3)
        self.inventory.set_link_state("R1", "R3", "up")

    def _undegrade(self):
        self.twin.clear()
        self.inventory.set_link_state("R1", "R3", "standby")

    def _build_scenarios(self):
        sc = {
            "paper": {"label": "① Demo del paper: conectar H1 → H2 con 1 Gb/s", "prompt": self.prompt,
                      "setup": self._remove_conn,
                      "about": "Reproduce las Figuras 3, 4 y 5: del aislamiento a la conectividad."},
            "hitl": {"label": "② Conectar H1 → H2 con aprobación humana (HITL)", "prompt": self.prompt,
                     "setup": self._remove_conn, "hitl": True,
                     "about": "La Task A2A se pausa en INPUT_REQUIRED tras el PCE; usted decide si se programa la red."},
            "status": {"label": "③ Consultar el estado de las conexiones",
                       "prompt": "Show me the status of the connections",
                       "about": "Otro workflow de ChromaDB: solo lectura de la NBI (GET /service/ip/status)."},
            "route": {"label": "④ Solo calcular la ruta H1 → H2 (what-if)",
                      "prompt": "Find the shortest path between H1 and H2",
                      "about": "Consulta al PCE sin tocar la red: no hay FLOW_MOD."},
            "remove": {"label": "⑤ Eliminar la conexión H1–H2 (vuelve el aislamiento)",
                       "prompt": "Remove the IP connection between H1 and H2", "setup": self._ensure_conn,
                       "about": "FLOW_MOD DELETE por cookie: el ping vuelve a fallar."},
        }
        if self.optical_agent is not None and self.p2mp is not None:
            sc["p2mp"] = {"label": "⑥ Ext. 2 · Asignar 3 subportadoras a ONU2 (agente óptico P2MP)",
                          "prompt": "Allocate 3 subcarriers to leaf ONU2 in the P2MP access tree",
                          "about": "El Orquestador hace Handoff al OpticalP2MPAgent por su Agent Card."}
        if self.twin is not None:
            sc["degradation"] = {"label": "⑦ Ext. 3 · Degradación en R2–R3 → diagnóstico y re-enrutamiento",
                                 "prompt": "Diagnose the latency degradation between H1 and H2 and reconfigure the connection",
                                 "setup": self._degrade, "teardown": self._undegrade,
                                 "about": "El gemelo digital detecta la degradación; el PCE evita R2–R3 y usa el bypass óptico."}
        sc["custom"] = {"label": "✍️ Prompt libre (escriba su propia intención)", "prompt": None,
                        "about": "Pruebe sus propias frases: ¿qué workflow recupera ChromaDB?"}
        return sc

    @contextmanager
    def _llm_mode(self, use_llm):
        agents = [a for a in (self.ip_agent, self.optical_agent) if a is not None]
        saved = (self.planner.llm, [(a, a.llm) for a in agents])
        if not (use_llm and self.real_llm.available):
            self.planner.llm = self._no_llm
            for a in agents:
                a.llm = None
        try:
            yield
        finally:
            self.planner.llm = saved[0]
            for a, l in saved[1]:
                a.llm = l

    # ------------------------------------------------------------------ ejecución
    def _snapshot_links(self):
        snap = {}
        for u, v, d in self.inventory.graph.edges(data=True):
            snap[frozenset((u, v))] = {"u": u, "v": v, **{k: d[k] for k in ("layer", "state", "capacity_gbps", "used_gbps")}}
        return snap

    def run(self, key="paper", custom_prompt="", use_llm=False, approve=True):
        sc = self.scenarios[key]
        prompt = (custom_prompt or "").strip() if key == "custom" else sc["prompt"]
        if not prompt:
            raise ValueError("Escriba un prompt")
        if sc.get("setup"):
            sc["setup"]()
        ping_before = self.ping_fn()
        error, response, hitl_info = None, None, None
        links_during = None
        t0 = time.time()
        with self._llm_mode(use_llm):
            self.wc.CAPTURE.reset()
            try:
                if sc.get("hitl"):
                    self.ip_agent.hitl = True
                response = self.wc.http_call("User", "Orchestrator", "POST", f"{self.orch_url}/prompt",
                                             {"prompt": prompt})
                if isinstance(response, dict) and response.get("status") == "TASK_STATE_INPUT_REQUIRED":
                    hitl_info = response
                    response = self.wc.http_call("User", "Orchestrator", "POST", f"{self.orch_url}/approve",
                                                 {"task_id": response["task_id"], "approved": bool(approve)})
            except Exception as exc:
                error = getattr(exc, "detail", None) or str(exc)
            finally:
                self.ip_agent.hitl = False
                links_during = self._snapshot_links()
                frames = copy.deepcopy(self.wc.CAPTURE.frames)
                if sc.get("teardown"):
                    sc["teardown"]()
        elapsed = time.time() - t0
        ping_after = self.ping_fn()
        status = requests.get(f"{self.sdn_url}/service/ip/status", timeout=10).json()
        trace = response.get("trace", []) if isinstance(response, dict) else []
        computed = next((s["result"]["path"] for s in trace
                         if isinstance(s.get("result"), dict) and "path" in s["result"] and s["tool"].startswith("compute_route")), None)
        if computed is None and hitl_info:
            computed = (hitl_info.get("details", {}).get("last_result") or {}).get("path")
        flows = {}
        if self.net is not None:
            for sid, sw in self.net.switches.items():
                flows[sid] = sw.dump_flows()
        self.result = {
            "key": key, "label": sc["label"], "about": sc.get("about", ""), "prompt": prompt,
            "use_llm": bool(use_llm and self.real_llm.available), "elapsed": elapsed,
            "frames": frames, "response": response, "error": error, "hitl": hitl_info, "approve": approve,
            "plan": copy.deepcopy(self.orch.last_plan), "handoff": self.orch.last_handoff,
            "retrieval": self.kb.search(prompt, k=self.kb.collection.count()),
            "extractor": self.planner.last_backend, "raw": copy.deepcopy(self.planner.last_raw),
            "ping_before": ping_before, "ping_after": ping_after, "sdn_status": status,
            "links": links_during, "computed_path": computed, "flows": flows,
            "p2mp": list(self.p2mp.slots) if self.p2mp is not None else None,
            "agents": {n: sorted(a["tools"]) for n, a in self.orch.agents.items()},
        }
        return self.result

    def visible_frames(self, show_mcp=True, show_of=True):
        if not self.result:
            return []
        out = []
        for f in self.result["frames"]:
            if f["protocol"] == "MCP" and not show_mcp:
                continue
            if f["protocol"] == "OpenFlow" and not show_of:
                continue
            out.append(f)
        return out

    # ------------------------------------------------------------------ renderizado
    def render_summary(self):
        r = self.result
        if not r:
            return STYLE + "<div class='wsd'>Sin ejecución todavía.</div>"
        counts = {}
        for f in r["frames"]:
            counts[f["protocol"]] = counts.get(f["protocol"], 0) + 1
        ok_before = "64 bytes" in r["ping_before"]
        ok_after = "64 bytes" in r["ping_after"]
        status = (f"<span class='bad'>❌ Error: {_esc(r['error'])}</span>" if r["error"]
                  else "<span class='ok'>✅ Flujo completado</span>")
        if r["hitl"] and not r["approve"]:
            status = "<span class='acc2'>🛑 Operación rechazada por el operador (Task CANCELED)</span>"
        llm = "Mistral 7B real (Ollama)" if r["use_llm"] else "LLM emulado (rápido)"
        pills = "".join(f"<span class='pill'>{_esc(k)}: {v}</span>" for k, v in counts.items())
        return STYLE + f"""<div class='wsd'>
<h4>{_esc(r['label'])}</h4>
<div>🗣️ <b>Prompt:</b> <code>{_esc(r['prompt'])}</code></div>
<div class='mut'>{_esc(r['about'])}</div>
<div style='margin-top:6px'>{status} · ⏱️ {r['elapsed']:.1f} s · 🧠 {llm} (extractor: {_esc(r['extractor'])})
 · 🤝 Handoff → <b>{_esc(r['handoff'])}</b></div>
<div style='margin-top:4px'>Plano de datos <code>h1 ping h2</code>: antes {'<span class=ok>✅ alcanzable</span>' if ok_before else '<span class=bad>⛔ inalcanzable</span>'}
 → después {'<span class=ok>✅ alcanzable</span>' if ok_after else '<span class=bad>⛔ inalcanzable</span>'}</div>
<div style='margin-top:4px'>Mensajes capturados: {pills}</div></div>"""

    def render_architecture(self, frames, idx):
        cur = frames[idx] if frames else None
        parts = [f"<svg viewBox='0 0 970 440' width='100%' style='background:#0b1220;border-radius:10px'>",
                 "<defs>" + "".join(
                     f"<marker id='m{k}' markerWidth='9' markerHeight='9' refX='8' refY='4.5' orient='auto'>"
                     f"<path d='M0,0 L9,4.5 L0,9 z' fill='{c}'/></marker>" for k, c in
                     [("req", PROTO_COLORS["req"]), ("resp", PROTO_COLORS["resp"]), ("MCP", PROTO_COLORS["MCP"]),
                      ("OpenFlow", PROTO_COLORS["OpenFlow"]), ("past", "#3a4c70")]) + "</defs>"]
        parts.append("<rect x='545' y='355' width='415' height='80' rx='10' fill='#122318' stroke='#2e5a3a'/>"
                     "<text x='555' y='372' fill='#81c784' font-size='12'>Plano de datos (Mininet / OVS)</text>")
        active = {cur["src"], cur["dst"]} if cur else set()
        used = set()
        for f in frames[: idx + 1]:
            used |= {f["src"], f["dst"]}
        # aristas pasadas (tenues)
        seen = set()
        for f in frames[:idx]:
            if f["src"] in NODES and f["dst"] in NODES and (f["src"], f["dst"]) not in seen:
                seen.add((f["src"], f["dst"]))
                x1, y1, x2, y2 = _arrow_coords(f["src"], f["dst"])
                parts.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='#3a4c70' "
                             f"stroke-width='1.5' marker-end='url(#mpast)'/>")
        for name, (x, y, w, h, title, sub) in NODES.items():
            if name == "OpticalAgent" and self.optical_agent is None:
                continue
            if name == "TFS (dry)" and name not in used:
                continue
            stroke = "#ffffff" if name in active else ("#4fc3f7" if name in used else "#2a3a5c")
            fill = "#17324a" if name in active else "#121b2e"
            sw = 3 if name in active else 1.3
            parts.append(f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='9' fill='{fill}' stroke='{stroke}' stroke-width='{sw}'/>")
            parts.append(f"<text x='{x + w / 2}' y='{y + (h / 2 + 4 if not sub else 23)}' fill='#fff' font-size='13' "
                         f"font-weight='700' text-anchor='middle'>{_esc(title)}</text>")
            if sub:
                parts.append(f"<text x='{x + w / 2}' y='{y + 41}' fill='#9fb0cc' font-size='11' text-anchor='middle'>{_esc(sub)}</text>")
        parts.append("<g font-size='11' fill='#9fb0cc'><text x='575' y='432'>H1 ─ R1 ─ R2 ─ R3 ─ H2</text></g>")
        if cur and cur["src"] in NODES and cur["dst"] in NODES:
            x1, y1, x2, y2 = _arrow_coords(cur["src"], cur["dst"])
            color = frame_color(cur)
            mk = "OpenFlow" if cur["protocol"] == "OpenFlow" else ("MCP" if cur["protocol"] == "MCP" else
                                                                  ("resp" if cur["info"].startswith("HTTP/1.1") else "req"))
            parts.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='{color}' "
                         f"stroke-width='4.5' marker-end='url(#m{mk})'/>")
            step, _ = explain(cur)
            if step:
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                bw = 22 if len(step) <= 2 else 12 + 8 * len(step)
                parts.append(f"<rect x='{mx - bw / 2:.0f}' y='{my - 11:.0f}' width='{bw}' height='22' rx='11' fill='{color}'/>"
                             f"<text x='{mx:.0f}' y='{my + 4:.0f}' fill='#04121f' font-size='11' font-weight='800' "
                             f"text-anchor='middle'>{_esc(step)}</text>")
        parts.append("<g font-size='11'>"
                     f"<text x='15' y='395' fill='{PROTO_COLORS['req']}'>━ HTTP request</text>"
                     f"<text x='15' y='412' fill='{PROTO_COLORS['resp']}'>━ HTTP response</text>"
                     f"<text x='150' y='395' fill='{PROTO_COLORS['MCP']}'>━ MCP (JSON-RPC)</text>"
                     f"<text x='150' y='412' fill='{PROTO_COLORS['OpenFlow']}'>━ OpenFlow 1.3</text>"
                     "<text x='15' y='429' fill='#6b7f9e'>etiqueta sobre la flecha = paso de la Figura 2 del paper</text></g>")
        parts.append("</svg>")
        return "".join(parts)

    def render_detail(self, frames, idx):
        if not frames:
            return STYLE + "<div class='wsd'>No hay mensajes con los filtros actuales.</div>"
        f = frames[idx]
        step, text = explain(f)
        payload = f.get("payload")
        body = ""
        if payload not in (None, ""):
            raw = json.dumps(payload, indent=2, ensure_ascii=False) if not isinstance(payload, str) else payload
            if len(raw) > 2500:
                raw = raw[:2500] + "\n… (truncado)"
            body = f"<pre>{_esc(raw)}</pre>"
        return STYLE + f"""<div class='wsd'>
<div><span class='pill' style='background:{frame_color(f)};color:#04121f;font-weight:700'>{_esc(f['protocol'])}</span>
<b>Mensaje {idx + 1} / {len(frames)}</b> · t = {f['time']:.4f} s · {('Paso ' + _esc(step)) if step else ''}</div>
<div style='margin:6px 0'><b>{_esc(f['src'])}</b> → <b>{_esc(f['dst'])}</b> &nbsp;<code>{_esc(f['info'])}</code></div>
<div class='card'>💡 {_esc(text)}</div>{body}</div>"""

    def render_table(self, frames, idx):
        rows = []
        for i, f in enumerate(frames):
            cls = "cur" if i == idx else ("past" if i < idx else "fut")
            step, _ = explain(f)
            rows.append(f"<tr class='{cls}'><td>{i + 1}</td><td>{f['time']:.4f}</td><td>{_esc(f['src'])}</td>"
                        f"<td>{_esc(f['dst'])}</td><td>{_esc(f['protocol'])}</td><td>{_esc(step)}</td>"
                        f"<td>{_esc(f['info'][:95])}</td></tr>")
        return STYLE + ("<div class='wsd'><div style='max-height:250px;overflow:auto'><table><tr><th>No.</th><th>Time</th>"
                        "<th>Source</th><th>Destination</th><th>Proto</th><th>Paso</th><th>Info</th></tr>"
                        + "".join(rows) + "</table></div></div>")

    def render_topology(self, switch="R2"):
        r = self.result
        if not r:
            return ""
        pos = {"H1": (60, 95), "R1": (230, 95), "R2": (420, 95), "R3": (610, 95), "H2": (780, 95)}
        conn = r["sdn_status"].get("connections", {}).get("H1-H2")
        conn_path = conn["path"] if conn else []
        conn_edges = {frozenset(e) for e in zip(conn_path, conn_path[1:])}
        comp = r["computed_path"] or []
        comp_edges = {frozenset(e) for e in zip(comp, comp[1:])}
        svg = ["<svg viewBox='0 0 840 250' width='100%' style='background:#0b1220;border-radius:10px'>"]
        for key, l in r["links"].items():
            (x1, y1), (x2, y2) = pos[l["u"]], pos[l["v"]]
            color = {"up": "#c9d6ea", "standby": "#ba68c8", "degraded": "#e57373"}.get(l["state"], "#9fb0cc")
            dash = "8 6" if l["state"] != "up" else ""
            residual = l["capacity_gbps"] - l["used_gbps"]
            if l["layer"] == "optical":
                d = f"M{x1},{y1 + 20} Q{(x1 + x2) / 2},{y1 + 150} {x2},{y2 + 20}"
                lx, ly = (x1 + x2) / 2, y1 + 100
            else:
                d = f"M{x1},{y1} L{x2},{y2}"
                lx, ly = (x1 + x2) / 2, y1 - 14
            if key in conn_edges:
                svg.append(f"<path d='{d}' fill='none' stroke='#4fc3f7' stroke-width='12' stroke-opacity='0.35'/>")
            if key in comp_edges:
                svg.append(f"<path d='{d}' fill='none' stroke='#ffb74d' stroke-width='3' stroke-dasharray='3 5' "
                           f"transform='translate(0,6)'/>")
            svg.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.5' stroke-dasharray='{dash}'/>")
            svg.append(f"<text x='{lx}' y='{ly}' fill='{color}' font-size='11' text-anchor='middle'>"
                       f"{l['state']} · {residual:g}/{l['capacity_gbps']:g} Gb/s libres</text>")
        for n, (x, y) in pos.items():
            if n.startswith("H"):
                svg.append(f"<circle cx='{x}' cy='{y}' r='24' fill='#9ecae1'/><text x='{x}' y='{y + 5}' "
                           f"fill='#04121f' font-weight='800' text-anchor='middle'>{n}</text>")
            else:
                stroke = "#ffffff" if n == switch else "#fdae6b"
                svg.append(f"<rect x='{x - 28}' y='{y - 20}' width='56' height='40' rx='6' fill='#fdae6b' "
                           f"stroke='{stroke}' stroke-width='3'/><text x='{x}' y='{y + 5}' fill='#04121f' "
                           f"font-weight='800' text-anchor='middle'>{n}</text>")
        svg.append("<g font-size='11'><text x='15' y='215' fill='#4fc3f7'>▬ conexión provisionada en el SDN</text>"
                   "<text x='15' y='232' fill='#ffb74d'>┅ ruta calculada por el PCE en esta ejecución</text>"
                   "<text x='330' y='215' fill='#ba68c8'>┅ bypass óptico en standby</text>"
                   "<text x='330' y='232' fill='#e57373'>┅ enlace degradado (gemelo digital)</text></g></svg>")
        conn_txt = (f"<b>H1-H2</b> UP por {' → '.join(conn_path)} ({conn['bitrate_gbps']:g} Gb/s, {conn['flow_mods']} FLOW_MOD)"
                    if conn else "<span class='bad'>sin conexión H1-H2: el aislamiento está activo</span>")
        flows = r["flows"].get(switch)
        flows_html = (f"<pre>$ ovs-ofctl -O OpenFlow13 dump-flows {switch.lower()}\n{_esc(flows)}</pre>" if flows is not None
                      else "<div class='mut'>Modo real: use <code>dump_flows('R2')</code> o <code>sudo ovs-ofctl -O OpenFlow13 dump-flows r2</code>.</div>")
        return STYLE + f"""<div class='wsd'>{''.join(svg)}
<div class='card'>🎛️ Estado del controlador SDN: {conn_txt}</div>
<div class='grid2'><div><h4>mininet&gt; h1 ping h2 · ANTES</h4><pre>{_esc(r['ping_before'])}</pre></div>
<div><h4>mininet&gt; h1 ping h2 · DESPUÉS</h4><pre>{_esc(r['ping_after'])}</pre></div></div>
<h4>Tabla de flujos de {switch} (seleccione el switch arriba)</h4>{flows_html}</div>"""

    def render_reasoning(self):
        r = self.result
        if not r:
            return ""
        bars = []
        best = r["retrieval"][0]["name"] if r["retrieval"] else None
        for h in r["retrieval"]:
            pct = max(0.0, min(1.0, h["similarity"])) * 100
            color = "#4fc3f7" if h["name"] == best else "#3a4c70"
            bars.append(f"<tr><td style='width:34%'>{'◀ ' if h['name'] == best else ''}<b>{_esc(h['name'])}</b></td>"
                        f"<td><div class='bar'><div style='width:{pct:.0f}%;background:{color}'></div></div></td>"
                        f"<td style='width:60px'>{h['similarity']:.3f}</td></tr>")
        agents = []
        needed = {p["tool"] for p in (r["plan"] or {}).get("plans", [])}
        for name, tools in r["agents"].items():
            sel = name == r["handoff"]
            tl = " ".join(f"<span class='pill' style='{'border-color:#81c784;color:#81c784' if t in needed else ''}'>{_esc(t)}</span>"
                          for t in tools)
            agents.append(f"<div class='card' style='{'border-color:#4fc3f7' if sel else ''}'>"
                          f"{'🤝 <b>' if sel else ''}{_esc(name)}{'</b> ← Handoff' if sel else ''}<br>{tl}</div>")
        plan = json.dumps(r["plan"], indent=2, ensure_ascii=False) if r["plan"] else "—"
        trace = json.dumps(r["response"], indent=2, ensure_ascii=False) if r["response"] else _esc(r["error"])
        return STYLE + f"""<div class='wsd'>
<h4>1 · ChromaDB: similitud coseno entre el prompt y cada workflow (zero-shot)</h4><table>{''.join(bars)}</table>
<h4 style='margin-top:10px'>2 · LLM: argumentos extraídos ({_esc(r['extractor'])})</h4><pre>{_esc(json.dumps(r['raw'], ensure_ascii=False))}</pre>
<h4>3 · A2A: agentes registrados (verde = herramienta que el plan necesita)</h4>{''.join(agents)}
<div class='grid2'><div><h4>Plan · mensaje 2</h4><pre>{_esc(plan)}</pre></div>
<div><h4>Respuesta al operador · mensaje 8</h4><pre>{_esc(trace)}</pre></div></div></div>"""

    def render_p2mp(self):
        r = self.result
        if not r or r["p2mp"] is None:
            return STYLE + "<div class='wsd mut'>La Extensión 2 (P2MP) no está disponible en esta sesión.</div>"
        palette = ["#4fc3f7", "#ffb74d", "#81c784", "#ba68c8", "#f06292", "#ffd54f"]
        leaves = sorted({s for s in r["p2mp"] if s})
        color = {l: palette[i % len(palette)] for i, l in enumerate(leaves)}
        cells = "".join(f"<rect x='{10 + i * 50}' y='20' width='46' height='46' rx='5' fill='{color.get(s, '#2a3a5c')}'/>"
                        f"<text x='{33 + i * 50}' y='48' font-size='11' text-anchor='middle' fill='#04121f' font-weight='700'>{i}</text>"
                        for i, s in enumerate(r["p2mp"]))
        legend = "".join(f"<span class='pill' style='border-color:{color[l]};color:{color[l]}'>{_esc(l)}: "
                         f"{r['p2mp'].count(l)} SC = {25 * r['p2mp'].count(l)} Gb/s</span>" for l in leaves)
        used = sum(1 for s in r["p2mp"] if s)
        return STYLE + f"""<div class='wsd'><h4>Espectro del Hub P2MP: 16 subportadoras × 25 Gb/s</h4>
<svg viewBox='0 0 820 80' width='100%'>{cells}</svg>
<div>{legend or '<span class=mut>sin asignaciones</span>'}</div>
<div class='mut' style='margin-top:6px'>Ocupación del Hub: {used}/16 ({used * 25} de 400 Gb/s). Política First-Fit contigua,
sin solapamiento espectral; máx. 4 SC por hoja. Ejecute el escenario ⑥ varias veces o con otras hojas en «Prompt libre»
(p.ej. <code>Allocate 4 subcarriers to leaf ONU3 in the P2MP access tree</code>).</div></div>"""

    def render_static(self):
        frames = self.visible_frames()
        idx = len(frames) - 1 if frames else 0
        return (self.render_summary() + self.render_architecture(frames, idx) + self.render_table(frames, idx)
                + self.render_topology() + self.render_reasoning() + self.render_p2mp())

    # ------------------------------------------------------------------ interfaz ipywidgets
    def show(self, default="paper"):
        if not WIDGETS_AVAILABLE:
            self.run(default)
            display(HTML(self.render_static()))
            print("ℹ️ Instale ipywidgets para la versión interactiva: pip install ipywidgets")
            return
        W = widgets
        scenario = W.Dropdown(options=[(v["label"], k) for k, v in self.scenarios.items()], value=default,
                              description="Escenario:", layout=W.Layout(width="620px"),
                              style={"description_width": "80px"})
        custom = W.Text(placeholder="p.ej. Establish an IP connection from H1 to H2 with 2 Gbps",
                        description="Prompt:", layout=W.Layout(width="620px", display="none"),
                        style={"description_width": "80px"})
        use_llm = W.Checkbox(value=False, indent=False,
                             description=("Usar Mistral 7B real (Ollama) — más lento en CPU" if self.real_llm.available
                                          else "Mistral 7B no disponible: se usa el LLM emulado"),
                             disabled=not self.real_llm.available, layout=W.Layout(width="420px"))
        decision = W.ToggleButtons(options=[("✅ Aprobar", True), ("🛑 Rechazar", False)], value=True,
                                   description="Operador:", layout=W.Layout(display="none"))
        run_btn = W.Button(description="▶ Ejecutar escenario", button_style="primary", icon="play",
                           layout=W.Layout(width="200px"))
        status = W.HTML()
        show_mcp = W.Checkbox(value=True, description="Mostrar MCP", indent=False, layout=W.Layout(width="140px"))
        show_of = W.Checkbox(value=True, description="Mostrar OpenFlow", indent=False, layout=W.Layout(width="160px"))
        play = W.Play(value=0, min=0, max=0, step=1, interval=1100, description="Reproducir")
        slider = W.IntSlider(value=0, min=0, max=0, description="Mensaje:", continuous_update=True,
                             layout=W.Layout(width="520px"))
        W.jslink((play, "value"), (slider, "value"))
        summary_html, arch_html, detail_html, table_html = W.HTML(), W.HTML(), W.HTML(), W.HTML()
        switch_sel = W.ToggleButtons(options=["R1", "R2", "R3"], value="R2", description="Switch:")
        topo_html, reason_html, p2mp_html = W.HTML(), W.HTML(), W.HTML()

        state = {"frames": []}

        def refresh_frames(*_):
            state["frames"] = self.visible_frames(show_mcp.value, show_of.value)
            n = max(len(state["frames"]) - 1, 0)
            slider.max = play.max = n
            slider.value = play.value = 0
            refresh_frame()

        def refresh_frame(*_):
            frames, idx = state["frames"], min(slider.value, max(len(state["frames"]) - 1, 0))
            arch_html.value = self.render_architecture(frames, idx)
            detail_html.value = self.render_detail(frames, idx)
            table_html.value = self.render_table(frames, idx)

        def refresh_all():
            summary_html.value = self.render_summary()
            topo_html.value = self.render_topology(switch_sel.value)
            reason_html.value = self.render_reasoning()
            p2mp_html.value = self.render_p2mp()
            refresh_frames()

        def on_scenario(change):
            custom.layout.display = "" if change["new"] == "custom" else "none"
            decision.layout.display = "" if self.scenarios[change["new"]].get("hitl") else "none"

        def on_run(_):
            run_btn.disabled = True
            status.value = ("<span style='color:#ffb74d'>⏳ Ejecutando el flujo agéntico… "
                            + ("(Mistral en CPU puede tardar 1–2 min)" if use_llm.value else "") + "</span>")
            try:
                self.run(scenario.value, custom.value, use_llm.value, decision.value)
                status.value = "<span style='color:#81c784'>Listo. Use ▶ o el deslizador para recorrer los mensajes.</span>"
                refresh_all()
            except Exception as exc:
                status.value = f"<span style='color:#e57373'>Error: {_esc(exc)}</span>"
            finally:
                run_btn.disabled = False

        scenario.observe(on_scenario, names="value")
        run_btn.on_click(on_run)
        slider.observe(refresh_frame, names="value")
        show_mcp.observe(refresh_frames, names="value")
        show_of.observe(refresh_frames, names="value")
        switch_sel.observe(lambda c: setattr(topo_html, "value", self.render_topology(c["new"])), names="value")

        tab_flow = W.VBox([W.HBox([play, slider, show_mcp, show_of]), arch_html, detail_html, table_html])
        tab = W.Tab(children=[tab_flow, W.VBox([switch_sel, topo_html]), reason_html, p2mp_html])
        for i, t in enumerate(["🔁 Flujo de mensajes", "🔀 Plano de datos", "🧠 Razonamiento y A2A", "🌈 Espectro P2MP"]):
            tab.set_title(i, t)
        controls = W.VBox([scenario, custom, W.HBox([use_llm, decision]), W.HBox([run_btn, status])])
        display(W.VBox([W.HTML(STYLE + "<div class='wsd'><h4>🛰️ Visualización interactiva del sistema multi-agéntico</h4>"
                                       "Elija un escenario y pulse <b>Ejecutar</b>: se ejecuta de verdad sobre los servicios "
                                       "levantados en este notebook. Luego recorra los mensajes con ▶ o el deslizador.</div>"),
                        controls, summary_html, tab]))
        on_scenario({"new": scenario.value})
        on_run(None)
