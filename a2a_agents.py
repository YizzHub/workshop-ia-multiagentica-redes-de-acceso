"""
a2a_agents.py
=============
Agentes especializados interoperables vía A2A (Agent2Agent, https://a2a-protocol.org).

  * ``SpecializedAgent``: clase base reutilizable (Agent Card, ciclo de vida de
    Task, sesión MCP, asociación paso->herramienta, Human-in-the-Loop).
  * ``IPConnectionAgent``: agente de la capa de paquetes del paper.
  * ``create_agent_app``: *bindings* HTTP del agente:
        GET  /.well-known/agent-card.json   (A2A v1.0; alias legado /.well-known/agent.json)
        POST /execute_plan                  (binding usado en el paper, Fig. 3)
        POST /a2a                           (JSON-RPC 2.0: SendMessage, GetTask)
"""
import asyncio
import copy
import datetime as dt
import uuid
from typing import Any

import workshop_common as wc
from knowledge_base import HashingEmbedder
from mcp_server import MCPSession

COMPLETED, FAILED = "TASK_STATE_COMPLETED", "TASK_STATE_FAILED"
WORKING, INPUT_REQUIRED, SUBMITTED = "TASK_STATE_WORKING", "TASK_STATE_INPUT_REQUIRED", "TASK_STATE_SUBMITTED"
CANCELED = "TASK_STATE_CANCELED"
TERMINAL_STATES = {COMPLETED, FAILED, CANCELED, "TASK_STATE_REJECTED"}

TOOL_MAPPING_PROMPT = (
    "You map a network operation step to exactly one tool. Available tools:\n{tools}\n"
    "Reply ONLY with JSON: {{\"tool\": \"<tool name>\"}}"
)


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SpecializedAgent:
    """Agente especializado A2A que ejecuta planes invocando herramientas MCP."""

    name = "SpecializedAgent"
    description = ""
    version = "1.0.0"
    skills: list[dict] = []
    allowed_tools: set[str] | None = None     # None = todas las herramientas MCP
    hitl_after_tools: set[str] = set()        # pausa HITL tras estas herramientas

    def __init__(self, base_url: str = "", llm: Any = None, hitl: bool = False,
                 mcp_factory: Any = MCPSession, label: str = "Agent"):
        self.base_url = base_url
        self.llm = llm
        self.hitl = hitl
        self.mcp_factory = mcp_factory
        self.label = label
        self.tasks: dict[str, dict] = {}
        self._plans: dict[str, dict] = {}
        self._embed = HashingEmbedder()
        self.last_associations: list[dict] = []

    # ------------------------------------------------------------------ Agent Card
    def agent_card(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "provider": {"organization": "Workshop Redes de Acceso", "url": self.base_url},
            "supportedInterfaces": [
                {"url": f"{self.base_url}/a2a", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"},
                {"url": f"{self.base_url}/execute_plan", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            ],
            "capabilities": {"streaming": False, "pushNotifications": False},
            "defaultInputModes": ["application/json", "text/plain"],
            "defaultOutputModes": ["application/json"],
            "skills": self.skills,
        }

    def supported_tools(self) -> set[str]:
        return {t.split(":", 1)[1] for s in self.skills for t in s.get("tags", []) if t.startswith("tool:")}

    # ------------------------------------------------------------------ Task lifecycle
    def _new_task(self, plan: dict) -> dict:
        task = {"id": str(uuid.uuid4()), "contextId": str(uuid.uuid4()),
                "status": {"state": SUBMITTED, "timestamp": _now()},
                "artifacts": [], "history": [], "metadata": {"agent": self.name}}
        self.tasks[task["id"]] = task
        self._plans[task["id"]] = {"plan": copy.deepcopy(plan), "cursor": 0, "trace": []}
        return task

    def _set_state(self, task: dict, state: str, text: str | None = None, data: Any = None) -> None:
        status: dict[str, Any] = {"state": state, "timestamp": _now()}
        if text or data is not None:
            parts: list[dict] = []
            if text:
                parts.append({"text": text})
            if data is not None:
                parts.append({"data": data})
            status["message"] = {"messageId": str(uuid.uuid4()), "role": "ROLE_AGENT",
                                 "taskId": task["id"], "contextId": task["contextId"], "parts": parts}
        task["status"] = status
        task["history"].append({"state": state, "timestamp": status["timestamp"]})

    # ------------------------------------------------------------------ Paso -> herramienta
    def _associate_semantic(self, intent: str, tools: dict[str, dict]) -> str:
        q = self._embed([intent])[0]
        scored = [(_cosine(q, self._embed([f"{n.replace('_', ' ')}: {t['description']}"])[0]), n)
                  for n, t in tools.items()]
        return max(scored)[1]

    def _associate_llm(self, intent: str, tools: dict[str, dict]) -> str | None:
        listing = "\n".join(f"- {n}: {t['description']}" for n, t in tools.items())
        try:
            answer = self.llm.chat_json(TOOL_MAPPING_PROMPT.format(tools=listing), f"Step: {intent}")
            tool = str(answer.get("tool", "")).strip()
            return tool if tool in tools else None
        except Exception:
            return None

    async def associate_tool(self, intent: str, planned_tool: str | None, tools: dict[str, dict]) -> str:
        """Asocia un paso del plan con una herramienta MCP (LLM si está disponible)."""
        method, chosen = "semantic", None
        if self.llm is not None and getattr(self.llm, "available", False):
            chosen = await asyncio.to_thread(self._associate_llm, intent, tools)
            method = f"llm:{self.llm.model}" if chosen else "semantic"
        if chosen is None:
            chosen = self._associate_semantic(intent, tools)
        if planned_tool and planned_tool not in tools:
            raise ValueError(f"La herramienta '{planned_tool}' no está disponible para {self.name}")
        final = planned_tool or chosen
        self.last_associations.append({"intent": intent, "planned": planned_tool, "associated": chosen,
                                       "method": method, "used": final, "agree": chosen == planned_tool})
        return final

    @staticmethod
    def validate_args(args: dict, schema: dict) -> dict:
        """Validación determinista de argumentos contra el JSON Schema de la herramienta MCP."""
        props, required = schema.get("properties", {}), schema.get("required", [])
        missing = [r for r in required if r not in args]
        if missing:
            raise ValueError(f"Faltan argumentos requeridos: {missing}")
        unknown = [a for a in args if a not in props]
        if unknown:
            raise ValueError(f"Argumentos no definidos en el esquema: {unknown}")
        for key, value in args.items():
            expected = props[key].get("type")
            if expected == "string" and not isinstance(value, str):
                raise ValueError(f"'{key}' debe ser string")
            if expected in ("number", "integer") and not isinstance(value, (int, float)):
                raise ValueError(f"'{key}' debe ser numérico")
        return args

    # ------------------------------------------------------------------ Ejecución
    async def _run(self, task: dict) -> dict:
        state = self._plans[task["id"]]
        plan = state["plan"]
        self._set_state(task, WORKING)
        try:
            async with self.mcp_factory(caller=self.label) as session:
                tools = {t["name"]: t for t in await session.list_tools()}
                if self.allowed_tools is not None:
                    tools = {n: t for n, t in tools.items() if n in self.allowed_tools}
                while state["cursor"] < len(plan["plans"]):
                    idx = state["cursor"]
                    step, item = plan["steps"][idx], plan["plans"][idx]
                    tool = await self.associate_tool(step["intent"], item.get("tool"), tools)
                    args = self.validate_args(dict(item.get("args", {})), tools[tool]["inputSchema"])
                    result = await session.call_tool(tool, args)
                    state["trace"].append({"step": idx + 1, "intent": step["intent"], "tool": tool,
                                           "args": args, "agent": self.name, "result": result})
                    state["cursor"] += 1
                    if self.hitl and tool in self.hitl_after_tools and state["cursor"] < len(plan["plans"]):
                        nxt = plan["plans"][state["cursor"]]
                        self._set_state(task, INPUT_REQUIRED,
                                        text=f"HITL: ¿Aprueba ejecutar '{nxt['tool']}' con {nxt['args']} "
                                             f"usando el resultado anterior? Responda {{'approved': true|false}}",
                                        data={"pending_step": state["cursor"] + 1, "last_result": result})
                        return task
        except Exception as exc:
            self._set_state(task, FAILED, text=f"{type(exc).__name__}: {exc}",
                            data={"trace": state["trace"]})
            return task
        task["artifacts"] = [{"artifactId": str(uuid.uuid4()), "name": "execution-trace",
                              "parts": [{"data": {"trace": state["trace"]}}]}]
        self._set_state(task, COMPLETED)
        return task

    async def execute_plan(self, plan: dict) -> dict:
        self.last_associations = []
        return await self._run(self._new_task(plan))

    async def resume(self, task_id: str, approved: bool) -> dict:
        task = self.tasks[task_id]
        if task["status"]["state"] != INPUT_REQUIRED:
            raise ValueError(f"La tarea {task_id} no espera input (estado {task['status']['state']})")
        if not approved:
            self._set_state(task, CANCELED, text="Operación rechazada por el operador (HITL)")
            return task
        return await self._run(task)


class IPConnectionAgent(SpecializedAgent):
    name = "IPConnectionAgent"
    description = ("Specialized packet-layer agent: computes routes with the PCE and provisions, "
                   "removes or audits IP connections through the SDN controller NBI exposed via MCP.")
    allowed_tools = {"compute_route", "establish_ip_connection", "remove_ip_connection", "get_connection_status"}
    hitl_after_tools = {"compute_route"}
    skills = [
        {"id": "ip-connection-provisioning", "name": "IP connection provisioning",
         "description": "Compute a route and establish an IP connection between two nodes with a bitrate.",
         "tags": ["ip", "sdn", "packet", "tool:compute_route", "tool:establish_ip_connection"],
         "examples": ["Establish an IP connection from H1 to H2 with 1GB bandwidth"],
         "inputModes": ["application/json"], "outputModes": ["application/json"]},
        {"id": "ip-connection-lifecycle", "name": "IP connection lifecycle",
         "description": "Remove IP connections and report their status.",
         "tags": ["ip", "sdn", "tool:remove_ip_connection", "tool:get_connection_status"],
         "examples": ["Remove the IP connection between H1 and H2"],
         "inputModes": ["application/json"], "outputModes": ["application/json"]},
    ]


def create_agent_app(agent: SpecializedAgent):
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI(title=f"A2A agent: {agent.name}")

    @app.get("/.well-known/agent-card.json")
    @app.get("/.well-known/agent.json")
    def agent_card():
        return agent.agent_card()

    @app.post("/execute_plan")
    async def execute_plan(plan: dict):
        if "plans" not in plan or "steps" not in plan:
            raise HTTPException(status_code=400, detail="El plan debe contener 'steps' y 'plans'")
        return await agent.execute_plan(plan)

    @app.post("/a2a")
    async def jsonrpc(request: Request):
        req = await request.json()
        rid, method, params = req.get("id"), req.get("method"), req.get("params", {})
        try:
            if method == "SendMessage":
                msg = params["message"]
                data = next((p["data"] for p in msg.get("parts", []) if "data" in p), {})
                if msg.get("taskId"):
                    result = await agent.resume(msg["taskId"], bool(data.get("approved")))
                else:
                    result = await agent.execute_plan(data)
            elif method == "GetTask":
                result = agent.tasks[params["id"]]
            else:
                return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "Method not found"}}
        except (KeyError, ValueError) as exc:
            return {"jsonrpc": "2.0", "id": rid, "error": {"code": -32602, "message": str(exc)}}
        return {"jsonrpc": "2.0", "id": rid, "result": {"task": result}}

    return app
