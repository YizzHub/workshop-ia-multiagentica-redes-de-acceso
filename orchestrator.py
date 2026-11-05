"""
orchestrator.py
===============
Agente Orquestador centralizado (Fig. 1 y 2 del paper):

  1. Recibe el prompt del operador            (POST /prompt)
  2. Pide el plan al servicio LLM             (POST /plan)
  3. Selecciona el agente especializado por sus *skills* (Agent Card A2A) y
     le delega el plan -> patrón **Handoff**  (POST /execute_plan)
  4. Devuelve al operador la traza de ejecución  {"trace": [...]}
"""
from typing import Any

import workshop_common as wc

COMPLETED, INPUT_REQUIRED = "TASK_STATE_COMPLETED", "TASK_STATE_INPUT_REQUIRED"


class Orchestrator:
    def __init__(self, llm_url: str):
        self.llm_url = llm_url
        self.agents: dict[str, dict[str, Any]] = {}
        self.pending_tasks: dict[str, str] = {}
        self.last_plan: dict | None = None
        self.last_handoff: str | None = None

    # ------------------------------------------------------------------ Descubrimiento A2A
    def register_agent(self, base_url: str, label: str = "Agent") -> dict:
        card = wc.http_call("Orchestrator", label, "GET", f"{base_url}/.well-known/agent-card.json")
        tools = {t.split(":", 1)[1] for s in card.get("skills", []) for t in s.get("tags", [])
                 if t.startswith("tool:")}
        self.agents[card["name"]] = {"card": card, "base_url": base_url, "label": label, "tools": tools}
        return card

    def select_agent(self, plan: dict) -> dict:
        """Handoff: elige el agente cuyas skills cubren todas las herramientas del plan."""
        needed = {p["tool"] for p in plan["plans"]}
        ranked = sorted(self.agents.values(), key=lambda a: len(needed & a["tools"]), reverse=True)
        if not ranked or not needed <= ranked[0]["tools"]:
            raise LookupError(f"Ningún agente registrado soporta las herramientas {sorted(needed)}")
        return ranked[0]

    # ------------------------------------------------------------------ Flujo principal
    def _finish(self, task: dict, agent: dict) -> dict:
        state = task["status"]["state"]
        if state == COMPLETED:
            self.pending_tasks.pop(task["id"], None)
            for artifact in task.get("artifacts", []):
                for part in artifact.get("parts", []):
                    if "trace" in part.get("data", {}):
                        return {"trace": part["data"]["trace"]}
            return {"trace": []}
        if state == INPUT_REQUIRED:
            self.pending_tasks[task["id"]] = agent["card"]["name"]
            parts = task["status"]["message"]["parts"]
            return {"status": state, "task_id": task["id"], "agent": agent["card"]["name"],
                    "question": next((p["text"] for p in parts if "text" in p), ""),
                    "details": next((p["data"] for p in parts if "data" in p), {})}
        msg = task["status"].get("message", {}).get("parts", [{}])[0].get("text", "")
        raise RuntimeError(f"El agente {agent['card']['name']} terminó en {state}: {msg}")

    def handle_prompt(self, prompt: str) -> dict:
        plan = wc.http_call("Orchestrator", "LLM", "POST", f"{self.llm_url}/plan", {"prompt": prompt})
        self.last_plan = plan
        agent = self.select_agent(plan)
        self.last_handoff = agent["card"]["name"]
        task = wc.http_call("Orchestrator", agent["label"], "POST", f"{agent['base_url']}/execute_plan", plan)
        return self._finish(task, agent)

    def approve(self, task_id: str, approved: bool) -> dict:
        agent = self.agents[self.pending_tasks[task_id]]
        rpc = {"jsonrpc": "2.0", "id": 1, "method": "SendMessage",
               "params": {"message": {"messageId": task_id + "-hitl", "role": "ROLE_USER", "taskId": task_id,
                                      "parts": [{"data": {"approved": approved}}]}}}
        resp = wc.http_call("Orchestrator", agent["label"], "POST", f"{agent['base_url']}/a2a", rpc)
        if "error" in resp:
            raise RuntimeError(resp["error"]["message"])
        task = resp["result"]["task"]
        if task["status"]["state"] == "TASK_STATE_CANCELED":
            self.pending_tasks.pop(task_id, None)
            return {"status": "TASK_STATE_CANCELED", "task_id": task_id}
        return self._finish(task, agent)


def create_orchestrator_app(orch: Orchestrator):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class PromptRequest(BaseModel):
        prompt: str

    class ApprovalRequest(BaseModel):
        task_id: str
        approved: bool

    app = FastAPI(title="Centralized Orchestrator")

    @app.post("/prompt")
    def prompt(req: PromptRequest):
        try:
            return orch.handle_prompt(req.prompt)
        except wc.HTTPCallError as exc:
            raise HTTPException(status_code=502, detail=exc.detail)
        except (LookupError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.post("/approve")
    def approve(req: ApprovalRequest):
        try:
            return orch.approve(req.task_id, req.approved)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Tarea {req.task_id} no pendiente")
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    @app.get("/agents")
    def agents():
        return {name: {"url": a["base_url"], "tools": sorted(a["tools"])} for name, a in orch.agents.items()}

    return app
