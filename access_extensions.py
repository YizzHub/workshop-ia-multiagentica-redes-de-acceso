"""
access_extensions.py
====================
Extensiones de la arquitectura hacia las REDES DE ACCESO (precargadas para la
exploración interactiva). Siguen exactamente el mismo patrón del paper:
herramienta MCP determinista + workflow en ChromaDB + agente A2A.

  * Ref. [2] · P2MP con subportadoras digitales (DSCM):
      - herramienta MCP  allocate_subcarriers(leaf_node, num_sc)
      - nuevo agente A2A OpticalP2MPAgent  (el Orquestador le hace Handoff)
  * Ref. [3] · Gemelo digital (OCATA simplificado):
      - herramienta MCP  get_e2e_telemetry(src_node, dst_node)
      - workflow «diagnose_latency_degradation»: telemetría -> ruta -> re-provisión
"""
import networkx as nx
import requests

import a2a_agents
import workshop_common as wc

HUB_SUBCARRIERS = 16    # el Hub recibe 16 subportadoras digitales
SC_RATE_GBPS = 25       # 25 Gb/s por subportadora (16 x 25 = 400 Gb/s)
LEAF_MAX_SC = 4         # cada hoja modula como máximo 4 subportadoras (100 Gb/s)
P2MP_LEAVES = ("ONU1", "ONU2", "ONU3", "ONU4", "ONU5")


class P2MPSubcarrierManager:
    """Asignación First-Fit de subportadoras CONTIGUAS sin solapamiento espectral."""

    def __init__(self, leaves=P2MP_LEAVES):
        self.leaves = set(leaves)
        self.slots = [None] * HUB_SUBCARRIERS

    def allocate_subcarriers(self, leaf_node: str, num_sc: int) -> dict:
        if leaf_node not in self.leaves:
            raise ValueError(f"{leaf_node} no pertenece al árbol P2MP")
        if not 1 <= num_sc <= LEAF_MAX_SC:
            raise ValueError(f"num_sc debe estar entre 1 y {LEAF_MAX_SC}")
        previous = list(self.slots)
        self.release(leaf_node)
        for start in range(HUB_SUBCARRIERS - num_sc + 1):
            block = range(start, start + num_sc)
            if all(self.slots[i] is None for i in block):
                for i in block:
                    self.slots[i] = leaf_node
                used = sum(s is not None for s in self.slots)
                return {"leaf_node": leaf_node, "subcarriers": list(block),
                        "capacity_gbps": num_sc * SC_RATE_GBPS, "hub_utilization": used / HUB_SUBCARRIERS}
        self.slots = previous
        raise ValueError(f"No hay {num_sc} subportadoras contiguas libres en el Hub")

    def release(self, leaf_node: str) -> None:
        self.slots = [None if s == leaf_node else s for s in self.slots]

    def spectrum_map(self) -> str:
        return "|" + "".join("·" if s is None else s[-1] for s in self.slots) + "|"


class OpticalP2MPAgent(a2a_agents.SpecializedAgent):
    name = "OpticalP2MPAgent"
    description = ("Specialized optical-layer agent for point-to-multipoint (P2MP) access trees based on "
                   "digital subcarrier multiplexing (DSCM): allocates 25 Gb/s subcarriers to leaf nodes.")
    allowed_tools = {"allocate_subcarriers"}
    skills = [
        {"id": "p2mp-subcarrier-allocation", "name": "P2MP subcarrier allocation",
         "description": "Allocate contiguous digital subcarriers to a leaf node without spectral overlap.",
         "tags": ["optical", "p2mp", "dscm", "access", "tool:allocate_subcarriers"],
         "examples": ["Allocate 2 subcarriers to leaf ONU1 in the P2MP access tree"],
         "inputModes": ["application/json"], "outputModes": ["application/json"]},
    ]


class OCATADigitalTwinMock:
    """Gemelo digital simplificado: estima latencia e2e y BER pre-FEC de la conexión provisionada."""

    BER_FEC_THRESHOLD = 4e-3

    def __init__(self, inventory, d_max_ms=1.5):
        self.inventory, self.d_max_ms = inventory, d_max_ms
        self.impairments = {}

    def inject_degradation(self, u, v, extra_ms=2.0, ber=6e-3):
        self.impairments[frozenset((u, v))] = {"extra_ms": extra_ms, "ber": ber}
        self.inventory.set_link_state(u, v, "degraded")      # alarma -> el PCE evitará el enlace

    def clear(self):
        for key in list(self.impairments):
            u, v = tuple(key)
            self.inventory.set_link_state(u, v, "up")
        self.impairments.clear()

    def _provisioned_path(self, src, dst):
        conns = requests.get(f"{wc.url('sdn')}/service/ip/status", timeout=10).json().get("connections", {})
        if f"{src}-{dst}" in conns:
            return conns[f"{src}-{dst}"]["path"]
        return nx.shortest_path(self.inventory.graph, src, dst)

    def estimate(self, src, dst):
        path = self._provisioned_path(src, dst)
        latency, ber = 0.0, 1e-6
        for u, v in zip(path, path[1:]):
            imp = self.impairments.get(frozenset((u, v)), {})
            latency += self.inventory.graph.edges[u, v]["delay_ms"] + imp.get("extra_ms", 0.0)
            ber = max(ber, imp.get("ber", 1e-6))
        return {"src": src, "dst": dst, "path": path, "latency_ms": round(latency, 3),
                "d_max_ms": self.d_max_ms, "pre_fec_ber": ber,
                "qot_ok": ber < self.BER_FEC_THRESHOLD, "degraded": latency > self.d_max_ms}


DIAGNOSE_WORKFLOW = {
    "name": "diagnose_latency_degradation",
    "description": "Diagnose latency degradation (delay, QoT, BER) of an existing connection between two nodes "
                   "using telemetry and the digital twin, and reconfigure it over a new route.",
    "defaults": {"bitrate_gbps": 1},
    "steps": [
        {"intent": "retrieve end-to-end latency and QoT telemetry from the digital twin to diagnose degradation",
         "tool": "get_e2e_telemetry", "args": ["src_node", "dst_node"]},
        {"intent": "compute the shortest route between source and destination nodes for a candidate connection",
         "tool": "compute_route", "args": ["src_node", "dst_node"]},
        {"intent": "provision and bring up a new IP connection on the SDN-controlled network",
         "tool": "establish_ip_connection", "args": ["src_node", "dst_node", "bitrate_gbps"]},
    ],
}

P2MP_WORKFLOW = {
    "name": "allocate_p2mp_subcarriers",
    "description": "Allocate or assign digital subcarriers (capacity) to a leaf node (ONU) of the optical "
                   "point-to-multipoint P2MP access tree.",
    "defaults": {},
    "steps": [{"intent": "allocate contiguous digital subcarriers to a leaf node in the P2MP access tree",
               "tool": "allocate_subcarriers", "args": ["leaf_node", "num_sc"]}],
}


def _register_tool(mcp, fn):
    try:
        mcp.remove_tool(fn.__name__)      # re-ejecución idempotente de la celda
    except Exception:
        pass
    mcp.tool(fn)


def install(*, mcp, kb, planner, orchestrator, ip_agent, inventory, llm=None):
    """Registra herramientas MCP, workflows, el agente óptico y el gemelo digital."""
    p2mp = P2MPSubcarrierManager()
    twin = OCATADigitalTwinMock(inventory)

    def allocate_subcarriers(leaf_node: str, num_sc: int) -> dict:
        """Allocate contiguous 25 Gb/s digital subcarriers (DSCM) to a leaf node of the optical
        point-to-multipoint (P2MP) access tree without spectral overlap."""
        return p2mp.allocate_subcarriers(leaf_node, num_sc)

    def get_e2e_telemetry(src_node: str, dst_node: str) -> dict:
        """Get end-to-end telemetry of a provisioned connection from the OCATA digital twin:
        latency versus d_max, pre-FEC BER and QoT, to diagnose latency degradation."""
        return twin.estimate(src_node, dst_node)

    _register_tool(mcp, allocate_subcarriers)
    _register_tool(mcp, get_e2e_telemetry)
    kb.add_workflow(P2MP_WORKFLOW)
    kb.add_workflow(DIAGNOSE_WORKFLOW)
    planner.known_nodes |= set(P2MP_LEAVES)

    optical_agent = OpticalP2MPAgent(llm=llm, label="OpticalAgent")
    optical_agent.base_url = wc.start_server("optical_agent", a2a_agents.create_agent_app(optical_agent),
                                             "optical_agent")
    orchestrator.register_agent(optical_agent.base_url, label="OpticalAgent")

    ip_agent.allowed_tools = set(ip_agent.allowed_tools) | {"get_e2e_telemetry"}
    if not any(s["id"] == "latency-diagnosis" for s in ip_agent.skills):
        ip_agent.skills = list(ip_agent.skills) + [{
            "id": "latency-diagnosis", "name": "Latency degradation diagnosis",
            "description": "Diagnose latency/QoT degradation with digital-twin telemetry and reconfigure the connection.",
            "tags": ["telemetry", "digital-twin", "tool:get_e2e_telemetry", "tool:compute_route",
                     "tool:establish_ip_connection"],
            "examples": ["Diagnose the latency degradation between H1 and H2 and reconfigure the connection"],
            "inputModes": ["application/json"], "outputModes": ["application/json"]}]
    orchestrator.register_agent(ip_agent.base_url, label="Agent")
    return {"p2mp": p2mp, "optical_agent": optical_agent, "twin": twin}
