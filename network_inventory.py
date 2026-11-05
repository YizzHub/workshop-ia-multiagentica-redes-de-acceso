"""
network_inventory.py
====================
Herramientas DETERMINISTAS de red usadas por el agente vía MCP:

  * ``NetworkInventory``: inventario/estado de red sobre un grafo NetworkX
    (nodos H1, R1, R2, R3, H2; capacidad, uso, capa y estado de cada enlace).
  * ``PathComputationElement`` (PCE): cálculo de ruta más corta (en saltos)
    sobre los enlaces operativos. Misma entrada => misma salida, siempre.
  * ``create_pce_app``: expone el PCE como servicio REST (POST /compute_route).
"""
import copy

import networkx as nx

from access_topology import TOPOLOGY


class RouteNotFoundError(Exception):
    pass


class NetworkInventory:
    def __init__(self, topology: dict = TOPOLOGY):
        self.topology = copy.deepcopy(topology)
        self.graph = nx.Graph()
        for hid, h in self.topology["hosts"].items():
            self.graph.add_node(hid, kind="host", **h)
        for sid, s in self.topology["switches"].items():
            self.graph.add_node(sid, kind="switch", **s)
        for link in self.topology["links"]:
            attrs = {k: v for k, v in link.items() if k not in ("a", "b")}
            self.graph.add_edge(link["a"], link["b"], **attrs)

    def residual_gbps(self, u: str, v: str) -> float:
        e = self.graph.edges[u, v]
        return e["capacity_gbps"] - e["used_gbps"]

    def set_link_state(self, u: str, v: str, state: str) -> None:
        self.graph.edges[u, v]["state"] = state

    def operational_graph(self) -> nx.Graph:
        """Subgrafo con los enlaces en estado 'up'."""
        edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d["state"] == "up"]
        return self.graph.edge_subgraph(edges).copy()

    def reserve(self, path: list[str], bitrate_gbps: float) -> None:
        for u, v in zip(path, path[1:]):
            self.graph.edges[u, v]["used_gbps"] += bitrate_gbps

    def links_table(self) -> list[dict]:
        rows = []
        for u, v, d in self.graph.edges(data=True):
            rows.append({"link": f"{u}-{v}", "layer": d["layer"], "state": d["state"],
                         "capacity_gbps": d["capacity_gbps"], "used_gbps": d["used_gbps"],
                         "residual_gbps": d["capacity_gbps"] - d["used_gbps"]})
        return rows


class PathComputationElement:
    """PCE determinista: ruta de mínimo número de saltos sobre la red operativa."""

    def __init__(self, inventory: NetworkInventory):
        self.inventory = inventory

    def compute_route(self, src: str, dst: str) -> dict:
        graph = self.inventory.operational_graph()
        for node in (src, dst):
            if node not in self.inventory.graph:
                raise RouteNotFoundError(f"Nodo desconocido: {node}")
        try:
            path = nx.shortest_path(graph, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
            raise RouteNotFoundError(f"No existe ruta {src}->{dst}") from exc
        return {"src": src, "dst": dst, "path": path, "hops": len(path) - 1}


def create_pce_app(pce: PathComputationElement):
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class RouteRequest(BaseModel):
        src_node: str
        dst_node: str

    app = FastAPI(title="Path Computation Element (PCE)")

    @app.post("/compute_route")
    def compute_route(req: RouteRequest):
        try:
            return pce.compute_route(req.src_node, req.dst_node)
        except RouteNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/topology")
    def topology():
        return {"nodes": list(pce.inventory.graph.nodes), "links": pce.inventory.links_table()}

    return app
