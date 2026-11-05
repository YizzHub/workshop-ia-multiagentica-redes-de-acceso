"""
access_topology.py
==================
Única fuente de verdad de la topología del laboratorio (la usan Mininet, Ryu,
el emulador SDN y el inventario/PCE).

    H1 ──[p1] R1 [p2]──[p1] R2 [p2]──[p1] R3 [p2]── H2
              [p3] ════════ bypass óptico ════════ [p3]   (standby)

  * H1: nodo de acceso (ONU/CPE agregada)       10.0.0.1
  * H2: nodo de borde (BNG / Edge Server / MEC)  10.0.0.2
  * R1-R3: switches OpenFlow 1.3 (OVS) de agregación metro-acceso
  * R1-R3 (óptico): lightpath de *bypass* en standby (Ejercicio 1). No se
    instancia en Mininet; solo existe en el inventario y en el emulador.
"""
from __future__ import annotations

TOPOLOGY: dict = {
    "hosts": {
        "H1": {"name": "h1", "ip": "10.0.0.1", "mac": "00:00:00:00:00:01",
               "switch": "R1", "port": 1, "role": "Acceso: ONU/CPE agregada"},
        "H2": {"name": "h2", "ip": "10.0.0.2", "mac": "00:00:00:00:00:02",
               "switch": "R3", "port": 2, "role": "Borde: BNG / Edge Server (MEC)"},
    },
    "switches": {
        "R1": {"name": "r1", "dpid": 1, "role": "Agregación de acceso (OVS)"},
        "R2": {"name": "r2", "dpid": 2, "role": "Agregación metro (OVS)"},
        "R3": {"name": "r3", "dpid": 3, "role": "Borde metro / PoP (OVS)"},
    },
    "links": [
        {"a": "H1", "b": "R1", "a_port": None, "b_port": 1, "layer": "packet",
         "capacity_gbps": 10.0, "used_gbps": 0.0, "delay_ms": 0.05, "state": "up", "in_mininet": True},
        {"a": "R1", "b": "R2", "a_port": 2, "b_port": 1, "layer": "packet",
         "capacity_gbps": 10.0, "used_gbps": 0.0, "delay_ms": 0.50, "state": "up", "in_mininet": True},
        {"a": "R2", "b": "R3", "a_port": 2, "b_port": 1, "layer": "packet",
         "capacity_gbps": 10.0, "used_gbps": 8.0, "delay_ms": 0.50, "state": "up", "in_mininet": True},
        {"a": "R3", "b": "H2", "a_port": 2, "b_port": None, "layer": "packet",
         "capacity_gbps": 10.0, "used_gbps": 0.0, "delay_ms": 0.05, "state": "up", "in_mininet": True},
        {"a": "R1", "b": "R3", "a_port": 3, "b_port": 3, "layer": "optical",
         "capacity_gbps": 100.0, "used_gbps": 0.0, "delay_ms": 0.30, "state": "standby", "in_mininet": False},
    ],
}


def port_towards(node: str, neighbor: str, topology: dict = TOPOLOGY) -> int:
    """Puerto OpenFlow del switch ``node`` que conduce hacia ``neighbor``."""
    for host_name, host in topology["hosts"].items():
        if neighbor == host_name and host["switch"] == node:
            return host["port"]
    for link in topology["links"]:
        if link["a"] == node and link["b"] == neighbor and link["a_port"] is not None:
            return link["a_port"]
        if link["b"] == node and link["a"] == neighbor and link["b_port"] is not None:
            return link["b_port"]
    raise ValueError(f"No existe enlace {node} -> {neighbor} en la topología")


def default_path(src: str, dst: str, topology: dict = TOPOLOGY) -> list[str]:
    """Camino BFS por los enlaces de paquetes activos (respaldo si no llega ruta del PCE)."""
    adj: dict[str, list[str]] = {}
    for link in topology["links"]:
        if link["state"] == "up":
            adj.setdefault(link["a"], []).append(link["b"])
            adj.setdefault(link["b"], []).append(link["a"])
    frontier, parent = [src], {src: None}
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adj.get(node, []):
                if nb not in parent:
                    parent[nb] = node
                    nxt.append(nb)
        frontier = nxt
    if dst not in parent:
        raise ValueError(f"No hay camino entre {src} y {dst}")
    path, cur = [], dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]
