"""
sdn_emulator.py
===============
Simulador de respaldo *transparente* para ``SIMULATION_MODE = True``.

Reproduce, en el propio proceso de Jupyter y sin privilegios root:
  * Switches OVS con tablas de flujo OpenFlow 1.3 (match/prioridad/acciones/cookie).
  * Un plano de datos que reenvía ARP e ICMP salto a salto según esas tablas.
  * ``EmulatedRyuController``: MISMA lógica que ``ryu_controller.py`` (switch L2
    de aprendizaje + aislamiento H1<->H2 + servicio IP), registrando cada
    OFPT_FLOW_MOD / OFPT_PACKET_IN en la captura del workshop.
  * ``create_sdn_app``: MISMA NBI REST que Ryu (/service/ip/create|status|delete).
"""
import random
import threading
from typing import Any

from access_topology import TOPOLOGY, default_path, port_towards
import workshop_common as wc

BROADCAST = "ff:ff:ff:ff:ff:ff"
ETH_ARP, ETH_IP = 0x0806, 0x0800
LEARNING_COOKIE = 0x1000
SERVICE_COOKIE_BASE = 0xA000


# =============================================================================
# Plano de datos emulado (OVS)
# =============================================================================
class FlowEntry:
    def __init__(self, priority: int, match: dict, actions: list, cookie: int = 0):
        self.priority, self.match, self.actions, self.cookie = priority, match, actions, cookie
        self.n_packets = 0

    def matches(self, pkt: dict, in_port: int) -> bool:
        for key, value in self.match.items():
            actual = in_port if key == "in_port" else pkt.get(key)
            if actual != value:
                return False
        return True

    def __repr__(self) -> str:
        acts = ",".join(f"{a[0]}:{a[1]}" if len(a) > 1 else a[0] for a in self.actions) or "drop"
        match = ",".join(f"{k}={v:#06x}" if k == "eth_type" else f"{k}={v}" for k, v in self.match.items())
        return (f"cookie={self.cookie:#x}, priority={self.priority}, n_packets={self.n_packets}, "
                f"{match or '(any)'} actions={acts}")


class EmulatedSwitch:
    def __init__(self, sid: str, name: str, dpid: int):
        self.sid, self.name, self.dpid = sid, name, dpid
        self.ports: dict[int, tuple[str, int | None]] = {}
        self.flow_table: list[FlowEntry] = []

    def flow_mod(self, command: str, priority: int = 0, match: dict | None = None,
                 actions: list | None = None, cookie: int = 0) -> None:
        if command == "ADD":
            self.flow_table = [f for f in self.flow_table
                               if not (f.priority == priority and f.match == (match or {}))]
            self.flow_table.append(FlowEntry(priority, match or {}, actions or [], cookie))
            self.flow_table.sort(key=lambda f: -f.priority)
        elif command == "DELETE":
            self.flow_table = [f for f in self.flow_table if f.cookie != cookie]

    def lookup(self, pkt: dict, in_port: int) -> FlowEntry | None:
        for entry in self.flow_table:
            if entry.matches(pkt, in_port):
                entry.n_packets += 1
                return entry
        return None

    def dump_flows(self) -> str:
        """Equivalente a: ovs-ofctl -O OpenFlow13 dump-flows <switch>"""
        return "\n".join(f" {f}" for f in self.flow_table) or " (tabla vacía)"


class EmulatedNetwork:
    """Plano de datos: hosts + switches cableados según access_topology.py."""

    def __init__(self, topology: dict = TOPOLOGY):
        self.topology = topology
        self.hosts = topology["hosts"]
        self.switches = {sid: EmulatedSwitch(sid, s["name"], s["dpid"])
                         for sid, s in topology["switches"].items()}
        self.controller: EmulatedRyuController | None = None
        self._lock = threading.RLock()
        self.arp_cache: dict[str, dict[str, str]] = {h: {} for h in self.hosts}
        for link in topology["links"]:
            a, b = link["a"], link["b"]
            if a in self.switches and link["a_port"] is not None:
                self.switches[a].ports[link["a_port"]] = (b, link["b_port"])
            if b in self.switches and link["b_port"] is not None:
                self.switches[b].ports[link["b_port"]] = (a, link["a_port"])
        self._link_state = {frozenset((l["a"], l["b"])): l for l in topology["links"]}

    def attach_controller(self, controller: "EmulatedRyuController") -> None:
        self.controller = controller
        for sw in self.switches.values():
            controller.switch_connected(sw)

    def _port_up(self, sw: EmulatedSwitch, port: int) -> bool:
        peer = sw.ports[port][0]
        return self._link_state[frozenset((sw.sid, peer))]["state"] == "up"

    # --------------------------------------------------------------- reenvío
    def _process(self, sw: EmulatedSwitch, in_port: int, pkt: dict, ctx: dict) -> None:
        ctx["hops"] += 1
        if ctx["hops"] > 64:
            return
        entry = sw.lookup(pkt, in_port)
        if entry is None:
            return
        for action in entry.actions:
            if action[0] == "CONTROLLER":
                ctx["packet_ins"] += 1
                for out_port in self.controller.packet_in(sw, in_port, pkt):
                    self._emit(sw, out_port, in_port, pkt, ctx)
            elif action[0] == "OUTPUT":
                self._emit(sw, action[1], in_port, pkt, ctx)

    def _emit(self, sw: EmulatedSwitch, out_port: Any, in_port: int, pkt: dict, ctx: dict) -> None:
        ports = ([p for p in sw.ports if p != in_port and self._port_up(sw, p)]
                 if out_port == "FLOOD" else [out_port])
        for port in ports:
            if port not in sw.ports or (sw.sid, port) in ctx["visited"]:
                continue
            ctx["visited"].add((sw.sid, port))
            peer, peer_port = sw.ports[port]
            if peer in self.hosts:
                if pkt["eth_dst"] in (BROADCAST, self.hosts[peer]["mac"]):
                    ctx["delivered"].append(peer)
            else:
                self._process(self.switches[peer], peer_port, pkt, ctx)

    def send(self, host: str, pkt: dict) -> dict:
        h = self.hosts[host]
        ctx = {"hops": 0, "packet_ins": 0, "delivered": [], "visited": set()}
        self._process(self.switches[h["switch"]], h["port"], pkt, ctx)
        return ctx

    # --------------------------------------------------------------- ping
    def _arp_resolve(self, src: str, dst: str) -> tuple[bool, dict]:
        s, d = self.hosts[src], self.hosts[dst]
        req = {"eth_src": s["mac"], "eth_dst": BROADCAST, "eth_type": ETH_ARP,
               "arp_spa": s["ip"], "arp_tpa": d["ip"], "arp_op": 1}
        ctx = self.send(src, req)
        if dst not in ctx["delivered"]:
            return False, ctx
        rep = {"eth_src": d["mac"], "eth_dst": s["mac"], "eth_type": ETH_ARP,
               "arp_spa": d["ip"], "arp_tpa": s["ip"], "arp_op": 2}
        ctx2 = self.send(dst, rep)
        ctx["packet_ins"] += ctx2["packet_ins"]
        if src not in ctx2["delivered"]:
            return False, ctx
        self.arp_cache[src][d["ip"]] = d["mac"]
        self.arp_cache[dst][s["ip"]] = s["mac"]
        return True, ctx

    def ping(self, src: str = "H1", dst: str = "H2", count: int = 3) -> str:
        """Emula ``mininet> h1 ping h2`` con salida idéntica en formato a iputils-ping."""
        with self._lock:
            s, d = self.hosts[src], self.hosts[dst]
            lines = [f"PING {d['ip']} ({d['ip']}) 56(84) bytes of data."]
            received, errors, rtts = 0, 0, []
            for seq in range(1, count + 1):
                extra_ms = 0.0
                if d["ip"] not in self.arp_cache[src]:
                    ok, ctx = self._arp_resolve(src, dst)
                    if not ok:
                        errors += 1
                        lines.append(f"From {s['ip']} icmp_seq={seq} Destination Host Unreachable")
                        continue
                    extra_ms += random.uniform(8.0, 11.0) + 3.0 * ctx["packet_ins"]
                echo = {"eth_src": s["mac"], "eth_dst": d["mac"], "eth_type": ETH_IP,
                        "ipv4_src": s["ip"], "ipv4_dst": d["ip"], "icmp_type": 8}
                ctx = self.send(src, echo)
                if dst not in ctx["delivered"]:
                    self.arp_cache[src].pop(d["ip"], None)
                    continue
                reply = {"eth_src": d["mac"], "eth_dst": s["mac"], "eth_type": ETH_IP,
                         "ipv4_src": d["ip"], "ipv4_dst": s["ip"], "icmp_type": 0}
                ctx2 = self.send(dst, reply)
                if src not in ctx2["delivered"]:
                    continue
                base = 0.25 if seq == 2 else 0.07
                rtt = extra_ms + base + 0.004 * (ctx["hops"] + ctx2["hops"]) + random.uniform(0, 0.04)
                rtt += 3.0 * (ctx["packet_ins"] + ctx2["packet_ins"])
                rtts.append(rtt)
                received += 1
                rtt_txt = f"{rtt:.1f}" if rtt >= 10 else (f"{rtt:.2f}" if rtt >= 1 else f"{rtt:.3f}")
                lines.append(f"64 bytes from {d['ip']}: icmp_seq={seq} ttl=64 time={rtt_txt} ms")
            loss = 100 * (count - received) // count
            lines += ["", f"--- {d['ip']} ping statistics ---"]
            err_txt = f", +{errors} errors" if errors else ""
            lines.append(f"{count} packets transmitted, {received} received{err_txt}, "
                         f"{loss}% packet loss, time {1000 * (count - 1) + 3}ms")
            if rtts:
                avg = sum(rtts) / len(rtts)
                lines.append(f"rtt min/avg/max/mdev = {min(rtts):.3f}/{avg:.3f}/{max(rtts):.3f}/"
                             f"{(max(rtts) - min(rtts)) / 2:.3f} ms")
            # caché ARP con vida corta: cada ping vuelve a validar la conectividad
            self.arp_cache = {h: {} for h in self.hosts}
            return "\n".join(lines)

    def dump_flows(self, sid: str) -> str:
        sw = self.switches[sid]
        return f"$ ovs-ofctl -O OpenFlow13 dump-flows {sw.name}\n{sw.dump_flows()}"


# =============================================================================
# Controlador emulado: misma lógica que ryu_controller.py
# =============================================================================
class EmulatedRyuController:
    def __init__(self, network: EmulatedNetwork, label: str = "SDN"):
        self.net = network
        self.label = label
        self.mac_to_port: dict[int, dict[str, int]] = {}
        self.connections: dict[str, dict] = {}
        self._next_cookie = SERVICE_COOKIE_BASE
        h = network.hosts
        self.isolated_pairs = {frozenset((h["H1"]["ip"], h["H2"]["ip"]))}
        network.attach_controller(self)

    def _flow_mod(self, sw: EmulatedSwitch, command: str, priority: int = 0, match: dict | None = None,
                  actions: list | None = None, cookie: int = 0, log: bool = True) -> None:
        sw.flow_mod(command, priority, match, actions, cookie)
        if log:
            if command == "ADD":
                desc = ", ".join(f"{k}={v:#06x}" if k == "eth_type" else f"{k}={v}"
                                 for k, v in (match or {}).items()) or "table-miss"
                acts = ",".join(f"{a[0]}:{a[1]}" if len(a) > 1 else a[0] for a in (actions or []))
                info = f"Type: OFPT_FLOW_MOD (ADD prio={priority} {desc} -> {acts})"
            else:
                info = f"Type: OFPT_FLOW_MOD (DELETE cookie={cookie:#x})"
            wc.CAPTURE.record(self.label, f"OVS {sw.name}", "OpenFlow", info, length=80 + 16 * len(match or {}))

    def switch_connected(self, sw: EmulatedSwitch) -> None:
        self._flow_mod(sw, "ADD", 0, {}, [("CONTROLLER",)], log=False)

    @staticmethod
    def _ip_pair(pkt: dict) -> tuple[str, str] | None:
        if pkt.get("eth_type") == ETH_ARP:
            return pkt["arp_spa"], pkt["arp_tpa"]
        if pkt.get("eth_type") == ETH_IP:
            return pkt["ipv4_src"], pkt["ipv4_dst"]
        return None

    def packet_in(self, sw: EmulatedSwitch, in_port: int, pkt: dict) -> list:
        wc.CAPTURE.record(f"OVS {sw.name}", self.label, "OpenFlow",
                          f"Type: OFPT_PACKET_IN (in_port={in_port}, eth_type={pkt['eth_type']:#06x})")
        self.mac_to_port.setdefault(sw.dpid, {})[pkt["eth_src"]] = in_port
        pair = self._ip_pair(pkt)
        if pair and frozenset(pair) in self.isolated_pairs:
            return []                                   # AISLAMIENTO: descartado
        out = self.mac_to_port[sw.dpid].get(pkt["eth_dst"], "FLOOD")
        if out != "FLOOD":
            self._flow_mod(sw, "ADD", 1, {"in_port": in_port, "eth_dst": pkt["eth_dst"],
                                          "eth_src": pkt["eth_src"]}, [("OUTPUT", out)], LEARNING_COOKIE)
        return [out]

    # --------------------------------------------------------------- NBI
    def establish_connection(self, src_node: str, dst_node: str, bitrate_gbps: float,
                             path: list[str] | None = None) -> dict:
        hosts = self.net.hosts
        if src_node not in hosts or dst_node not in hosts:
            raise ValueError(f"Nodos desconocidos: {src_node}, {dst_node}")
        connection_id = f"{src_node}-{dst_node}"
        if connection_id in self.connections:
            self.remove_connection(connection_id)
        path = path or default_path(src_node, dst_node, self.net.topology)
        src_ip, dst_ip = hosts[src_node]["ip"], hosts[dst_node]["ip"]
        cookie = self._next_cookie
        self._next_cookie += 1
        flow_mods = 0
        for i in range(1, len(path) - 1):
            sw = self.net.switches[path[i]]
            fwd = port_towards(path[i], path[i + 1], self.net.topology)
            rev = port_towards(path[i], path[i - 1], self.net.topology)
            for s_ip, d_ip, port in ((src_ip, dst_ip, fwd), (dst_ip, src_ip, rev)):
                self._flow_mod(sw, "ADD", 100, {"eth_type": ETH_IP, "ipv4_src": s_ip, "ipv4_dst": d_ip},
                               [("OUTPUT", port)], cookie)
                self._flow_mod(sw, "ADD", 100, {"eth_type": ETH_ARP, "arp_spa": s_ip, "arp_tpa": d_ip},
                               [("OUTPUT", port)], cookie)
                flow_mods += 2
        self.isolated_pairs.discard(frozenset((src_ip, dst_ip)))
        self.connections[connection_id] = {"src_node": src_node, "dst_node": dst_node,
                                           "bitrate_gbps": bitrate_gbps, "path": path,
                                           "cookie": cookie, "flow_mods": flow_mods, "status": "UP"}
        return {"connection_id": connection_id, "status": "UP",
                "message": f"Connection {connection_id} established"}

    def remove_connection(self, connection_id: str) -> dict:
        conn = self.connections.pop(connection_id, None)
        if conn is None:
            raise KeyError(connection_id)
        for sw in self.net.switches.values():
            self._flow_mod(sw, "DELETE", cookie=conn["cookie"])
            sw.flow_mod("DELETE", cookie=LEARNING_COOKIE)
        hosts = self.net.hosts
        self.isolated_pairs.add(frozenset((hosts[conn["src_node"]]["ip"], hosts[conn["dst_node"]]["ip"])))
        return {"connection_id": connection_id, "status": "DOWN",
                "message": f"Connection {connection_id} removed"}

    def status(self) -> dict:
        return {
            "connections": {cid: {k: v for k, v in c.items() if k != "cookie"}
                            for cid, c in self.connections.items()},
            "isolated_pairs": [sorted(p) for p in self.isolated_pairs],
            "switches": sorted(sw.dpid for sw in self.net.switches.values()),
        }


def create_sdn_app(controller: EmulatedRyuController):
    """NBI REST idéntica a la de ryu_controller.py, servida con FastAPI."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    class CreateRequest(BaseModel):
        src_node: str
        dst_node: str
        bitrate_gbps: float = 0.0
        path: list[str] | None = None

    app = FastAPI(title="Ryu SDN NBI (emulada)")

    @app.post("/service/ip/create")
    def create(req: CreateRequest):
        try:
            with controller.net._lock:
                return controller.establish_connection(req.src_node, req.dst_node, req.bitrate_gbps, req.path)
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.get("/service/ip/status")
    def status():
        return controller.status()

    @app.delete("/service/ip/delete")
    def delete(connection_id: str):
        try:
            with controller.net._lock:
                return controller.remove_connection(connection_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Conexión {connection_id} no existe")

    return app
