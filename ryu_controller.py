"""
ryu_controller.py  (a.k.a. ryu_isolation_controller.py en el instructivo)
=========================================================================
Aplicación Ryu (OpenFlow 1.3) usada en la Sección 3 del paper:

  1. Funciona como *switch* de aprendizaje L2 estándar (reglas MAC -> puerto).
  2. POR DEFECTO AÍSLA el tráfico entre H1 y H2: el ARP/IPv4 entre ambos se
     descarta en el controlador, por lo que ``h1 ping h2`` devuelve
     "Destination Host Unreachable".
  3. Expone una API REST (ryu.app.wsgi) que actúa como NBI del controlador:
        POST   /service/ip/create   -> levanta el aislamiento e instala flujos
        GET    /service/ip/status   -> estado de conexiones y aislamiento
        DELETE /service/ip/delete   -> elimina flujos y restablece el aislamiento

Ejecución (Linux, Ryu instalado):
    ryu-manager ryu_controller.py --ofp-tcp-listen-port 6653 --wsapi-port 8080
"""
import json

from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, DEAD_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib.packet import arp, ether_types, ethernet, ipv4, packet
from ryu.ofproto import ofproto_v1_3
from webob import Response

from access_topology import TOPOLOGY, default_path, port_towards

API_INSTANCE = "ip_service_api_app"
LEARNING_COOKIE = 0x1000
SERVICE_COOKIE_BASE = 0xA000
SERVICE_PRIORITY = 100
LEARNING_PRIORITY = 1


class IsolationController(app_manager.RyuApp):
    """Switch L2 con aislamiento H1<->H2 por defecto + NBI REST."""

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS = {"wsgi": WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.datapaths = {}
        self.connections = {}
        self._next_cookie = SERVICE_COOKIE_BASE
        h1, h2 = TOPOLOGY["hosts"]["H1"]["ip"], TOPOLOGY["hosts"]["H2"]["ip"]
        self.isolated_pairs = {frozenset((h1, h2))}
        kwargs["wsgi"].register(IPServiceRestAPI, {API_INSTANCE: self})

    # ------------------------------------------------------------------ OpenFlow
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp = ev.msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        self.datapaths[dp.id] = dp
        # Regla table-miss: todo paquete desconocido sube al controlador (PACKET_IN)
        actions = [parser.OFPActionOutput(ofp.OFPP_CONTROLLER, ofp.OFPCML_NO_BUFFER)]
        self.add_flow(dp, 0, parser.OFPMatch(), actions)
        self.logger.info("Switch dpid=%s conectado (OpenFlow 1.3)", dp.id)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            self.datapaths[dp.id] = dp
        elif ev.state == DEAD_DISPATCHER:
            self.datapaths.pop(dp.id, None)

    def add_flow(self, dp, priority, match, actions, cookie=0, idle_timeout=0, buffer_id=None):
        """Envía un OFPT_FLOW_MOD (ADD) al switch."""
        ofp, parser = dp.ofproto, dp.ofproto_parser
        inst = [parser.OFPInstructionActions(ofp.OFPIT_APPLY_ACTIONS, actions)]
        kwargs = dict(datapath=dp, cookie=cookie, priority=priority, match=match,
                      instructions=inst, idle_timeout=idle_timeout)
        if buffer_id is not None and buffer_id != ofp.OFP_NO_BUFFER:
            kwargs["buffer_id"] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kwargs))

    def delete_flows(self, dp, cookie):
        """Envía un OFPT_FLOW_MOD (DELETE) filtrando por cookie."""
        ofp, parser = dp.ofproto, dp.ofproto_parser
        dp.send_msg(parser.OFPFlowMod(
            datapath=dp, cookie=cookie, cookie_mask=0xFFFFFFFFFFFFFFFF,
            table_id=ofp.OFPTT_ALL, command=ofp.OFPFC_DELETE,
            out_port=ofp.OFPP_ANY, out_group=ofp.OFPG_ANY))

    @staticmethod
    def _ip_pair(pkt):
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt is not None:
            return arp_pkt.src_ip, arp_pkt.dst_ip
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt is not None:
            return ip_pkt.src, ip_pkt.dst
        return None

    def _is_isolated(self, pkt):
        pair = self._ip_pair(pkt)
        return pair is not None and frozenset(pair) in self.isolated_pairs

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp, parser = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        if eth.ethertype in (ether_types.ETH_TYPE_LLDP, ether_types.ETH_TYPE_IPV6):
            return

        self.mac_to_port.setdefault(dp.id, {})[eth.src] = in_port

        # --- Política de aislamiento: H1 <-> H2 bloqueado hasta que la NBI lo autorice
        if self._is_isolated(pkt):
            self.logger.info("AISLAMIENTO: descartado %s -> %s en dpid=%s", *self._ip_pair(pkt), dp.id)
            return

        # --- Switch de aprendizaje L2 estándar
        out_port = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]
        if out_port != ofp.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst, eth_src=eth.src)
            self.add_flow(dp, LEARNING_PRIORITY, match, actions, cookie=LEARNING_COOKIE, idle_timeout=60)
        data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                        in_port=in_port, actions=actions, data=data))

    # ------------------------------------------------------------------ Servicio IP (NBI)
    def establish_connection(self, src_node, dst_node, bitrate_gbps, path=None):
        hosts, switches = TOPOLOGY["hosts"], TOPOLOGY["switches"]
        if src_node not in hosts or dst_node not in hosts:
            raise ValueError(f"Nodos desconocidos: {src_node}, {dst_node}")
        connection_id = f"{src_node}-{dst_node}"
        if connection_id in self.connections:   # idempotente: re-provisiona
            self.remove_connection(connection_id)
        path = path or default_path(src_node, dst_node)
        src_ip, dst_ip = hosts[src_node]["ip"], hosts[dst_node]["ip"]
        cookie = self._next_cookie
        self._next_cookie += 1

        flow_mods = 0
        for i in range(1, len(path) - 1):
            node = path[i]
            dp = self.datapaths.get(switches[node]["dpid"])
            if dp is None:
                raise RuntimeError(f"Switch {node} (dpid={switches[node]['dpid']}) no conectado")
            parser = dp.ofproto_parser
            fwd, rev = port_towards(node, path[i + 1]), port_towards(node, path[i - 1])
            for s_ip, d_ip, port in ((src_ip, dst_ip, fwd), (dst_ip, src_ip, rev)):
                actions = [parser.OFPActionOutput(port)]
                self.add_flow(dp, SERVICE_PRIORITY,
                              parser.OFPMatch(eth_type=ether_types.ETH_TYPE_IP, ipv4_src=s_ip, ipv4_dst=d_ip),
                              actions, cookie=cookie)
                self.add_flow(dp, SERVICE_PRIORITY,
                              parser.OFPMatch(eth_type=ether_types.ETH_TYPE_ARP, arp_spa=s_ip, arp_tpa=d_ip),
                              actions, cookie=cookie)
                flow_mods += 2

        self.isolated_pairs.discard(frozenset((src_ip, dst_ip)))
        self.connections[connection_id] = {
            "src_node": src_node, "dst_node": dst_node, "bitrate_gbps": bitrate_gbps,
            "path": path, "cookie": cookie, "flow_mods": flow_mods, "status": "UP",
        }
        self.logger.info("Conexión %s UP por %s (%d FLOW_MOD)", connection_id, path, flow_mods)
        return {"connection_id": connection_id, "status": "UP",
                "message": f"Connection {connection_id} established"}

    def remove_connection(self, connection_id):
        conn = self.connections.pop(connection_id, None)
        if conn is None:
            raise KeyError(connection_id)
        for dp in self.datapaths.values():
            self.delete_flows(dp, conn["cookie"])
            self.delete_flows(dp, LEARNING_COOKIE)
        hosts = TOPOLOGY["hosts"]
        self.isolated_pairs.add(frozenset((hosts[conn["src_node"]]["ip"], hosts[conn["dst_node"]]["ip"])))
        return {"connection_id": connection_id, "status": "DOWN",
                "message": f"Connection {connection_id} removed"}

    def status(self):
        return {
            "connections": {cid: {k: v for k, v in c.items() if k != "cookie"}
                            for cid, c in self.connections.items()},
            "isolated_pairs": [sorted(p) for p in self.isolated_pairs],
            "switches": sorted(self.datapaths),
        }


def _json(obj, status=200):
    return Response(status=status, content_type="application/json", charset="utf-8",
                    text=json.dumps(obj))


class IPServiceRestAPI(ControllerBase):
    """NBI REST del controlador (3 endpoints)."""

    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[API_INSTANCE]

    @route("ipservice", "/service/ip/create", methods=["POST"])
    def create(self, req, **kwargs):
        try:
            body = json.loads(req.body.decode("utf-8") or "{}")
            result = self.app.establish_connection(
                body["src_node"], body["dst_node"], float(body.get("bitrate_gbps", 0)), body.get("path"))
        except KeyError as exc:
            return _json({"error": f"Falta el campo {exc}"}, 400)
        except (ValueError, RuntimeError) as exc:
            return _json({"error": str(exc)}, 409)
        return _json(result)

    @route("ipservice", "/service/ip/status", methods=["GET"])
    def get_status(self, req, **kwargs):
        return _json(self.app.status())

    @route("ipservice", "/service/ip/delete", methods=["DELETE"])
    def delete(self, req, **kwargs):
        connection_id = req.params.get("connection_id")
        if not connection_id and req.body:
            connection_id = json.loads(req.body.decode("utf-8")).get("connection_id")
        try:
            return _json(self.app.remove_connection(connection_id))
        except KeyError:
            return _json({"error": f"Conexión {connection_id} no existe"}, 404)
