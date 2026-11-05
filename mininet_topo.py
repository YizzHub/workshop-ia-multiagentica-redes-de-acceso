#!/usr/bin/env python3
"""
mininet_topo.py
===============
Topología Mininet del laboratorio (OpenFlow 1.3 + controlador remoto Ryu):

    h1 (10.0.0.1) -- r1 -- r2 -- r3 -- h2 (10.0.0.2)

Modos de uso (requiere Linux + root):
    sudo python3 mininet_topo.py --mode cli     # CLI interactiva (mininet> h1 ping h2)
    sudo python3 mininet_topo.py --mode test    # ping h1->h2 y termina
    sudo python3 mininet_topo.py --mode serve   # mantiene la red y expone un canal HTTP
                                                # GET /ping?src=h1&dst=h2&count=3
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mininet.cli import CLI                                   # noqa: E402
from mininet.log import info, setLogLevel                     # noqa: E402
from mininet.net import Mininet                               # noqa: E402
from mininet.node import OVSSwitch, RemoteController          # noqa: E402
from mininet.topo import Topo                                 # noqa: E402

from access_topology import TOPOLOGY                          # noqa: E402


class AccessAggregationTopo(Topo):
    """H1 (acceso) - R1 - R2 - R3 - H2 (borde), puertos fijos según access_topology.py."""

    def build(self):
        nodes = {}
        for hid, h in TOPOLOGY["hosts"].items():
            nodes[hid] = self.addHost(h["name"], ip=f"{h['ip']}/24", mac=h["mac"])
        for sid, s in TOPOLOGY["switches"].items():
            nodes[sid] = self.addSwitch(s["name"], dpid=f"{s['dpid']:016x}", protocols="OpenFlow13")
        for link in TOPOLOGY["links"]:
            if not link["in_mininet"]:
                continue  # el bypass óptico no se emula en Mininet
            opts = {}
            if link["a_port"] is not None:
                opts["port1"] = link["a_port"]
            if link["b_port"] is not None:
                opts["port2"] = link["b_port"]
            self.addLink(nodes[link["a"]], nodes[link["b"]], **opts)


def build_network(controller_ip, controller_port):
    net = Mininet(topo=AccessAggregationTopo(), switch=OVSSwitch, build=False,
                  controller=None, autoSetMacs=False)
    net.addController("ryu", controller=RemoteController, ip=controller_ip, port=controller_port)
    net.build()
    net.start()
    return net


def ping(net, src, dst, count=3):
    dst_ip = net.get(dst).IP()
    return net.get(src).cmd(f"ping -c {count} -W 1 {dst_ip}")


def serve(net, port):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if parsed.path == "/health":
                self._send(200, {"status": "ok", "hosts": [h.name for h in net.hosts]})
            elif parsed.path == "/ping":
                out = ping(net, qs.get("src", "h1"), qs.get("dst", "h2"), int(qs.get("count", 3)))
                self._send(200, {"output": out})
            else:
                self._send(404, {"error": "not found"})

        def log_message(self, *args):
            pass

    info(f"*** Canal de control Mininet en http://127.0.0.1:{port}\n")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--mode", choices=["cli", "test", "serve"], default="cli")
    parser.add_argument("--serve-port", type=int, default=9000)
    args = parser.parse_args()

    setLogLevel("info")
    net = build_network(args.controller_ip, args.controller_port)
    try:
        if args.mode == "test":
            info("*** Prueba de aislamiento inicial: h1 ping h2\n")
            print(ping(net, "h1", "h2"))
        elif args.mode == "serve":
            serve(net, args.serve_port)
        else:
            CLI(net)
    except KeyboardInterrupt:
        pass
    finally:
        net.stop()


if __name__ == "__main__":
    main()
