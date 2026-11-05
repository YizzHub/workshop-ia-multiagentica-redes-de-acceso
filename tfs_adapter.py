"""
tfs_adapter.py
==============
Adaptador de la herramienta MCP ``establish_ip_connection`` hacia ETSI
TeraFlowSDN (TFS), el controlador SDN *cloud-native* de ETSI.

Mapeo conceptual Ryu -> TFS:
    POST /service/ip/create (Ryu)  ==>  NBI "TFS API":
        POST /tfs-api/context/{context_uuid}/services
    que el microservicio NBI traduce a la RPC gRPC ``ServiceService.CreateService``
    + ``UpdateService``. ``Service`` invoca a ``PathComp`` para la ruta y a
    ``Device`` (drivers OpenFlow/OpenConfig/TAPI...) para configurar los equipos.

El descriptor sigue el formato protobuf-JSON de los *descriptors* de TFS
(context.proto). Verifique los campos exactos contra el Swagger de su despliegue.
Con ``dry_run=True`` NO se contacta a ningún TFS real: se devuelve el descriptor.
"""
from __future__ import annotations

from access_topology import TOPOLOGY, port_towards
import workshop_common as wc


class TeraFlowSDNAdapter:
    def __init__(self, base_url: str = "http://127.0.0.1:80", context_uuid: str = "admin",
                 dry_run: bool = True):
        self.base_url = base_url.rstrip("/")
        self.context_uuid = context_uuid
        self.dry_run = dry_run

    def _endpoint(self, device: str, port: int) -> dict:
        return {"device_id": {"device_uuid": {"uuid": device}},
                "endpoint_uuid": {"uuid": str(port)}}

    def build_service_descriptor(self, src_node: str, dst_node: str, bitrate_gbps: float,
                                 path: list[str] | None = None,
                                 service_type: str = "SERVICETYPE_L3NM") -> dict:
        """Construye el descriptor de un servicio L3NM (o L2NM / TAPI_CONNECTIVITY_SERVICE)."""
        hosts = TOPOLOGY["hosts"]
        src_sw, dst_sw = hosts[src_node]["switch"], hosts[dst_node]["switch"]
        service_uuid = f"{src_node}-{dst_node}"
        descriptor = {
            "service_id": {"context_id": {"context_uuid": {"uuid": self.context_uuid}},
                           "service_uuid": {"uuid": service_uuid}},
            "name": service_uuid,
            "service_type": service_type,
            "service_status": {"service_status": "SERVICESTATUS_PLANNED"},
            "service_endpoint_ids": [
                self._endpoint(src_sw, port_towards(src_sw, src_node)),
                self._endpoint(dst_sw, port_towards(dst_sw, dst_node)),
            ],
            "service_constraints": [
                {"custom": {"constraint_type": "bandwidth[gbps]", "constraint_value": str(float(bitrate_gbps))}},
            ],
            "service_config": {"config_rules": [
                {"action": "CONFIGACTION_SET", "custom": {
                    "resource_key": "/settings",
                    "resource_value": {"src_ip": hosts[src_node]["ip"], "dst_ip": hosts[dst_node]["ip"]}}},
            ]},
        }
        if path:
            # Ruta sugerida por el PCE externo: TFS la recibe como restricción (en
            # producción la calcularía su propio microservicio PathComp).
            descriptor["service_constraints"].append(
                {"custom": {"constraint_type": "explicit_route", "constraint_value": ",".join(path)}})
        return {"services": [descriptor]}

    def create_ip_service(self, src_node: str, dst_node: str, bitrate_gbps: float,
                          path: list[str] | None = None) -> dict:
        body = self.build_service_descriptor(src_node, dst_node, bitrate_gbps, path)
        endpoint = f"{self.base_url}/tfs-api/context/{self.context_uuid}/services"
        service_uuid = f"{src_node}-{dst_node}"
        if self.dry_run:
            wc.CAPTURE.record("Agent", "TFS (dry)", "HTTP",
                              f"POST /tfs-api/context/{self.context_uuid}/services HTTP/1.1  (dry-run)", body)
        else:
            wc.http_call("Agent", "TFS", "POST", endpoint, body)
        return {"connection_id": service_uuid, "status": "UP",
                "message": f"Connection {service_uuid} established"}

    def delete_ip_service(self, connection_id: str) -> dict:
        endpoint = f"{self.base_url}/tfs-api/context/{self.context_uuid}/service/{connection_id}"
        if self.dry_run:
            wc.CAPTURE.record("Agent", "TFS (dry)", "HTTP",
                              f"DELETE /tfs-api/context/{self.context_uuid}/service/{connection_id} (dry-run)")
        else:
            wc.http_call("Agent", "TFS", "DELETE", endpoint)
        return {"connection_id": connection_id, "status": "DOWN",
                "message": f"Connection {connection_id} removed"}
