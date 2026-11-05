"""
mcp_server.py
=============
Servidor MCP (Model Context Protocol) construido con FastMCP. Expone como
*Tools* las capacidades DETERMINISTAS de la red:

    compute_route(src_node, dst_node)                      -> PCE   (POST /compute_route)
    establish_ip_connection(src_node, dst_node, bitrate)   -> NBI SDN (POST /service/ip/create)
    remove_ip_connection(src_node, dst_node)               -> NBI SDN (DELETE /service/ip/delete)
    get_connection_status()                                -> NBI SDN (GET /service/ip/status)

El LLM NUNCA calcula rutas ni programa flujos: solo elige herramientas y
argumentos. La ejecución es código determinista y auditable.

Uso standalone (transporte HTTP "streamable"):
    python mcp_server.py      # -> http://127.0.0.1:8005/mcp
"""
from __future__ import annotations

import inspect
import json
from typing import Annotated, Any, Callable

from pydantic import Field

import workshop_common as wc

try:
    from fastmcp import Client, FastMCP
    FASTMCP_AVAILABLE = True
except ImportError:  # pragma: no cover - respaldo mínimo compatible
    FASTMCP_AVAILABLE = False

    class FastMCP:  # type: ignore[no-redef]
        """Implementación mínima (solo para entornos sin fastmcp)."""

        def __init__(self, name: str, instructions: str = ""):
            self.name, self.instructions = name, instructions
            self._tools: dict[str, Callable] = {}

        def tool(self, fn: Callable | None = None, *, name: str | None = None, description: str | None = None):
            def register(f: Callable) -> Callable:
                f.__mcp_description__ = description or inspect.getdoc(f) or ""
                self._tools[name or f.__name__] = f
                return f
            return register(fn) if fn is not None else register

    Client = None  # type: ignore[assignment]


def _schema_for(fn: Callable) -> dict:
    from pydantic import create_model
    fields = {}
    for pname, param in inspect.signature(fn).parameters.items():
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[pname] = (param.annotation, default)
    schema = create_model(f"{fn.__name__}_args", **fields).model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


mcp = FastMCP(
    name="TelecomNetworkMCP",
    instructions="Herramientas deterministas de red: PCE e interfaz Norte (NBI) del controlador SDN.",
)

# Caché de rutas calculadas por el PCE: establish_ip_connection reutiliza la ruta
_ROUTE_CACHE: dict[tuple[str, str], list[str]] = {}

NodeName = Annotated[str, Field(description="Network node identifier, e.g. 'H1', 'R2', 'H2'")]


def _sdn_backend():
    if wc.CONFIG.get("SDN_BACKEND") == "tfs":
        from tfs_adapter import TeraFlowSDNAdapter
        return TeraFlowSDNAdapter(wc.CONFIG["TFS_URL"], wc.CONFIG["TFS_CONTEXT"], wc.CONFIG["TFS_DRY_RUN"])
    return None


@mcp.tool
def compute_route(src_node: NodeName, dst_node: NodeName) -> dict:
    """Compute the shortest route between source and destination nodes for a candidate
    connection, using the deterministic Path Computation Element (PCE)."""
    result = wc.http_call("Agent", "PCE", "POST", f"{wc.url('pce')}/compute_route",
                          {"src_node": src_node, "dst_node": dst_node})
    _ROUTE_CACHE[(src_node, dst_node)] = result["path"]
    return result


@mcp.tool
def establish_ip_connection(
    src_node: NodeName,
    dst_node: NodeName,
    bitrate_gbps: Annotated[float, Field(description="Requested bandwidth in Gb/s", gt=0)],
) -> dict:
    """Provision and bring up a new IP connection on the SDN-controlled network through
    the SDN controller Northbound Interface (installs OpenFlow rules along the route)."""
    path = _ROUTE_CACHE.get((src_node, dst_node))
    tfs = _sdn_backend()
    if tfs is not None:
        return tfs.create_ip_service(src_node, dst_node, bitrate_gbps, path)
    return wc.http_call("Agent", "SDN", "POST", f"{wc.url('sdn')}/service/ip/create",
                        {"src_node": src_node, "dst_node": dst_node,
                         "bitrate_gbps": bitrate_gbps, "path": path})


@mcp.tool
def remove_ip_connection(src_node: NodeName, dst_node: NodeName) -> dict:
    """Tear down (remove/delete) an existing IP connection on the SDN-controlled network
    and restore the default isolation policy."""
    connection_id = f"{src_node}-{dst_node}"
    tfs = _sdn_backend()
    if tfs is not None:
        return tfs.delete_ip_service(connection_id)
    return wc.http_call("Agent", "SDN", "DELETE",
                        f"{wc.url('sdn')}/service/ip/delete?connection_id={connection_id}")


@mcp.tool
def get_connection_status() -> dict:
    """Retrieve the status of the IP connections and isolation policies from the SDN controller."""
    return wc.http_call("Agent", "SDN", "GET", f"{wc.url('sdn')}/service/ip/status")


# =============================================================================
# Cliente MCP utilizado por los agentes (sesión + descubrimiento + invocación)
# =============================================================================
class MCPSession:
    """Sesión MCP del agente. Usa transporte en memoria (por defecto) o HTTP."""

    def __init__(self, server: Any = None, caller: str = "Agent", server_url: str | None = None):
        self.server = server if server is not None else mcp
        self.caller = caller
        self.server_url = server_url
        self._client = None

    async def __aenter__(self) -> "MCPSession":
        wc.CAPTURE.record(self.caller, "MCP Server", "MCP", "JSON-RPC initialize")
        if FASTMCP_AVAILABLE:
            self._client = Client(self.server_url or self.server)
            await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)
            self._client = None

    async def list_tools(self) -> list[dict]:
        wc.CAPTURE.record(self.caller, "MCP Server", "MCP", "JSON-RPC tools/list")
        if FASTMCP_AVAILABLE:
            tools = []
            for t in await self._client.list_tools():
                schema = getattr(t, "input_schema", None)
                if schema is None:
                    schema = t.inputSchema
                tools.append({"name": t.name, "description": t.description or "", "inputSchema": schema})
        else:
            tools = [{"name": n, "description": f.__mcp_description__, "inputSchema": _schema_for(f)}
                     for n, f in self.server._tools.items()]
        wc.CAPTURE.record("MCP Server", self.caller, "MCP",
                          f"JSON-RPC result: {len(tools)} tools", [t["name"] for t in tools])
        return tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        wc.CAPTURE.record(self.caller, "MCP Server", "MCP", f"JSON-RPC tools/call name={name}", arguments)
        if FASTMCP_AVAILABLE:
            res = await self._client.call_tool(name, arguments)
            data = res.structured_content
            if isinstance(data, dict) and set(data) == {"result"}:
                data = data["result"]
            if data is None:
                data = getattr(res, "data", None)
            if data is None and res.content:
                data = json.loads(res.content[0].text)
        else:
            data = self.server._tools[name](**arguments)
        wc.CAPTURE.record("MCP Server", self.caller, "MCP", f"JSON-RPC result ({name})", data)
        return data


if __name__ == "__main__":
    mcp.run(transport="http", host=wc.CONFIG["HOST"], port=wc.CONFIG["PORTS"]["mcp_http"])
