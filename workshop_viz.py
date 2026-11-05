"""
workshop_viz.py
===============
Visualizaciones pequeñas (matplotlib) que acompañan cada estación del workshop:

  * plot_network   : topología H1–R1–R2–R3–H2 con estado de enlaces, ruta del PCE,
                     conexión provisionada y política de aislamiento.
  * plot_similarity: similitud coseno (ChromaDB) entre prompts y workflows.
  * plot_sequence  : diagrama de secuencia generado desde la captura real (Fig. 2/3).
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

POS = {"H1": (0, 0), "R1": (1.4, 0), "R2": (2.8, 0), "R3": (4.2, 0), "H2": (5.6, 0)}
ROLES = {"H1": "ONU/CPE\n10.0.0.1", "H2": "BNG/Edge\n10.0.0.2"}
C = {"bg": "#0b1220", "txt": "#e6edf7", "mut": "#9fb0cc", "host": "#9ecae1", "sw": "#fdae6b",
     "up": "#c9d6ea", "standby": "#ba68c8", "degraded": "#e57373", "conn": "#4fc3f7", "pce": "#ffb74d",
     "req": "#4fc3f7", "resp": "#81c784", "of": "#f06292", "mcp": "#ffb74d", "bad": "#e57373"}


def _style(ax, fig):
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    ax.axis("off")


def _links(source):
    """Acepta un NetworkInventory (con .graph) o el dict TOPOLOGY de access_topology.py."""
    if hasattr(source, "graph"):
        return [(u, v, d) for u, v, d in source.graph.edges(data=True)]
    return [(l["a"], l["b"], l) for l in source["links"]]


def plot_network(source, title, connection_path=None, computed_path=None, isolated=False):
    """Dibuja la red de acceso/agregación y su estado actual."""
    fig, ax = plt.subplots(figsize=(10, 3.3))
    _style(ax, fig)
    conn_edges = {frozenset(e) for e in zip(connection_path or [], (connection_path or [])[1:])}
    pce_edges = {frozenset(e) for e in zip(computed_path or [], (computed_path or [])[1:])}
    for u, v, d in _links(source):
        (x1, y1), (x2, y2) = POS[u], POS[v]
        residual = d["capacity_gbps"] - d["used_gbps"]
        color = C.get(d["state"], C["up"])
        style = "-" if d["state"] == "up" else "--"
        rad = 0.45 if d["layer"] == "optical" else 0.0
        key = frozenset((u, v))
        if key in conn_edges:
            ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                         color=C["conn"], lw=16, alpha=0.55, zorder=1))
        if key in pce_edges:
            ax.add_patch(FancyArrowPatch((x1, y1 - 0.06), (x2, y2 - 0.06), connectionstyle=f"arc3,rad={rad}",
                                         arrowstyle="-", color=C["pce"], lw=2.5, linestyle=":", zorder=3))
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-",
                                     color=color, lw=2, linestyle=style, zorder=2))
        lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 + (0.16 if d["layer"] == "packet" else -1.18)
        label = f"{d['state']} · {residual:g}/{d['capacity_gbps']:g} Gb/s libres"
        if d["layer"] == "optical":
            label = "bypass óptico R1–R3 · " + label
        ax.text(lx, ly, label, color=color, fontsize=8, ha="center")
    for n, (x, y) in POS.items():
        if n.startswith("H"):
            ax.add_patch(plt.Circle((x, y), 0.22, color=C["host"], zorder=4))
            ax.text(x, y - 0.5, ROLES[n], color=C["mut"], fontsize=8, ha="center", va="top")
        else:
            ax.add_patch(plt.Rectangle((x - 0.25, y - 0.17), 0.5, 0.34, color=C["sw"], zorder=4))
        ax.text(x, y, n, color="#04121f", fontsize=10, ha="center", va="center", fontweight="bold", zorder=5)
    if isolated:
        ax.text(2.8, 0.62, "✖  Política de aislamiento activa: H1 y H2 no se alcanzan", color=C["bad"],
                fontsize=11, ha="center", fontweight="bold")
    elif connection_path:
        ax.text(2.8, 0.62, "✔  Conexión H1–H2 activa: " + " → ".join(connection_path), color=C["conn"],
                fontsize=11, ha="center", fontweight="bold")
    legend = [("conexión provisionada (SDN)", C["conn"]), ("ruta calculada (PCE)", C["pce"]),
              ("enlace standby", C["standby"]), ("enlace degradado", C["degraded"])]
    for i, (txt, col) in enumerate(legend):
        ax.text(-0.3 + i * 1.6, -1.68, "■ " + txt, color=col, fontsize=8)
    ax.set_xlim(-0.5, 6.1)
    ax.set_ylim(-1.78, 0.85)
    ax.set_title(title, color=C["txt"], fontsize=12, loc="left")
    plt.tight_layout()
    plt.show()


def plot_similarity(kb, prompts):
    """Barras de similitud coseno entre cada prompt y todos los workflows de ChromaDB."""
    n = len(prompts)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 2.9), sharey=True)
    axes = [axes] if n == 1 else axes
    names = sorted(h["name"] for h in kb.search(prompts[0], k=kb.collection.count()))
    for ax, prompt in zip(axes, prompts):
        hits = {h["name"]: h["similarity"] for h in kb.search(prompt, k=kb.collection.count())}
        best = max(hits, key=hits.get)
        vals = [hits[nm] for nm in names]
        ax.barh(names, vals, color=[C["conn"] if nm == best else "#3a4c70" for nm in names])
        ax.set_facecolor(C["bg"])
        ax.set_xlim(min(0, min(vals)) - 0.05, 1)
        ax.tick_params(colors=C["mut"], labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2a3a5c")
        short = prompt if len(prompt) <= 38 else prompt[:36] + "…"
        ax.set_title(f"«{short}»", color=C["txt"], fontsize=8.5)
    fig.patch.set_facecolor(C["bg"])
    fig.suptitle("ChromaDB · similitud coseno prompt ↔ workflow (azul = workflow recuperado)",
                 color=C["txt"], fontsize=11, x=0.01, ha="left")
    plt.tight_layout()
    plt.show()


LANES = ["User", "Orchestrator", "LLM", "Agent", "PCE", "SDN", "OVS"]
LANE_LABEL = {"User": "Operador", "Orchestrator": "Orquestador", "LLM": "LLM+ChromaDB", "Agent": "Agente (A2A)",
              "PCE": "PCE", "SDN": "Ryu SDN", "OVS": "OVS (Mininet)"}


def plot_sequence(frames, step_fn, title="Diagrama de secuencia generado desde la captura real"):
    """Diagrama de secuencia (HTTP + OpenFlow agregado) a partir de wc.CAPTURE.frames."""
    rows, i = [], 0
    while i < len(frames):
        f = frames[i]
        if f["protocol"] == "OpenFlow":
            j = i
            while j < len(frames) and frames[j]["protocol"] == "OpenFlow":
                j += 1
            n_mod = sum("FLOW_MOD" in x["info"] for x in frames[i:j])
            if n_mod:
                rows.append(("SDN", "OVS", f"{n_mod}× OFPT_FLOW_MOD", C["of"], "6"))
            i = j
            continue
        if f["protocol"] == "HTTP" and f["src"].split()[0] in LANES and f["dst"].split()[0] in LANES:
            resp = f["info"].startswith("HTTP/1.1")
            label = f["info"].replace("  (application/json)", "").replace(" HTTP/1.1", "")
            rows.append((f["src"], f["dst"], label, C["resp"] if resp else C["req"], step_fn(f)))
        i += 1
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(rows) + 1.2))
    _style(ax, fig)
    xs = {lane: k for k, lane in enumerate(LANES)}
    for lane, x in xs.items():
        ax.plot([x, x], [0.4, len(rows) + 0.4], color="#2a3a5c", lw=1.2, zorder=1)
        ax.text(x, 0, LANE_LABEL[lane], color=C["txt"], fontsize=9, ha="center", va="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", fc="#121b2e", ec="#4fc3f7"))
    for r, (src, dst, label, color, step) in enumerate(rows, 1):
        x1, x2 = xs[src], xs[dst]
        ax.annotate("", xy=(x2, r), xytext=(x1, r), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8), zorder=2)
        ax.text((x1 + x2) / 2, r - 0.12, label, color=color, fontsize=8, ha="center", va="bottom")
        if step:
            ax.text(min(x1, x2) - 0.12, r, step, color="#04121f", fontsize=8, ha="right", va="center",
                    fontweight="bold", bbox=dict(boxstyle="circle,pad=0.25", fc=color, ec="none"))
    ax.set_xlim(-0.7, len(LANES) - 0.4)
    ax.set_ylim(len(rows) + 0.7, -0.6)
    ax.set_title(title + "   (círculo = paso de la Figura 2)", color=C["txt"], fontsize=11, loc="left")
    plt.tight_layout()
    plt.show()
