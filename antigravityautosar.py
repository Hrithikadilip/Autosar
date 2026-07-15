import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import time
from sklearn.ensemble import IsolationForest

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="TSN Automotive Ethernet – Advanced Dashboard",
    layout="wide",
    page_icon="🚗"
)

# =========================================================
# DARK CYBERPUNK CSS
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
.stApp { background: linear-gradient(135deg, #050b14 0%, #0a1628 40%, #050d1f 100%); color: #c8d6e5; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #060e1e 0%, #0a1628 100%); border-right: 1px solid rgba(14,246,255,0.13); }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #a0c4ff !important; }
h1 { font-family: 'Rajdhani', sans-serif !important; font-size: 2.4rem !important; font-weight: 700 !important;
     background: linear-gradient(90deg, #0ef6ff, #7a7fff, #ff6ef7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: 2px; }
h2, h3 { font-family: 'Rajdhani', sans-serif !important; color: #7dd3fc !important; letter-spacing: 1px; }
[data-testid="stMetric"] { background: linear-gradient(135deg, #0d1f3e 0%, #122040 100%);
    border: 1px solid rgba(14,246,255,0.2); border-radius: 12px; padding: 14px 18px;
    box-shadow: 0 0 18px rgba(14,246,255,0.08); }
[data-testid="stMetricLabel"] { color: #7da8d0 !important; font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1.5px; }
[data-testid="stMetricValue"] { color: #0ef6ff !important; font-family: 'Share Tech Mono', monospace !important; font-size: 1.3rem !important; }
hr { border-color: rgba(14,246,255,0.13) !important; }
.section-banner { background: linear-gradient(90deg, rgba(14,246,255,0.07), rgba(122,127,255,0.07), transparent);
    border-left: 4px solid #0ef6ff; border-radius: 0 8px 8px 0; padding: 8px 18px; margin: 20px 0 12px 0;
    font-family: 'Rajdhani', sans-serif; font-size: 1.4rem; font-weight: 700; color: #0ef6ff; letter-spacing: 2px; text-transform: uppercase; }
.omnet-banner { background: linear-gradient(90deg, rgba(255,110,247,0.07), rgba(122,127,255,0.07), transparent);
    border-left: 4px solid #ff6ef7; color: #ff6ef7; }
.info-card { background: #0d1f3e; border: 1px solid rgba(122,127,255,0.27); border-radius: 10px;
    padding: 14px 20px; font-size: 0.92rem; color: #b0c8e8; margin: 8px 0; }
.info-card b { color: #0ef6ff; }
.obs-item { background: linear-gradient(90deg, #0a1a2f, #0d2244); border-left: 3px solid #0ef6ff;
    border-radius: 0 6px 6px 0; padding: 10px 14px; margin: 6px 0; color: #c8d6e5; font-size: 0.93rem; }
</style>
""", unsafe_allow_html=True)

# =========================================================
# COLOR CONSTANTS
# =========================================================
NEON_BLUE   = "#0ef6ff"
NEON_PINK   = "#ff6ef7"
NEON_GREEN  = "#39ff14"
NEON_GOLD   = "#ffd700"
NEON_PURPLE = "#b44fff"

BG_DARK  = "#060e1e"
BG_PLOT  = "#0a1628"
GRID_COL = "#1a3a5c"
TEXT_COL = "#c8d6e5"
AXIS_COL = "#4a6f8a"

# =========================================================
# THEME HELPER — Plotly 6 safe
# =========================================================
def dark_theme(fig, title="", height=340, xlabel="", ylabel="", show_legend=True):
    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PLOT,
        height=height,
        showlegend=show_legend,
        margin=dict(l=40, r=20, t=55, b=40),
    )
    fig.update_layout(
        legend=dict(bgcolor=BG_PLOT, bordercolor=GRID_COL,
                    font=dict(color=TEXT_COL, size=11)),
        font=dict(family="monospace", color=TEXT_COL),
        title=dict(text=title, x=0.5,
                   font=dict(size=13, color="#7dd3fc")),
    )
    fig.update_xaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     color=AXIS_COL, title_text=xlabel,
                     title_font=dict(color=AXIS_COL))
    fig.update_yaxes(gridcolor=GRID_COL, zerolinecolor=GRID_COL,
                     color=AXIS_COL, title_text=ylabel,
                     title_font=dict(color=AXIS_COL))

# =========================================================
# DATASET FILES
# =========================================================
TSN_FILE = "driving 2 original.csv"
ETH_FILE = "driving 2 injected.csv"

# =========================================================
# LOAD CSV
# =========================================================
@st.cache_data
def load_csv(file_name):
    path = Path(file_name)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_name}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Time", "Source", "Destination", "Protocol", "Length", "Info"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_name} missing columns: {missing}")
    df["Time"]   = pd.to_numeric(df["Time"],   errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df = df.dropna(subset=["Time", "Length"]).sort_values("Time").reset_index(drop=True)
    df["InterArrival"]   = df["Time"].diff().fillna(0)
    df["Bits"]           = df["Length"] * 8
    df["PacketDelay_ms"] = df["InterArrival"] * 1000
    return df

# =========================================================
# METRICS
# =========================================================
@st.cache_data
def calculate_metrics(df, link_speed_mbps=100):
    duration      = max(df["Time"].max() - df["Time"].min(), 1e-9)
    total_packets = len(df)
    total_bytes   = df["Length"].sum()
    throughput_mbps      = (total_bytes * 8) / duration / 1e6
    mean_interarrival_ms = df["InterArrival"].mean() * 1000
    jitter_ms            = df["InterArrival"].std() * 1000
    avg_pkt_len          = df["Length"].mean()
    serialization_ms     = ((avg_pkt_len * 8) / (link_speed_mbps * 1e6)) * 1000
    utilization          = min(throughput_mbps / max(link_speed_mbps, 1e-9), 0.95)
    queue_delay_ms       = serialization_ms * (utilization / max(1 - utilization, 1e-6))
    total_delay_ms       = serialization_ms + queue_delay_ms
    packet_rate          = total_packets / duration
    return {
        "Packets":                   total_packets,
        "Duration (s)":              duration,
        "Packet Rate (pkt/s)":       packet_rate,
        "Throughput (Mbps)":         throughput_mbps,
        "Mean InterArrival (ms)":    mean_interarrival_ms,
        "Jitter (ms)":               jitter_ms,
        "Avg Packet Length (Bytes)": avg_pkt_len,
        "SOME/IP Serialization (ms)": serialization_ms,
        "RTE Queue Delay (ms)":      queue_delay_ms,
        "E2E Latency (ms)":          total_delay_ms,
        "Utilization (%)":           utilization * 100,
    }

# =========================================================
# TRAFFIC OVER TIME
# =========================================================
@st.cache_data
def traffic_over_time(df, window=0.1):
    start = df["Time"].min()
    end   = df["Time"].max() + window
    bins  = np.arange(start, end + window, window)
    if len(bins) < 2:
        bins = np.array([start, start + window])
    temp = df.copy()
    temp["Window"] = pd.cut(temp["Time"], bins=bins, labels=bins[:-1], include_lowest=True)
    grouped = temp.groupby("Window", observed=False).agg(
        packets=("Length", "count"),
        bytes=("Length", "sum")
    ).reset_index()
    grouped["Window"]          = grouped["Window"].astype(float)
    grouped["throughput_mbps"] = (grouped["bytes"] * 8) / window / 1e6
    return grouped

# =========================================================
# SIMULATION MODEL
# =========================================================
def simulate_network(link_speed_mbps, packet_size_bytes, offered_load_mbps, tsn=True):
    serialization_ms = ((packet_size_bytes * 8) / (link_speed_mbps * 1e6)) * 1000
    rho = min(offered_load_mbps / max(link_speed_mbps, 1e-9), 0.95)
    if tsn:
        queue_delay_ms = serialization_ms * (0.25 * rho / max(1 - rho, 1e-6))
        jitter_ms      = 0.10 + 0.50 * rho
        packet_loss    = max(0, 0.15 * rho)
        priority_score = 95 - 10 * rho
    else:
        queue_delay_ms = serialization_ms * (1.20 * rho / max(1 - rho, 1e-6))
        jitter_ms      = 0.50 + 2.50 * rho
        packet_loss    = max(0, 1.20 * rho)
        priority_score = 70 - 25 * rho
    return serialization_ms + queue_delay_ms, jitter_ms, packet_loss, priority_score

@st.cache_data
def simulation_curves(link_speed_mbps, packet_size_bytes):
    loads = np.arange(10, 100, 5)
    tsn_delay, eth_delay   = [], []
    tsn_jitter, eth_jitter = [], []
    tsn_loss, eth_loss     = [], []
    for load in loads:
        d1, j1, l1, _ = simulate_network(link_speed_mbps, packet_size_bytes, load, tsn=True)
        d2, j2, l2, _ = simulate_network(link_speed_mbps, packet_size_bytes, load, tsn=False)
        tsn_delay.append(d1);  eth_delay.append(d2)
        tsn_jitter.append(j1); eth_jitter.append(j2)
        tsn_loss.append(l1);   eth_loss.append(l2)
    return loads, tsn_delay, eth_delay, tsn_jitter, eth_jitter, tsn_loss, eth_loss

# =========================================================
# PLOTLY CHART FUNCTIONS  (all Plotly 6 safe)
# =========================================================
def plotly_bar(title, ylabel, tsn_val, eth_val):
    fig = go.Figure(data=[
        go.Bar(name="TSN",      x=["TSN"],      y=[tsn_val],
               marker_color=NEON_BLUE, marker_line_width=0,
               text=[f"{tsn_val:.4f}"], textposition="outside",
               textfont=dict(color=NEON_BLUE, size=11)),
        go.Bar(name="Ethernet", x=["Ethernet"], y=[eth_val],
               marker_color=NEON_PINK, marker_line_width=0,
               text=[f"{eth_val:.4f}"], textposition="outside",
               textfont=dict(color=NEON_PINK, size=11)),
    ])
    dark_theme(fig, title=title, height=300, ylabel=ylabel, show_legend=False)
    fig.update_layout(bargroupgap=0.3)
    st.plotly_chart(fig, use_container_width=True)

def plotly_line(x1, y1, x2, y2, title, ylabel, xlabel="Time / Load"):
    fig = go.Figure([
        go.Scatter(x=list(x1), y=list(y1), mode="lines", name="TSN",
                   line=dict(color=NEON_BLUE, width=3),
                   fill="tozeroy", fillcolor="rgba(14,246,255,0.09)"),
        go.Scatter(x=list(x2), y=list(y2), mode="lines", name="Ethernet",
                   line=dict(color=NEON_PINK, width=3),
                   fill="tozeroy", fillcolor="rgba(255,110,247,0.09)"),
    ])
    dark_theme(fig, title=title, height=320, xlabel=xlabel, ylabel=ylabel)
    st.plotly_chart(fig, use_container_width=True)

def plotly_histogram(d1, d2, title, xlabel):
    fig = go.Figure([
        go.Histogram(x=d1, name="TSN",      nbinsx=50,
                     opacity=0.75, marker_color=NEON_BLUE),
        go.Histogram(x=d2, name="Ethernet", nbinsx=50,
                     opacity=0.75, marker_color=NEON_PINK),
    ])
    dark_theme(fig, title=title, height=320, xlabel=xlabel, ylabel="Frequency")
    fig.update_layout(barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

def plotly_horizontal_bar(all_labels, tsn_vals, eth_vals, title):
    fig = go.Figure([
        go.Bar(y=all_labels, x=tsn_vals, name="TSN",
               orientation="h", marker_color=NEON_BLUE, opacity=0.85),
        go.Bar(y=all_labels, x=eth_vals, name="Ethernet",
               orientation="h", marker_color=NEON_PINK, opacity=0.85),
    ])
    dark_theme(fig, title=title, height=380)
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed", gridcolor=GRID_COL,
                     zerolinecolor=GRID_COL, color=AXIS_COL)
    st.plotly_chart(fig, use_container_width=True)

def plotly_radar(tsn_m, eth_m):
    cats = ["Throughput", "Low Latency", "Low Jitter", "Utilization", "Determinism"]
    def norm(v, mn, mx): return (v - mn) / max(mx - mn, 1e-9)

    def scores(tm, em):
        s1 = norm(tm["Throughput (Mbps)"], 0,
                  max(tm["Throughput (Mbps)"], em["Throughput (Mbps)"]))
        s2 = 1 - norm(tm["E2E Latency (ms)"], 0,
                      max(tm["E2E Latency (ms)"], em["E2E Latency (ms)"]) + 1e-9)
        s3 = 1 - norm(tm["Jitter (ms)"], 0,
                      max(tm["Jitter (ms)"], em["Jitter (ms)"]) + 1e-9)
        s4 = norm(tm["Utilization (%)"], 0, 100)
        return [s1*100, s2*100, s3*100, s4*100, 92]

    t_s = scores(tsn_m, eth_m)
    e_s = scores(eth_m, tsn_m)

    fig = go.Figure([
        go.Scatterpolar(r=t_s + [t_s[0]], theta=cats + [cats[0]],
                        fill="toself", name="TSN",
                        line=dict(color=NEON_BLUE, width=2),
                        fillcolor="rgba(14,246,255,0.18)"),
        go.Scatterpolar(r=e_s + [e_s[0]], theta=cats + [cats[0]],
                        fill="toself", name="Ethernet",
                        line=dict(color=NEON_PINK, width=2),
                        fillcolor="rgba(255,110,247,0.18)"),
    ])
    dark_theme(fig, title="AUTOSAR Network Performance Radar", height=420)
    fig.update_layout(
        polar=dict(
            bgcolor=BG_PLOT,
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor=GRID_COL,
                            tickfont=dict(color=AXIS_COL, size=9)),
            angularaxis=dict(gridcolor=GRID_COL,
                             tickfont=dict(color="#7da8d0", size=11)),
        )
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# OMNeT++ TOPOLOGY
# =========================================================
def build_topology():
    nodes = {
        "ECU1\n(Engine)":        (0.10, 0.50),
        "ECU2\n(Body Ctrl)":     (0.10, 0.22),
        "ECU6\n(Brake Ctrl)":    (0.10, 0.76),
        "SW1\n[TSN Switch]":     (0.40, 0.38),
        "SW2\n[Edge Switch]":    (0.65, 0.65),
        "ECU3\n(ADAS)":          (0.90, 0.80),
        "ECU4\n(Infotainment)":  (0.90, 0.50),
        "ECU5\n(Gateway)":       (0.90, 0.20),
    }
    edges = [
        ("ECU1\n(Engine)",       "SW1\n[TSN Switch]"),
        ("ECU2\n(Body Ctrl)",    "SW1\n[TSN Switch]"),
        ("ECU6\n(Brake Ctrl)",   "SW1\n[TSN Switch]"),
        ("SW1\n[TSN Switch]",    "SW2\n[Edge Switch]"),
        ("SW1\n[TSN Switch]",    "ECU5\n(Gateway)"),
        ("SW2\n[Edge Switch]",   "ECU3\n(ADAS)"),
        ("SW2\n[Edge Switch]",   "ECU4\n(Infotainment)"),
    ]
    return nodes, edges

def draw_topology(nodes, edges, tsn_mode=True, active_path=None, step=0):
    ecu_color    = NEON_BLUE   if tsn_mode else NEON_PINK
    sw_color     = NEON_PURPLE if tsn_mode else "#888888"
    active_color = NEON_GREEN  if tsn_mode else NEON_GOLD
    link_color   = GRID_COL

    fig = go.Figure()

    # Draw edges
    for (a, b) in edges:
        x0, y0 = nodes[a]
        x1, y1 = nodes[b]
        is_active = bool(active_path and ((a, b) in active_path or (b, a) in active_path))
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(color=active_color if is_active else link_color,
                      width=5 if is_active else 2),
            hoverinfo="skip",
            showlegend=False,
        ))
        if is_active:
            t    = (step % 10) / 10.0
            px_  = x0 + t * (x1 - x0)
            py_  = y0 + t * (y1 - y0)
            fig.add_trace(go.Scatter(
                x=[px_], y=[py_],
                mode="markers",
                marker=dict(size=16, color=active_color,
                            line=dict(color=active_color, width=2)),
                hoverinfo="skip",
                showlegend=False,
            ))

    # Draw nodes
    for name, (x, y) in nodes.items():
        is_sw   = "SW" in name
        is_act  = bool(active_path and any(
            n == name for edge in active_path for n in edge))
        color   = sw_color if is_sw else ecu_color
        size    = 46 if is_sw else 36
        symbol  = "square" if is_sw else "circle"
        b_color = "#ffffff" if is_act else "rgba(255,255,255,0.25)"
        b_width = 4 if is_act else 1
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color=b_color, width=b_width)),
            text=[name],
            textposition="middle center",
            textfont=dict(color="#ffffff", size=8, family="monospace"),
            hovertemplate=(f"<b>{name.replace(chr(10),' ')}</b><br>"
                           + ("TSN Switch" if is_sw else "ECU Node")
                           + "<extra></extra>"),
            showlegend=False,
        ))

    mode_label = "🟢 TSN Scheduled Mode" if tsn_mode else "🔴 Standard Ethernet Mode"
    mode_col   = NEON_GREEN if tsn_mode else NEON_GOLD

    fig.add_annotation(
        text=mode_label, x=0.5, y=0.02, xref="paper", yref="paper",
        showarrow=False,
        font=dict(color=mode_col, size=13, family="monospace"),
        bgcolor=BG_DARK, bordercolor=mode_col, borderwidth=1, borderpad=6,
    )

    fig.update_layout(
        paper_bgcolor=BG_DARK,
        plot_bgcolor=BG_PLOT,
        height=480,
        margin=dict(l=10, r=10, t=55, b=10),
        title=dict(text="🛰️  Automotive ECU Network Topology – Packet Flow Simulation",
                   x=0.5, font=dict(size=14, color="#7dd3fc")),
        font=dict(family="monospace", color=TEXT_COL),
        showlegend=False,
    )
    fig.update_xaxes(visible=False, range=[-0.05, 1.05])
    fig.update_yaxes(visible=False, range=[-0.05, 1.05])
    return fig

def run_omnet_sim(tsn_mode, link_speed, packet_size, offered_load):
    nodes, edges = build_topology()
    flows = [
        [("ECU1\n(Engine)",       "SW1\n[TSN Switch]"),
         ("SW1\n[TSN Switch]",    "ECU5\n(Gateway)")],
        [("ECU2\n(Body Ctrl)",    "SW1\n[TSN Switch]"),
         ("SW1\n[TSN Switch]",    "SW2\n[Edge Switch]"),
         ("SW2\n[Edge Switch]",   "ECU3\n(ADAS)")],
        [("ECU6\n(Brake Ctrl)",   "SW1\n[TSN Switch]"),
         ("SW1\n[TSN Switch]",    "SW2\n[Edge Switch]"),
         ("SW2\n[Edge Switch]",   "ECU4\n(Infotainment)")],
    ]
    delay, jitter, loss, prio = simulate_network(link_speed, packet_size, offered_load, tsn_mode)

    STEPS = 30
    topo_ph    = st.empty()
    metrics_ph = st.empty()
    pbar       = st.progress(0)

    for step in range(STEPS):
        flow_idx    = step % len(flows)
        edge_in_flow = step % len(flows[flow_idx])
        active_path = [flows[flow_idx][edge_in_flow]]
        fig = draw_topology(nodes, edges, tsn_mode, active_path, step)
        topo_ph.plotly_chart(fig, use_container_width=True)

        with metrics_ph.container():
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("⏱ Delay",       f"{delay:.4f} ms",
                      delta="▼ Low" if tsn_mode else "▲ High",
                      delta_color="normal" if tsn_mode else "inverse")
            c2.metric("📊 Jitter",      f"{jitter:.4f} ms",
                      delta="▼ Stable" if tsn_mode else "▲ Variable",
                      delta_color="normal" if tsn_mode else "inverse")
            c3.metric("📦 Pkt Loss",    f"{loss:.4f} %",
                      delta="▼ Minimal" if tsn_mode else "▲ Higher",
                      delta_color="normal" if tsn_mode else "inverse")
            c4.metric("🏆 Determinism", f"{prio:.2f}",
                      delta="▲ High" if tsn_mode else "▼ Low",
                      delta_color="normal" if tsn_mode else "inverse")

        pbar.progress((step + 1) / STEPS)
        time.sleep(0.12 if tsn_mode else 0.18)

    pbar.empty()
    label = "✅ TSN Simulation complete!" if tsn_mode else "⚠️ Ethernet simulation complete."
    st.success(label)

# =========================================================
# HEADER
# =========================================================
st.title("🚗 TSN in Automotive Ethernet — Advanced Dashboard")
st.markdown(
    "<div style='text-align:center;margin:-12px 0 18px;font-family:Rajdhani,sans-serif;font-size:1.1rem;color:#4a6f8a;letter-spacing:1px'>"
    "IEEE 802.1Qbv · AUTOSAR Classic Platform · SOME/IP · E2E Protection · OMNeT++ Simulation"
    "</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='info-card'>Dataset-based analysis + <b>OMNeT++-inspired simulation</b> comparing "
    "<b>TSN-enabled</b> vs <b>Standard Ethernet</b> for in-vehicle networks — mapped onto the "
    "<b>AUTOSAR Classic Platform</b> stack (RTE, SOME/IP, PDU Router, E2E Profile).</div>",
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Parameters")
    link_speed      = st.slider("Link Speed (Mbps)",              10, 1000, 100, 10)
    window_size     = st.slider("Traffic Window (s)",             0.01, 1.00, 0.10, 0.01)
    sim_packet_size = st.slider("Simulation Packet Size (Bytes)", 64, 1500, 500, 50)
    offered_load    = st.slider("Offered Load (Mbps)",            1, link_speed, min(40, link_speed), 1)

    st.markdown("---")
    st.markdown("## 🤖 ML Intrusion Detection")
    st.markdown("<small style='color:#a0c4ff'>Unsupervised Anomaly Detection</small>", unsafe_allow_html=True)
    ml_contamination = st.slider("Contamination Rate (%)", 0.1, 10.0, 1.0, 0.1) / 100.0
    ml_dataset = st.radio("Analyze Network", ["TSN", "Ethernet"])
    st.markdown("---")
    st.markdown("## 🛰 OMNeT++ Sim")
    omnet_mode = st.radio("Mode", ["TSN Scheduled", "Standard Ethernet"])
    run_omnet  = st.button("▶  Run Live Simulation", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📑 Table of Contents")
    st.markdown("""
<small style='color:#7dd3fc;font-family:monospace'>
§1 — Dataset KPIs<br>
§2 — Multi-Metric Radar<br>
§3 — Traffic Over Time<br>
§4 — Distribution Analysis<br>
§5 — Protocol & Node Analysis<br>
§6 — Analytical Simulation<br>
§7 — OMNeT++ Network Simulation<br>
§8 — AUTOSAR SWC & PDU Router<br>
§9 — AUTOSAR E2E Protection<br>
§10 — AUTOSAR Layer Data Flow<br>
§11 — ML Anomaly Detection<br>
§12 — Final Conclusions
</small>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<small style='color:#4a6f8a'>IEEE 802.1Qbv | CBS | TAS<br>SOME/IP | AUTOSAR CP/AP<br>ISO 26262 | E2E Profile</small>",
                unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
try:
    tsn_df = load_csv(TSN_FILE)
    eth_df = load_csv(ETH_FILE)
except Exception as e:
    st.error(f"⛔ Error loading files: {e}")
    st.info("Keep this script and the CSV files in the same folder.")
    st.stop()

tsn_metrics = calculate_metrics(tsn_df, link_speed)
eth_metrics = calculate_metrics(eth_df, link_speed)

# ─────────────────────────────────────────────────────────
# SECTION 1 — KPI CARDS
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>📡 Section 1 — Dataset Performance KPIs</div>",
            unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.metric("TSN Throughput",    f"{tsn_metrics['Throughput (Mbps)']:.3f} Mbps",
          delta=f"{tsn_metrics['Throughput (Mbps)']-eth_metrics['Throughput (Mbps)']:.3f} vs ETH")
c2.metric("ETH Throughput",    f"{eth_metrics['Throughput (Mbps)']:.3f} Mbps")
c3.metric("TSN E2E Latency",   f"{tsn_metrics['E2E Latency (ms)']:.6f} ms",
          delta=f"{tsn_metrics['E2E Latency (ms)']-eth_metrics['E2E Latency (ms)']:.6f}",
          delta_color="inverse")
c4.metric("ETH E2E Latency",   f"{eth_metrics['E2E Latency (ms)']:.6f} ms")
c5, c6, c7, c8 = st.columns(4)
c5.metric("TSN Jitter",         f"{tsn_metrics['Jitter (ms)']:.6f} ms",
          delta=f"{tsn_metrics['Jitter (ms)']-eth_metrics['Jitter (ms)']:.6f}",
          delta_color="inverse")
c6.metric("ETH Jitter",         f"{eth_metrics['Jitter (ms)']:.6f} ms")
c7.metric("TSN RTE Queue",      f"{tsn_metrics['RTE Queue Delay (ms)']:.6f} ms")
c8.metric("ETH RTE Queue",      f"{eth_metrics['RTE Queue Delay (ms)']:.6f} ms")
c9, c10, c11, c12 = st.columns(4)
c9.metric("TSN Utilization",    f"{tsn_metrics['Utilization (%)']:.2f} %")
c10.metric("ETH Utilization",   f"{eth_metrics['Utilization (%)']:.2f} %")
c11.metric("TSN SOME/IP Ser.",  f"{tsn_metrics['SOME/IP Serialization (ms)']:.6f} ms")
c12.metric("ETH SOME/IP Ser.",  f"{eth_metrics['SOME/IP Serialization (ms)']:.6f} ms")

with st.expander("📋 Full Metrics Table"):
    summary_df = pd.DataFrame([tsn_metrics, eth_metrics], index=["TSN", "Ethernet"]).T
    st.dataframe(summary_df.style.format("{:.6f}"))

# ─────────────────────────────────────────────────────────
# SECTION 2 — RADAR + BAR COMPARISONS
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>🎯 Section 2 — Multi-Metric Comparison</div>",
            unsafe_allow_html=True)
rc, _ = st.columns([2, 1])
with rc:
    plotly_radar(tsn_metrics, eth_metrics)

left, right = st.columns(2)
with left:
    plotly_bar("Throughput Comparison",       "Mbps",
               tsn_metrics["Throughput (Mbps)"],         eth_metrics["Throughput (Mbps)"])
    plotly_bar("Jitter Comparison",           "ms",
               tsn_metrics["Jitter (ms)"],               eth_metrics["Jitter (ms)"])
with right:
    plotly_bar("E2E Latency Comparison (AUTOSAR)", "ms",
               tsn_metrics["E2E Latency (ms)"],          eth_metrics["E2E Latency (ms)"])
    plotly_bar("Avg Packet Length",           "Bytes",
               tsn_metrics["Avg Packet Length (Bytes)"], eth_metrics["Avg Packet Length (Bytes)"])

# ─────────────────────────────────────────────────────────
# SECTION 3 — TRAFFIC OVER TIME
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>📈 Section 3 — Traffic Behaviour Over Time</div>",
            unsafe_allow_html=True)
tsn_flow = traffic_over_time(tsn_df, window_size)
eth_flow = traffic_over_time(eth_df, window_size)
plotly_line(tsn_flow["Window"], tsn_flow["packets"],
            eth_flow["Window"], eth_flow["packets"],
            "Packets Over Time", "Packets", "Time Window (s)")
plotly_line(tsn_flow["Window"], tsn_flow["throughput_mbps"],
            eth_flow["Window"], eth_flow["throughput_mbps"],
            "Throughput Over Time", "Throughput (Mbps)", "Time Window (s)")

# ─────────────────────────────────────────────────────────
# SECTION 4 — DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>🔬 Section 4 — Distribution Analysis</div>",
            unsafe_allow_html=True)
dl, dr = st.columns(2)
with dl:
    plotly_histogram(tsn_df["InterArrival"]*1000, eth_df["InterArrival"]*1000,
                     "Inter-Arrival Time Distribution", "Inter-Arrival Time (ms)")
with dr:
    plotly_histogram(tsn_df["Length"], eth_df["Length"],
                     "Packet Length Distribution", "Packet Length (Bytes)")

# ─────────────────────────────────────────────────────────
# SECTION 5 — PROTOCOL & NODE ANALYSIS
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>🔌 Section 5 — Protocol & Node Analysis</div>",
            unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    tsn_p = tsn_df["Protocol"].value_counts().head(10)
    eth_p = eth_df["Protocol"].value_counts().head(10)
    all_p = list(dict.fromkeys(list(tsn_p.index) + list(eth_p.index)))
    plotly_horizontal_bar(all_p,
                          tsn_p.reindex(all_p, fill_value=0).tolist(),
                          eth_p.reindex(all_p, fill_value=0).tolist(),
                          "Top Protocol Comparison")
with col2:
    tsn_s = tsn_df["Source"].value_counts().head(10)
    eth_s = eth_df["Source"].value_counts().head(10)
    all_s = list(dict.fromkeys(list(tsn_s.index) + list(eth_s.index)))
    plotly_horizontal_bar(all_s,
                          tsn_s.reindex(all_s, fill_value=0).tolist(),
                          eth_s.reindex(all_s, fill_value=0).tolist(),
                          "Top Source Node Comparison")

# ─────────────────────────────────────────────────────────
# SECTION 6 — ANALYTICAL SIMULATION
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>⚡ Section 6 — Analytical Simulation (Queueing Model)</div>",
            unsafe_allow_html=True)
tsn_sd, tsn_sj, tsn_sl, tsn_prio = simulate_network(link_speed, sim_packet_size, offered_load, True)
eth_sd, eth_sj, eth_sl, eth_prio = simulate_network(link_speed, sim_packet_size, offered_load, False)

ss1, ss2, ss3, ss4 = st.columns(4)
ss1.metric("Simulated TSN Delay",    f"{tsn_sd:.4f} ms")
ss2.metric("Simulated ETH Delay",    f"{eth_sd:.4f} ms",
           delta=f"+{eth_sd-tsn_sd:.4f}", delta_color="inverse")
ss3.metric("Simulated TSN Jitter",   f"{tsn_sj:.4f} ms")
ss4.metric("Simulated ETH Jitter",   f"{eth_sj:.4f} ms",
           delta=f"+{eth_sj-tsn_sj:.4f}", delta_color="inverse")
ss5, ss6, ss7, ss8 = st.columns(4)
ss5.metric("TSN Packet Loss",        f"{tsn_sl:.4f} %")
ss6.metric("ETH Packet Loss",        f"{eth_sl:.4f} %",
           delta=f"+{eth_sl-tsn_sl:.4f}", delta_color="inverse")
ss7.metric("TSN Determinism Score",  f"{tsn_prio:.2f}")
ss8.metric("ETH Determinism Score",  f"{eth_prio:.2f}")

loads, td, ed, tj, ej, tl, el = simulation_curves(link_speed, sim_packet_size)
sl, sr = st.columns(2)
with sl:
    plotly_line(loads, td, loads, ed, "Delay vs Offered Load",  "Delay (ms)",  "Offered Load (Mbps)")
with sr:
    plotly_line(loads, tj, loads, ej, "Jitter vs Offered Load", "Jitter (ms)", "Offered Load (Mbps)")
plotly_line(loads, tl, loads, el, "Packet Loss vs Offered Load", "Packet Loss (%)", "Offered Load (Mbps)")

# ─────────────────────────────────────────────────────────
# SECTION 7 — OMNeT++ SIMULATION
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner omnet-banner'>🛰️ Section 7 — OMNeT++-Style Network Simulation</div>",
            unsafe_allow_html=True)
st.markdown("""
<div class='info-card'>
<b>OMNeT++ Inspired Live Simulation</b> — Watch packets traverse the automotive ECU network in real time.
Nodes represent <b>ECUs</b> (Engine, ADAS, Body, Brake, Gateway) and <b>TSN Switches</b>.
An animated packet dot moves along active routes while metrics update live.<br><br>
📌 Select <b>TSN Scheduled</b> or <b>Standard Ethernet</b> in the sidebar, then click <b>▶ Run Live Simulation</b>.
</div>
""", unsafe_allow_html=True)

nodes, edges = build_topology()
tsn_mode_flag = (omnet_mode == "TSN Scheduled")
static_fig = draw_topology(nodes, edges, tsn_mode=tsn_mode_flag)
st.plotly_chart(static_fig, use_container_width=True)

l1c, l2c, l3c = st.columns(3)
with l1c:
    st.markdown("<div class='info-card'>🟦 <b>ECU Nodes</b><br>Engine, ADAS, Body, Brake, Infotainment, Gateway</div>",
                unsafe_allow_html=True)
with l2c:
    st.markdown("<div class='info-card'>🟪 <b>TSN Switches</b><br>SW1 (Backbone) · SW2 (Edge Switch)</div>",
                unsafe_allow_html=True)
with l3c:
    st.markdown("<div class='info-card'>🟢 <b>Active Packet</b><br>Animated dot traverses active link</div>",
                unsafe_allow_html=True)

if run_omnet:
    st.markdown(
        f"<div class='info-card'>Running <b>{'TSN Scheduled' if tsn_mode_flag else 'Standard Ethernet'}</b> · "
        f"Link: <b>{link_speed} Mbps</b> · Load: <b>{offered_load} Mbps</b> · "
        f"Pkt: <b>{sim_packet_size} B</b></div>",
        unsafe_allow_html=True
    )
    run_omnet_sim(tsn_mode_flag, link_speed, sim_packet_size, offered_load)

# =========================================================
# AUTOSAR HELPERS
# =========================================================
def draw_swc_diagram():
    """SWC communication diagram showing RTE / Virtual Function Bus."""
    fig = go.Figure()
    # RTE backbone rectangle
    fig.add_shape(type="rect", x0=0.25, x1=0.75, y0=0.38, y1=0.62,
                  fillcolor="rgba(122,127,255,0.18)", line=dict(color=NEON_PURPLE, width=3))
    fig.add_annotation(x=0.5, y=0.50, text="<b>RTE / Virtual Function Bus (VFB)</b>",
                       showarrow=False, font=dict(color=NEON_PURPLE, size=14, family="Rajdhani"))

    # SWC nodes
    swcs = {
        "EngineCtrl\n(SWC-P)":   (0.10, 0.85),
        "BrakeCtrl\n(SWC-P)":    (0.10, 0.15),
        "ADAS_Proc\n(SWC-R)":    (0.90, 0.75),
        "BodyECU\n(SWC-R)":      (0.90, 0.50),
        "Gateway\n(SWC-R/P)":    (0.90, 0.25),
    }
    swc_colors = {
        "EngineCtrl\n(SWC-P)": NEON_GREEN,
        "BrakeCtrl\n(SWC-P)":  NEON_GOLD,
        "ADAS_Proc\n(SWC-R)":  NEON_BLUE,
        "BodyECU\n(SWC-R)":    NEON_PINK,
        "Gateway\n(SWC-R/P)":  NEON_PURPLE,
    }
    for name, (x, y) in swcs.items():
        col = swc_colors[name]
        fig.add_shape(type="rect", x0=x-0.09, x1=x+0.09, y0=y-0.10, y1=y+0.10,
                      fillcolor="rgba(0,0,0,0.4)", line=dict(color=col, width=2))
        fig.add_annotation(x=x, y=y, text=f"<b>{name.replace(chr(10), '<br>')}</b>",
                           showarrow=False, font=dict(color=col, size=10, family="monospace"))

    # Signal arrows (P-Port → RTE → R-Port)
    signals = [
        (0.19, 0.85, 0.25, 0.55, "EngineSpeed\n(SOME/IP)", NEON_GREEN),
        (0.19, 0.15, 0.25, 0.45, "WheelSpeed\n(CAN-TP)",   NEON_GOLD),
        (0.75, 0.55, 0.81, 0.75, "ObjectDetect\n(SOME/IP)", NEON_BLUE),
        (0.75, 0.50, 0.81, 0.50, "LightCmd\n(UDP/IP)",      NEON_PINK),
        (0.75, 0.45, 0.81, 0.25, "NM_State\n(SOME/IP-SD)",  NEON_PURPLE),
    ]
    for x0, y0, x1, y1, label, col in signals:
        fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                      line=dict(color=col, width=2, dash="dot"))
        mx, my = (x0+x1)/2, (y0+y1)/2
        fig.add_annotation(x=mx, y=my, text=label.replace("\n","<br>"),
                           showarrow=False, font=dict(color=col, size=9, family="monospace"),
                           bgcolor=BG_DARK, bordercolor=col, borderwidth=1, borderpad=3)

    fig.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_PLOT, height=440,
                      margin=dict(l=10, r=10, t=55, b=10),
                      title=dict(text="⚡ AUTOSAR SWC Communication — Virtual Function Bus (VFB)",
                                 x=0.5, font=dict(size=15, color="#7dd3fc", family="Rajdhani")),
                      showlegend=False)
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def draw_pdu_table():
    """PDU Router & SOME/IP signal routing table."""
    headers = ["Signal Name", "Source SWC", "Target SWC", "PDU ID", "COM Stack", "UDP Port", "Priority (TSN)"]
    rows = [
        ["EngineSpeed",  "EngineCtrl",  "ADAS_Proc", "0x001", "SOME/IP",   "5000", "Class A (Crit.)"],
        ["WheelSpeed",   "BrakeCtrl",   "BodyECU",   "0x002", "CAN-TP",    "5001", "Class A (Crit.)"],
        ["ObjectDetect", "ADAS_Proc",   "Gateway",   "0x003", "SOME/IP",   "5002", "Class B (Safety)"],
        ["TailLightCmd", "BodyECU",     "EngineCtrl","0x004", "UDP/IP",    "5003", "Class C (Best-Effort)"],
        ["NM_State",     "Gateway",     "All SWCs",  "0x005", "SOME/IP-SD","5004", "Class C (Best-Effort)"],
        ["E2E_Heartbeat","EngineCtrl",  "BrakeCtrl", "0x006", "SOME/IP",  "5005",  "Class A (Crit.)"],
    ]
    row_colors = [
        [NEON_GREEN]*7, [NEON_GOLD]*7, [NEON_BLUE]*7,
        [NEON_PINK]*7, [NEON_PURPLE]*7, [NEON_GREEN]*7
    ]
    col_vals = [list(col) for col in zip(*rows)]
    fig = go.Figure(go.Table(
        header=dict(values=[f"<b>{h}</b>" for h in headers],
                    fill_color="#0d1f3e", font=dict(color=NEON_BLUE, size=12, family="Rajdhani"),
                    align="center", line_color="#1a3a5c", height=35),
        cells=dict(values=col_vals,
                   fill_color=[[BG_PLOT]*len(rows)]*len(headers),
                   font=dict(color=[rc[::-1] for rc in zip(*row_colors)], size=11, family="monospace"),
                   align="center", line_color="#1a3a5c", height=30)
    ))
    fig.update_layout(paper_bgcolor=BG_DARK, plot_bgcolor=BG_PLOT, height=300,
                      margin=dict(l=10, r=10, t=55, b=10),
                      title=dict(text="📋 AUTOSAR COM/PDU Router — SOME/IP Signal Routing Table",
                                 x=0.5, font=dict(size=15, color="#7dd3fc", family="Rajdhani")))
    return fig


def draw_e2e_profile():
    """AUTOSAR E2E Protection Profile overhead comparison chart."""
    profiles = ["E2E Profile 1\n(CRC-8, 1B)", "E2E Profile 2\n(CRC-32, 4B)", "E2E Profile 5\n(CRC-32P4, 4B)"]
    crc_bytes    = [1, 4, 4]
    counter_byte = [1, 1, 2]
    dataid_byte  = [2, 2, 4]
    # Overhead on TSN (lower base latency) vs Ethernet (higher base latency)
    tsn_base  = [tsn_metrics["SOME/IP Serialization (ms)"] * (1 + x * 0.05) for x in [1, 4, 6]]
    eth_base  = [eth_metrics["SOME/IP Serialization (ms)"] * (1 + x * 0.07) for x in [1, 4, 6]]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="CRC Overhead (Bytes)",    x=profiles, y=crc_bytes,
                         marker_color=NEON_BLUE,   opacity=0.85))
    fig.add_trace(go.Bar(name="Counter Field (Bytes)",   x=profiles, y=counter_byte,
                         marker_color=NEON_PURPLE, opacity=0.85))
    fig.add_trace(go.Bar(name="DataID Field (Bytes)",    x=profiles, y=dataid_byte,
                         marker_color=NEON_PINK,   opacity=0.85))
    fig.add_trace(go.Scatter(name="TSN Overhead (ms)",   x=profiles, y=tsn_base,
                             mode="lines+markers", line=dict(color=NEON_GREEN, width=3),
                             marker=dict(size=10, color=NEON_GREEN, symbol="diamond"),
                             yaxis="y2"))
    fig.add_trace(go.Scatter(name="ETH Overhead (ms)",   x=profiles, y=eth_base,
                             mode="lines+markers", line=dict(color=NEON_GOLD, width=3, dash="dash"),
                             marker=dict(size=10, color=NEON_GOLD, symbol="circle"),
                             yaxis="y2"))
    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PLOT, barmode="stack", height=380,
        margin=dict(l=10, r=60, t=55, b=10),
        title=dict(text="🔒 AUTOSAR E2E Protection Profile — CRC Overhead & Latency Impact",
                   x=0.5, font=dict(size=15, color="#7dd3fc", family="Rajdhani")),
        xaxis=dict(gridcolor=GRID_COL, color=TEXT_COL, tickfont=dict(family="monospace")),
        yaxis=dict(title="Bytes per PDU", gridcolor=GRID_COL, color=TEXT_COL),
        yaxis2=dict(title="Latency Overhead (ms)", overlaying="y", side="right",
                    gridcolor=GRID_COL, color=TEXT_COL),
        legend=dict(bgcolor=BG_DARK, bordercolor=GRID_COL, font=dict(color=TEXT_COL, size=10)),
        font=dict(family="monospace", color=TEXT_COL),
    )
    return fig

# ─────────────────────────────────────────────────────────
# SECTION 8 — AUTOSAR SWC & PDU
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>⚡ Section 8 — AUTOSAR SWC Communication & PDU Routing</div>",
            unsafe_allow_html=True)
st.markdown("<div class='info-card'>The <b>AUTOSAR Classic Platform</b> uses Software Components (SWCs) connected through the <b>Runtime Environment (RTE)</b> and <b>Virtual Function Bus (VFB)</b>. Signals are routed via the <b>COM / PDU Router</b> using SOME/IP, CAN-TP, or raw UDP/IP depending on the ECU domain.</div>",
            unsafe_allow_html=True)

sw_col1, sw_col2 = st.columns([3, 2])
with sw_col1:
    st.plotly_chart(draw_swc_diagram(), use_container_width=True)
with sw_col2:
    st.markdown("<div class='info-card'><b>SWC Port Types</b><br><br>"
                "🟢 <b>P-Port (Provided)</b>: SWC sends a signal<br>"
                "🔵 <b>R-Port (Required)</b>: SWC receives a signal<br>"
                "🟣 <b>P/R-Port (Gateway)</b>: Bridges ECU domains<br><br>"
                "<b>RTE Functions Used</b><br>"
                "📤 <code>Rte_Write_EngineSp()</code><br>"
                "📥 <code>Rte_Read_WheelSp()</code><br>"
                "📢 <code>Rte_Send_NM_State()</code></div>",
                unsafe_allow_html=True)

st.plotly_chart(draw_pdu_table(), use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION 9 — E2E PROTECTION
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>🔒 Section 9 — AUTOSAR E2E Protection Profiles</div>",
            unsafe_allow_html=True)
st.markdown("<div class='info-card'>AUTOSAR E2E (End-to-End) protection adds <b>CRC</b>, <b>Counter</b>, and <b>DataID</b> fields to each PDU for functional safety compliance per <b>ISO 26262</b>. TSN's deterministic scheduling reduces the latency overhead of E2E checking compared to standard Ethernet.</div>",
            unsafe_allow_html=True)
st.plotly_chart(draw_e2e_profile(), use_container_width=True)
e2e_c1, e2e_c2, e2e_c3 = st.columns(3)
e2e_c1.metric("E2E Profile 1  CRC-8",   "1 Byte CRC",   "ISO 26262 ASIL-B")
e2e_c2.metric("E2E Profile 2  CRC-32",  "4 Byte CRC",   "ISO 26262 ASIL-D")
e2e_c3.metric("E2E Profile 5  CRC-32P4","6 Byte header","Highest Safety Level")

def draw_autosar_architecture(step=0, is_tx=True):
    fig = go.Figure()
    layers = [
        ("Application Layer (SWCs)",        0.80, 0.95, NEON_PINK,  "Application Features: ADAS, Body, etc."),
        ("Runtime Environment (RTE)",       0.65, 0.78, NEON_PURPLE, "VFB Abstraction & Component Communication"),
        ("Services Layer (COM, PduR)",      0.50, 0.63, NEON_BLUE,  "Diagnostic (DoIP), SOME/IP, Network Mgmt"),
        ("ECU Abstraction (EthIf, TcpIp)",  0.35, 0.48, NEON_BLUE,  "TCP/IP Stack, Socket Adapter, EthIf"),
        ("MCAL (Eth, EthTrcv)",             0.20, 0.33, NEON_BLUE,  "Ethernet Driver, Transceiver Driver"),
        ("Hardware (Microcontroller / PHY)",0.05, 0.18, "#888888",   "Physical Layer (100BASE-T1, 1000BASE-T1)"),
    ]
    
    # Draw layers
    for name, y0, y1, color, desc in layers:
        fill_col = color.replace(")", ", 0.15)").replace("rgb", "rgba") if "rgb" in color else "rgba(14,246,255,0.15)" if color==NEON_BLUE else "rgba(255,110,247,0.15)" if color==NEON_PINK else "rgba(180,79,255,0.15)" if color==NEON_PURPLE else "rgba(136,136,136,0.15)"
        fig.add_shape(type="rect", x0=0.1, x1=0.9, y0=y0, y1=y1,
                      fillcolor=fill_col, line=dict(color=color, width=2))
        fig.add_annotation(x=0.5, y=y0 + (y1-y0)*0.65, text=f"<b>{name}</b>",
                           showarrow=False, font=dict(color=color, size=15, family="Rajdhani"))
        fig.add_annotation(x=0.5, y=y0 + (y1-y0)*0.25, text=desc,
                           showarrow=False, font=dict(color=TEXT_COL, size=11, family="monospace"))

    # Animated Packet Dot
    if step >= 0:
        total_steps = 20
        t = (step % total_steps) / float(total_steps - 1)
        
        # Path: top of app layer (0.95) to bottom of hardware (0.05)
        top_y = 0.95
        bot_y = 0.05
        
        if is_tx:
            # Transmit: Top to Bottom
            py = top_y - t * (top_y - bot_y)
            packet_color = NEON_GREEN
        else:
            # Receive: Bottom to Top
            py = bot_y + t * (top_y - bot_y)
            packet_color = NEON_GOLD
            
        fig.add_trace(go.Scatter(
            x=[0.15], y=[py],  # Fixed x-position on the left side
            mode="markers",
            marker=dict(size=18, color=packet_color, symbol="diamond",
                        line=dict(color="#ffffff", width=2)),
            showlegend=False, hoverinfo="skip"
        ))
        
        # Dynamic label next to the packet
        flow_label = "TX (Send)" if is_tx else "RX (Receive)"
        fig.add_annotation(
            x=0.20, y=py, text=f"<b>{flow_label} Packet</b>",
            showarrow=False, font=dict(color=packet_color, size=12, family="monospace"),
            xanchor="left"
        )

    direction_title = "Transmission (TX) Flow" if is_tx else "Reception (RX) Flow"
    title_text = f"⚙️ AUTOSAR Layered Architecture — <b>{direction_title}</b>" if step >= 0 else "⚙️ AUTOSAR Classic Platform — Ethernet & TSN Stack Architecture"

    fig.update_layout(
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_PLOT, height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text=title_text, x=0.5, font=dict(size=16, color="#7dd3fc", family="Rajdhani")),
        showlegend=False
    )
    fig.update_xaxes(visible=False, range=[0, 1])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig

# ─────────────────────────────────────────────────────────
# SECTION 8 — AUTOSAR ARCHITECTURE
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>⚙️ Section 10 — AUTOSAR Layer Data Flow (TX/RX Animation)</div>",
            unsafe_allow_html=True)
st.markdown("<div class='info-card'>Watch how data packets flow vertically through the <b>AUTOSAR Classic Platform</b> stack during Transmission (TX) and Reception (RX).</div>", unsafe_allow_html=True)

col_auto1, col_auto2 = st.columns([1, 1])
with col_auto1:
    run_tx = st.button("▶ Animate TX (Transmit Down)", use_container_width=True)
with col_auto2:
    run_rx = st.button("▶ Animate RX (Receive Up)", use_container_width=True)

autosar_ph = st.empty()

if run_tx or run_rx:
    is_tx_flow = bool(run_tx)
    for s in range(20):
        fig_layer = draw_autosar_architecture(step=s, is_tx=is_tx_flow)
        autosar_ph.plotly_chart(fig_layer, use_container_width=True)
        time.sleep(0.15)
else:
    # Static view before clicking
    autosar_ph.plotly_chart(draw_autosar_architecture(step=-1), use_container_width=True)

# ─────────────────────────────────────────────────────────
# SECTION 11 — ML ANOMALY DETECTION
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>🤖 Section 11 — ML Intrusion & Anomaly Detection</div>",
            unsafe_allow_html=True)
st.markdown("<div class='info-card'>Using <b>Isolation Forest</b> (Unsupervised Machine Learning) to detect abnormal packet behavior. Anomalies could indicate <b>Network Intrusions</b> (e.g., DoS, Spoofing) or <b>Babbling Idiot</b> hardware faults.</div>",
            unsafe_allow_html=True)

ml_df = tsn_df.copy() if ml_dataset == "TSN" else eth_df.copy()
ml_features = ml_df[["InterArrival", "Length"]].dropna()

if len(ml_features) > 10:
    # Train Model
    clf = IsolationForest(contamination=ml_contamination, random_state=42)
    ml_df["Anomaly"] = clf.fit_predict(ml_features)
    
    # 1 is Normal, -1 is Anomaly
    normal_data = ml_df[ml_df["Anomaly"] == 1]
    anomaly_data = ml_df[ml_df["Anomaly"] == -1]
    
    num_anomalies = len(anomaly_data)
    total_packets = len(ml_df)
    anomaly_pct = (num_anomalies / total_packets) * 100

    col_ml1, col_ml2, col_ml3 = st.columns(3)
    col_ml1.metric(f"Total {ml_dataset} Packets Analyzed", total_packets)
    col_ml2.metric("Detected Anomalies", num_anomalies, delta="Flagged" if num_anomalies > 0 else "Clean", delta_color="inverse" if num_anomalies > 0 else "normal")
    col_ml3.metric("Anomaly Rate", f"{anomaly_pct:.2f}%")

    fig_ml = go.Figure()
    fig_ml.add_trace(go.Scatter(
        x=normal_data["InterArrival"] * 1000, 
        y=normal_data["Length"],
        mode="markers",
        name="Normal Traffic",
        marker=dict(color=NEON_BLUE, size=6, opacity=0.6)
    ))
    fig_ml.add_trace(go.Scatter(
        x=anomaly_data["InterArrival"] * 1000, 
        y=anomaly_data["Length"],
        mode="markers",
        name="Anomalous Packets",
        marker=dict(color="#ff3333", size=10, symbol="x", line=dict(color="#ffffff", width=1))
    ))
    
    dark_theme(fig_ml, title=f"Isolation Forest Anomaly Map — {ml_dataset} Data", 
               xlabel="Inter-Arrival Time (ms)", ylabel="Packet Length (Bytes)", height=450)
    st.plotly_chart(fig_ml, use_container_width=True)
else:
    st.warning("Not enough data points to run Anomaly Detection.")

# ─────────────────────────────────────────────────────────
# SECTION 12 — CONCLUSIONS
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-banner'>✅ Section 12 — Final Conclusions</div>",
            unsafe_allow_html=True)
obs = []
if tsn_metrics["E2E Latency (ms)"] < eth_metrics["E2E Latency (ms)"]:
    obs.append("📉 Dataset confirms TSN achieves <b>lower E2E Latency</b> through the AUTOSAR stack than standard Ethernet.")
else:
    obs.append("📉 Dataset shows Ethernet has lower latency in this trace — possible low burst scenario.")
if tsn_metrics["Jitter (ms)"] < eth_metrics["Jitter (ms)"]:
    obs.append("📊 TSN exhibits <b>lower jitter</b> — highly suitable for RTE/VFB deterministic routing and ADAS safety-critical traffic.")
else:
    obs.append("📊 Ethernet shows lower jitter in this capture.")
if tsn_sd < eth_sd:
    obs.append("⚡ Simulation confirms TSN <b>reduces RTE Queue Delay</b> under increasing offered load.")
else:
    obs.append("⚡ Simulation shows Ethernet delay is lower under the selected parameters.")
if tsn_sj < eth_sj:
    obs.append("🔒 TSN provides <b>superior timing stability</b> — critical for SOME/IP Serialization and CAN-FD routing.")
else:
    obs.append("🔒 Ethernet shows lower simulated jitter under current configuration.")
if tsn_sl < eth_sl:
    obs.append("📦 TSN demonstrates <b>lower packet loss</b> at the MCAL layer under load — improving network reliability.")
else:
    obs.append("📦 Ethernet shows lower packet loss under the chosen condition.")
obs.append(
    f"🏆 TSN Determinism Score: <b>{tsn_prio:.1f}</b> vs Ethernet: <b>{eth_prio:.1f}</b> — "
    "IEEE 802.1Qbv time-aware scheduling provides deterministic QoS unavailable in standard Ethernet."
)
for text in obs:
    st.markdown(f"<div class='obs-item'>{text}</div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<div style='text-align:center;padding:16px 0;font-family:monospace'>
    <span style='color:#0ef6ff;font-size:1rem;letter-spacing:2px'>🚗 TSN AUTOMOTIVE ETHERNET DASHBOARD</span><br><br>
    <span style='color:#4a6f8a;font-size:0.78rem'>
    IEEE 802.1Qbv | Time-Aware Shaper (TAS) | Credit-Based Shaper (CBS)<br>
    AUTOSAR Classic Platform | SOME/IP | PDU Router | E2E Profile | ISO 26262<br>
    OMNeT++ Simulation | Plotly 6 | Streamlit | Scikit-Learn ML
    </span>
</div>""", unsafe_allow_html=True)