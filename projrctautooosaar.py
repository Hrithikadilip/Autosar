import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="TSN vs Ethernet Simulation Dashboard", layout="wide")

# =========================================================
# FILE PATHS
# =========================================================
TSN_FILE = "driving 1 orginal.csv"
ETH_FILE = "driving 1 injected.csv"

# If you want to use second pair instead, change to:
# TSN_FILE = "driving 2 original.csv"
# ETH_FILE = "driving 2 injected.csv"

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

    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")
    df["Length"] = pd.to_numeric(df["Length"], errors="coerce")
    df = df.dropna(subset=["Time", "Length"]).sort_values("Time").reset_index(drop=True)

    df["InterArrival"] = df["Time"].diff().fillna(0)
    df["Bits"] = df["Length"] * 8

    return df

# =========================================================
# METRICS
# =========================================================
def calculate_metrics(df, link_speed_mbps=100):
    duration = max(df["Time"].max() - df["Time"].min(), 1e-9)
    total_packets = len(df)
    total_bytes = df["Length"].sum()

    throughput_mbps = (total_bytes * 8) / duration / 1e6
    mean_interarrival_ms = df["InterArrival"].mean() * 1000
    jitter_ms = df["InterArrival"].std() * 1000
    avg_pkt_len = df["Length"].mean()

    serialization_ms = ((avg_pkt_len * 8) / (link_speed_mbps * 1e6)) * 1000
    utilization = min(throughput_mbps / max(link_speed_mbps, 1e-9), 0.95)
    queue_delay_ms = serialization_ms * (utilization / max(1 - utilization, 1e-6))
    total_delay_ms = serialization_ms + queue_delay_ms

    return {
        "Packets": total_packets,
        "Duration (s)": duration,
        "Throughput (Mbps)": throughput_mbps,
        "Mean InterArrival (ms)": mean_interarrival_ms,
        "Jitter (ms)": jitter_ms,
        "Average Packet Length (Bytes)": avg_pkt_len,
        "Serialization Delay (ms)": serialization_ms,
        "Queue Delay (ms)": queue_delay_ms,
        "Estimated End-to-End Delay (ms)": total_delay_ms,
        "Utilization (%)": utilization * 100,
    }

# =========================================================
# TRAFFIC OVER TIME
# =========================================================
def traffic_over_time(df, window=0.1):
    start = df["Time"].min()
    end = df["Time"].max() + window
    bins = np.arange(start, end + window, window)

    if len(bins) < 2:
        bins = np.array([start, start + window])

    temp = df.copy()
    temp["Window"] = pd.cut(temp["Time"], bins=bins, labels=bins[:-1], include_lowest=True)

    grouped = temp.groupby("Window", observed=False).agg(
        packets=("Length", "count"),
        bytes=("Length", "sum")
    ).reset_index()

    grouped["Window"] = grouped["Window"].astype(float)
    grouped["throughput_mbps"] = (grouped["bytes"] * 8) / window / 1e6
    return grouped

# =========================================================
# SIMULATION MODEL
# =========================================================
def simulate_network(link_speed_mbps=100, packet_size_bytes=500, offered_load=20, tsn=True):
    """
    Simple queueing simulation:
    Ethernet: larger queueing delay under load
    TSN: lower queueing due to scheduled traffic
    """
    serialization_ms = ((packet_size_bytes * 8) / (link_speed_mbps * 1e6)) * 1000
    rho = min(offered_load / max(link_speed_mbps, 1e-9), 0.95)

    if tsn:
        queue_delay_ms = serialization_ms * (0.25 * rho / max(1 - rho, 1e-6))
        jitter_ms = 0.15 + 0.6 * rho
        packet_loss = max(0, 0.2 * rho)
    else:
        queue_delay_ms = serialization_ms * (1.2 * rho / max(1 - rho, 1e-6))
        jitter_ms = 0.5 + 2.5 * rho
        packet_loss = max(0, 1.5 * rho)

    total_delay_ms = serialization_ms + queue_delay_ms
    return total_delay_ms, jitter_ms, packet_loss

def simulation_curve(link_speed_mbps, packet_size_bytes):
    loads = np.arange(10, 100, 5)
    tsn_delay, eth_delay = [], []
    tsn_jitter, eth_jitter = [], []
    tsn_loss, eth_loss = [], []

    for load in loads:
        d1, j1, l1 = simulate_network(link_speed_mbps, packet_size_bytes, load, tsn=True)
        d2, j2, l2 = simulate_network(link_speed_mbps, packet_size_bytes, load, tsn=False)

        tsn_delay.append(d1)
        eth_delay.append(d2)
        tsn_jitter.append(j1)
        eth_jitter.append(j2)
        tsn_loss.append(l1)
        eth_loss.append(l2)

    return loads, tsn_delay, eth_delay, tsn_jitter, eth_jitter, tsn_loss, eth_loss

# =========================================================
# PLOT HELPERS
# =========================================================
def bar_compare(title, y_label, tsn_val, eth_val):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["TSN", "Ethernet"], [tsn_val, eth_val])
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

def line_compare(x1, y1, x2, y2, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x1, y1, linewidth=2, label="TSN")
    ax.plot(x2, y2, linewidth=2, label="Ethernet")
    ax.set_title(title)
    ax.set_xlabel("Time / Load")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

def histogram_compare(data1, data2, title, xlabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(data1, bins=50, alpha=0.6, label="TSN")
    ax.hist(data2, bins=50, alpha=0.6, label="Ethernet")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# =========================================================
# APP TITLE
# =========================================================
st.title("TSN vs Ethernet Analysis and Simulation Dashboard")
st.write("This dashboard combines dataset-based traffic analysis and simulation-based network performance comparison.")

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Simulation Parameters")
link_speed = st.sidebar.slider("Link Speed (Mbps)", 10, 1000, 100, 10)
window_size = st.sidebar.slider("Traffic Window (s)", 0.01, 1.00, 0.10, 0.01)
sim_packet_size = st.sidebar.slider("Simulation Packet Size (Bytes)", 64, 1500, 500, 50)
offered_load = st.sidebar.slider("Offered Load (Mbps)", 1, link_speed, min(40, link_speed), 1)

# =========================================================
# LOAD DATA
# =========================================================
try:
    tsn_df = load_csv(TSN_FILE)
    eth_df = load_csv(ETH_FILE)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Keep CSV files in the same folder as this Python file.")
    st.stop()

# =========================================================
# DATASET ANALYSIS
# =========================================================
tsn_metrics = calculate_metrics(tsn_df, link_speed)
eth_metrics = calculate_metrics(eth_df, link_speed)

st.subheader("1. Dataset-Based Performance Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("TSN Throughput", f"{tsn_metrics['Throughput (Mbps)']:.3f} Mbps")
c2.metric("Ethernet Throughput", f"{eth_metrics['Throughput (Mbps)']:.3f} Mbps")
c3.metric("TSN Delay", f"{tsn_metrics['Estimated End-to-End Delay (ms)']:.6f} ms")
c4.metric("Ethernet Delay", f"{eth_metrics['Estimated End-to-End Delay (ms)']:.6f} ms")

c5, c6, c7, c8 = st.columns(4)
c5.metric("TSN Jitter", f"{tsn_metrics['Jitter (ms)']:.6f} ms")
c6.metric("Ethernet Jitter", f"{eth_metrics['Jitter (ms)']:.6f} ms")
c7.metric("TSN Utilization", f"{tsn_metrics['Utilization (%)']:.2f} %")
c8.metric("Ethernet Utilization", f"{eth_metrics['Utilization (%)']:.2f} %")

summary_df = pd.DataFrame([tsn_metrics, eth_metrics], index=["TSN", "Ethernet"]).T
st.dataframe(summary_df.style.format("{:.6f}"))

# =========================================================
# BAR COMPARISONS
# =========================================================
st.subheader("2. Direct Comparison Charts")
left, right = st.columns(2)

with left:
    bar_compare("Throughput Comparison", "Mbps",
                tsn_metrics["Throughput (Mbps)"],
                eth_metrics["Throughput (Mbps)"])

    bar_compare("Jitter Comparison", "ms",
                tsn_metrics["Jitter (ms)"],
                eth_metrics["Jitter (ms)"])

with right:
    bar_compare("Delay Comparison", "ms",
                tsn_metrics["Estimated End-to-End Delay (ms)"],
                eth_metrics["Estimated End-to-End Delay (ms)"])

    bar_compare("Average Packet Length", "Bytes",
                tsn_metrics["Average Packet Length (Bytes)"],
                eth_metrics["Average Packet Length (Bytes)"])

# =========================================================
# TRAFFIC TREND
# =========================================================
st.subheader("3. Traffic Trend Over Time")
tsn_flow = traffic_over_time(tsn_df, window_size)
eth_flow = traffic_over_time(eth_df, window_size)

line_compare(tsn_flow["Window"], tsn_flow["packets"],
             eth_flow["Window"], eth_flow["packets"],
             "Packets Over Time", "Packets")

line_compare(tsn_flow["Window"], tsn_flow["throughput_mbps"],
             eth_flow["Window"], eth_flow["throughput_mbps"],
             "Throughput Over Time", "Throughput (Mbps)")

# =========================================================
# DISTRIBUTION ANALYSIS
# =========================================================
st.subheader("4. Distribution Analysis")

histogram_compare(tsn_df["InterArrival"] * 1000,
                  eth_df["InterArrival"] * 1000,
                  "Inter-Arrival Time Distribution",
                  "Inter-Arrival Time (ms)")

histogram_compare(tsn_df["Length"],
                  eth_df["Length"],
                  "Packet Length Distribution",
                  "Packet Length (Bytes)")

# =========================================================
# PROTOCOL ANALYSIS
# =========================================================
st.subheader("5. Protocol and Source Analysis")

col1, col2 = st.columns(2)

with col1:
    tsn_protocols = tsn_df["Protocol"].value_counts().head(10)
    eth_protocols = eth_df["Protocol"].value_counts().head(10)
    all_protocols = list(dict.fromkeys(list(tsn_protocols.index) + list(eth_protocols.index)))
    tsn_vals = tsn_protocols.reindex(all_protocols, fill_value=0)
    eth_vals = eth_protocols.reindex(all_protocols, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(all_protocols))
    h = 0.4
    ax.barh(y - h/2, tsn_vals.values, height=h, label="TSN")
    ax.barh(y + h/2, eth_vals.values, height=h, label="Ethernet")
    ax.set_yticks(y)
    ax.set_yticklabels(all_protocols)
    ax.set_xlabel("Count")
    ax.set_title("Top Protocol Comparison")
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)

with col2:
    tsn_sources = tsn_df["Source"].value_counts().head(10)
    eth_sources = eth_df["Source"].value_counts().head(10)
    all_sources = list(dict.fromkeys(list(tsn_sources.index) + list(eth_sources.index)))
    tsn_vals = tsn_sources.reindex(all_sources, fill_value=0)
    eth_vals = eth_sources.reindex(all_sources, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(all_sources))
    h = 0.4
    ax.barh(y - h/2, tsn_vals.values, height=h, label="TSN")
    ax.barh(y + h/2, eth_vals.values, height=h, label="Ethernet")
    ax.set_yticks(y)
    ax.set_yticklabels(all_sources)
    ax.set_xlabel("Count")
    ax.set_title("Top Source Node Comparison")
    ax.legend()
    ax.invert_yaxis()
    st.pyplot(fig)

# =========================================================
# SIMULATION SECTION
# =========================================================
st.subheader("6. Simulation Section")

tsn_sim_delay, tsn_sim_jitter, tsn_sim_loss = simulate_network(
    link_speed_mbps=link_speed,
    packet_size_bytes=sim_packet_size,
    offered_load=offered_load,
    tsn=True
)

eth_sim_delay, eth_sim_jitter, eth_sim_loss = simulate_network(
    link_speed_mbps=link_speed,
    packet_size_bytes=sim_packet_size,
    offered_load=offered_load,
    tsn=False
)

s1, s2, s3 = st.columns(3)
s1.metric("Simulated TSN Delay", f"{tsn_sim_delay:.4f} ms")
s2.metric("Simulated Ethernet Delay", f"{eth_sim_delay:.4f} ms")
s3.metric("Current Offered Load", f"{offered_load} Mbps")

s4, s5, s6 = st.columns(3)
s4.metric("Simulated TSN Jitter", f"{tsn_sim_jitter:.4f} ms")
s5.metric("Simulated Ethernet Jitter", f"{eth_sim_jitter:.4f} ms")
s6.metric("Packet Size", f"{sim_packet_size} Bytes")

loads, tsn_delay, eth_delay, tsn_jitter, eth_jitter, tsn_loss, eth_loss = simulation_curve(
    link_speed, sim_packet_size
)

line_compare(loads, tsn_delay, loads, eth_delay,
             "Simulated End-to-End Delay vs Offered Load", "Delay (ms)")

line_compare(loads, tsn_jitter, loads, eth_jitter,
             "Simulated Jitter vs Offered Load", "Jitter (ms)")

line_compare(loads, tsn_loss, loads, eth_loss,
             "Simulated Packet Loss vs Offered Load", "Packet Loss (%)")

# =========================================================
# FINAL OBSERVATIONS
# =========================================================
st.subheader("7. Final Analysis and Conclusion")

observations = []

if tsn_metrics["Estimated End-to-End Delay (ms)"] < eth_metrics["Estimated End-to-End Delay (ms)"]:
    observations.append("From the dataset analysis, TSN shows lower delay than standard Ethernet.")
else:
    observations.append("From the dataset analysis, Ethernet shows lower delay than TSN in this trace.")

if tsn_metrics["Jitter (ms)"] < eth_metrics["Jitter (ms)"]:
    observations.append("TSN has lower jitter, which is better for deterministic real-time traffic.")
else:
    observations.append("Ethernet has lower jitter in this capture.")

if tsn_sim_delay < eth_sim_delay:
    observations.append("From the simulation, TSN maintains lower delay as offered load increases.")
else:
    observations.append("From the simulation, Ethernet delay appears lower under this selected load.")

if tsn_sim_jitter < eth_sim_jitter:
    observations.append("Simulation confirms that TSN provides better timing stability than Ethernet.")
else:
    observations.append("Simulation shows Ethernet jitter lower under this selected condition.")

if tsn_sim_loss < eth_sim_loss:
    observations.append("TSN is more reliable under high traffic due to lower simulated packet loss.")
else:
    observations.append("Ethernet shows lower packet loss under the chosen condition.")

for i, text in enumerate(observations, start=1):
    st.write(f"{i}. {text}")

st.success("Dashboard loaded successfully.")