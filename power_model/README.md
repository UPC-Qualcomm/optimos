# Power Model for AstraSim

Estimates energy consumption of distributed-training workloads simulated in AstraSim.
It supports **4 Low-Power-Mode (LPM) configurations** and works with any network topology
(FoldedClos, Jellyfish, Dragonfly, …) by reading a nodemap file.

---

## File Overview

| File | Role |
|------|------|
| `analyze_power.py` | **CLI entry point** – run this |
| `power_config.py` | Hardware parameters + LPM flags (`PowerConfig` dataclass) |
| `power_model.py` | Top-level `PowerModel`: combines compute + network |
| `compute_parser.py` | Parses AstraSim `.log` → `ComputeStats` (per-NPU timing) |
| `compute_model.py` | Per-NPU compute power and energy |
| `network_parser.py` | Parses `network_statistics.csv` + nodemap → `NetworkStats` |
| `nodemap_parser.py` | Parses any nodemap JSON → `TopologyInfo` (generic prefix map) |
| `network_model.py` | Network-side power (links + per-switch-type models) |
| `default_config.json` | H100 hardware defaults |
| `a100_config.json` | A100 hardware defaults |
| `test_json_config.py` | Config loading smoke-tests |

---

## Quick Start

```bash
# Single mode
python analyze_power.py \
  --compute  path/to/astrasim.log \
  --network  path/to/network_statistics.csv \
  --nodemap  path/to/nodemap.json \
  --mode     D \
  --config   default_config.json

# Compare all 4 modes
python analyze_power.py \
  --compute path/to/astrasim.log \
  --network path/to/network_statistics.csv \
  --nodemap path/to/nodemap.json \
  --compare

# Save results to JSON
python analyze_power.py \
  --compute path/to/astrasim.log \
  --network path/to/network_statistics.csv \
  --nodemap path/to/nodemap.json \
  --mode D \
  --output results.json
```

---

## The 4 LPM Modes

| Mode | Compute LPM | Comm LPM | Description |
|------|:-----------:|:--------:|-------------|
| A | ✗ | ✗ | Baseline – no power saving |
| B | ✓ | ✗ | GPU sleeps during communication phases |
| C | ✗ | ✓ | Network switches/links sleep when idle |
| D | ✓ | ✓ | Both LPMs active (maximum savings) |

---

## How the Power Model Works

### 1 · Inputs

| Input | Source |
|-------|--------|
| AstraSim compute log | AstraSim output `.log` |
| Network statistics CSV | g2 / NS3 `network_statistics.csv` |
| Nodemap JSON | topology generator output |
| Power config JSON | `default_config.json` or custom file |

---

### 2 · Compute Power (`compute_model.py`)

`compute_parser.py` reads the AstraSim log and builds **per-NPU** timing from the
`[statistics]` lines emitted at the end of each NPU's execution:

```
[ts] [workload] [info] [SUMMARY] sys[X] finished, N cycles, exposed communication M cycles,
[ts] [statistics] [info] sys[X], Wall time: N
[ts] [statistics] [info] sys[X], GPU time: N
[ts] [statistics] [info] sys[X], Comm time: N
```

This produces a `ComputeStats` object with an `npu_stats` dict keyed by NPU id:

```
ComputeStats
  └─ npu_stats: { npu_id → NPUStats }
       ├─ total_time    ← Wall time  (actual elapsed cycles)
       ├─ compute_time  ← GPU time   (cycles in compute ops)
       └─ comm_time     ← Comm time  (exposed communication cycles)
```

Aggregate properties (`total_exec_time`, `compute_time`, `comm_time`,
`compute_utilization`) are derived from `npu_stats` (max / mean across NPUs).

#### Per-NPU power formula

For each NPU $i$, with $U_i = t_{\text{comp},i} / T_{\text{wall},i}$:

$$P_i = \begin{cases}
P_\text{sleep} + (P_\text{peak} - P_\text{sleep}) \times U_i & \text{compute LPM ON} \\
P_\text{idle}  + (P_\text{peak} - P_\text{idle})  \times U_i & \text{compute LPM OFF}
\end{cases}$$

$$E_{\text{GPU},i} = P_i \times T_{\text{wall},i}$$

$$P_\text{GPU,total} = \sum_i P_i \qquad E_\text{GPU,total} = \sum_i E_{\text{GPU},i}$$

Each NPU is charged for its **own** wall time, not the slowest NPU's — avoiding
energy over-counting in heterogeneous execution.

---

### 3 · Network Power (`network_model.py`)

`network_parser.py` combines the CSV traffic file with the nodemap:

```
NetworkStats
  ├─ links[]            (LinkStats per simulated link)
  │    ├─ src_id, dst_id
  │    ├─ bytes_transmitted
  │    ├─ bandwidth
  │    └─ link_type      ← set by topology-aware classifier
  ├─ topology_info      (TopologyInfo from nodemap)
  │    ├─ hosts
  │    └─ switches_by_type  { "tor": {t0,…}, "nvswitch": {v0,…}, … }
  └─ switch_connections     { switch_id: set(neighbours) }
```

#### 3a · Link Power (`LinkModel`)

Links are classified by endpoint node types:

| Link type | Example | Power params |
|-----------|---------|-------------|
| `nvlink`  | host ↔ nvswitch | `nvlink_*_power` |
| `nic`     | host ↔ nic node | `nic_*_power` |
| `network` | switch ↔ switch | `fabric_*_power` |

$$u = \min\!\left(\frac{\text{bytes\_transmitted}}{\text{bandwidth} \times T},\;1\right)$$

$$P_\text{link} = \begin{cases}
P_\text{sleep} + (P_\text{active} - P_\text{sleep}) \times u & \text{comm LPM ON} \\
P_\text{idle}  + (P_\text{active} - P_\text{idle})  \times u & \text{comm LPM OFF}
\end{cases}$$

#### 3b · Switch Port Power (`SwitchTypeModel`)

One model per discovered switch type (tor, nvswitch, aggregation, core, nic, per\_npu, …).

**Ports per switch** `d` is resolved in priority order:
1. `switch_degrees[type]` in the config JSON — explicit user value.
2. Average connectivity from topology graph — automatic when not supplied.
3. Fallback: 1.

$$N_\text{ports} = N_\text{switches} \times d$$

$$P_\text{static} = N_\text{ports} \times P_\text{idle/sleep}$$

$$P_\text{dynamic} = N_\text{ports} \times (P_\text{active} - P_\text{base}) \times \bar{u}$$

$$P_\text{switch type} = P_\text{static} + P_\text{dynamic}$$

where $\bar{u}$ is the average port utilisation (idle ports contribute 0).

---

### 4 · Total Power and Energy

$$P_\text{total} = P_\text{GPU,total} + \sum_\text{links} P_\text{link} + \sum_\text{switch types} P_\text{switch type}$$

GPU and network energy are computed separately:

$$E_\text{GPU} = \sum_i P_i \times T_{\text{wall},i} \qquad E_\text{network} = P_\text{network} \times T_\text{exec}$$

$$E_\text{total} = E_\text{GPU} + E_\text{network}$$

$$\text{Efficiency} = \frac{\text{batch\_size} \times \text{iterations}}{E_\text{total}} \quad \text{[samples/J]}$$

---

## Topology Configuration

Nodes are classified by name prefix (configurable in the JSON):

```json
"switch_type_prefixes": {
  "nvswitch":    "v",
  "tor":         "t",
  "aggregation": "a",
  "core":        "c",
  "nic":         "n",
  "per_npu":     "p"
},
"host_prefix": "h"
```

Override these if your nodemap uses different naming conventions.

Optionally supply the port count **per switch** (model multiplies by switch count automatically):

```json
"switch_degrees": {
  "tor":      32,
  "nvswitch": 18
}
```

---

## Hardware Config Files

Partial overrides are supported — unspecified fields use defaults.

```json
{
  "comment": "NVIDIA A100 (400 W TDP)",
  "gpu_peak_power":  400.0,
  "gpu_idle_power":   60.0,
  "gpu_sleep_power":  15.0,
  "switch_degrees": { "tor": 32 }
}
```
