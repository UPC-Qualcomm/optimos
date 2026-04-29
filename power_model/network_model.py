"""
Network Power Model

Calculates power consumption for network links and switches.

Switch power is topology-aware:
  1. Switch counts come from the nodemap topology (via NetworkStats.topology_info).
     Only switch types present in the topology are modelled; absent types are ignored.
  2. Port count (degree) is taken from PowerConfig.switch_port_counts[type] when
     present — populated automatically from the {type}_N_ports JSON config keys
     that feed the datasheet converter.  Falls back to average topology-connectivity
     degree when N_ports is not provided for a type.
  3. Port utilisation is derived from the link-traffic CSV.

Static / Dynamic separation (no expansion multiplier):
  static_power  = switch_count × degree × P_base
  dynamic_power = switch_count × degree × (P_active − P_base) × avg_utilisation

Link power uses the link_type field set by the parser (actual switch-type name):
  - 'nvlink'      → nvswitch power params
  - 'nic'         → nic power params
  - 'tor'         → tor power params
  - 'aggregation' → aggregation power params
  - 'core'        → core power params
  - 'per_npu'     → per_npu power params (defaults to nic params when unconfigured)
  - anything else → tor power params (fallback)
"""

from typing import Dict, List, Optional, Set
from power_config import PowerConfig
from network_parser import NetworkStats, LinkStats


# ---------------------------------------------------------------------------
# Link model
# ---------------------------------------------------------------------------

class LinkModel:
    """
    Power for a single directed network link.

    Static  = multiplier × P_base
    Dynamic = (P_active − P_base) × U   (U = 1.0 when LPM off)

    Type detection priority:
      1. link.link_type field (set by topology-aware parser)
      2. Node-name prefix fallback (for CSV files without topology info)
    """

    def __init__(self, link: LinkStats, total_time: float, config: PowerConfig,
                 active_time: float = None):
        self.link        = link
        self.total_time  = total_time
        self.active_time = active_time if (active_time and active_time > 0) else total_time
        self.config      = config

    def utilization(self) -> float:
        """Link utilisation relative to the active (comm) window."""
        return self.link.utilization(self.active_time)

    def _link_params(self) -> tuple:
        """Return (active, idle, sleep) for this link type.

        Reuses the same per-type switch params as SwitchTypeModel so that
        both the link-wire and switch-port sides share a consistent power
        budget.  The link_type field is the actual switch-type name set by
        the topology-aware parser (e.g. 'nvlink', 'nic', 'tor',
        'aggregation', 'core', …), so every fabric layer gets its own
        power params.  Falls back to tor params for unknown fabric types.
        """
        cfg = self.config
        type_map = {
            'nvlink':      (cfg.nvswitch_active_power,  cfg.nvswitch_idle_power,      cfg.nvswitch_sleep_power),
            'nic':         (cfg.nic_active_power,        cfg.nic_idle_power,           cfg.nic_sleep_power),
            'tor':         (cfg.tor_active_power,        cfg.tor_idle_power,           cfg.tor_sleep_power),
            'aggregation': (cfg.aggregation_active_power, cfg.aggregation_idle_power, cfg.aggregation_sleep_power),
            'core':        (cfg.core_active_power,       cfg.core_idle_power,          cfg.core_sleep_power),
            'per_npu':     (cfg.per_npu_active_power,    cfg.per_npu_idle_power,       cfg.per_npu_sleep_power),
        }
        return type_map.get(self.link.link_type, type_map['tor'])

    def static_power(self) -> float:
        _, idle, sleep = self._link_params()
        return sleep if self.config.comm_lpm_enabled else idle

    def dynamic_power(self) -> float:
        active, idle, sleep = self._link_params()
        U = min(self.utilization(), 1.0)
        if self.config.comm_lpm_enabled:
            return (active - sleep) * U
        return (active - idle) * U

    def power(self) -> float:
        return self.static_power() + self.dynamic_power()

    # --- type helpers ---

    def _is_nvlink(self) -> bool:
        return self.link.link_type == "nvlink"

    def _is_nic(self) -> bool:
        return self.link.link_type == "nic"


# ---------------------------------------------------------------------------
# Switch-type model
# ---------------------------------------------------------------------------

class SwitchTypeModel:
    """
    Power model for all switches of one type (e.g. all ToR switches).

    Only instantiated for switch types that exist in the topology;
    absent types are never created and contribute zero power.

    Degree (ports per switch)
    -------------------------
    1. config.switch_port_counts[switch_type]  — from {type}_N_ports in the JSON
       (same value used by the datasheet converter); use if present.
    2. Average topology-connectivity degree (from switch_connections).
    3. Fallback: 1 (warn once).

    Power formula (both static and dynamic use the same switch port params):
      total_ports   = switch_count × degree
      static_power  = total_ports × P_idle_or_sleep          (no expansion)
      dynamic_power = total_ports × (P_active − P_base) × U  (no expansion)
    """

    def __init__(self,
                 switch_type: str,
                 switch_ids: Set[str],
                 config: PowerConfig,
                 switch_connections: dict,
                 link_traffic: dict,
                 link_bandwidth_map: dict,
                 total_time: float,
                 active_time: float = None):
        self.switch_type        = switch_type
        self.switch_ids         = {str(s) for s in switch_ids}
        self.config             = config
        self.switch_connections = switch_connections   # node_id → set of neighbours
        self.link_traffic       = link_traffic         # (src, dst) → bytes
        self.link_bandwidth_map = link_bandwidth_map   # (src, dst) → bandwidth in Bytes/s
        self.total_time         = total_time
        self.active_time        = active_time if (active_time and active_time > 0) else total_time

    # ------------------------------------------------------------------
    # Counts and degree
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of switches of this type in the topology."""
        return len(self.switch_ids)

    def _get_degree(self) -> int:
        """
        Ports per switch.

        Priority:
          1. config.switch_port_counts[type]  — from {type}_N_ports JSON key
             (same N_ports used by the datasheet converter).
          2. Average number of neighbours per switch from topology connectivity.
          3. 1 (silent fallback).
        """
        n_ports = self.config.switch_port_counts.get(self.switch_type, 0)
        if n_ports and n_ports > 0:
            return n_ports

        if self.switch_connections and self.switch_ids:
            degrees = [
                len(self.switch_connections.get(sw, set()))
                for sw in self.switch_ids
            ]
            non_zero = [d for d in degrees if d > 0]
            if non_zero:
                return max(1, round(sum(non_zero) / len(non_zero)))

        return 1

    def total_ports(self) -> int:
        """Total port count = switch_count × ports_per_switch."""
        return self.count * self._get_degree()

    # ------------------------------------------------------------------
    # Utilisation
    # ------------------------------------------------------------------

    def _per_switch_utilization(self) -> Dict[str, float]:
        """
        Per-switch average port utilization.

        For each switch s:
            raw_sum(s) = sum of link utilizations for every link in the
                         traffic CSV whose src OR dst is s
                          uses per-link bandwidth
                          util(s)    = raw_sum(s) / degree

        The raw per-link utilization is NOT capped here; the switch-level
        saturation cap min(util(s), 1.0) is applied in dynamic_power().

        Returns
        -------
        dict  { switch_id: utilization }  – one entry per switch in switch_ids.
        """
        degree = self._get_degree()
        if self.active_time <= 0 or degree == 0:
            return {sw: 0.0 for sw in self.switch_ids}

        raw: Dict[str, float] = {sw: 0.0 for sw in self.switch_ids}
        for (src, dst), bytes_tx in self.link_traffic.items():
            src_s, dst_s = str(src), str(dst)
            # Use per-link bandwidth; divide by active_time (comm window)
            bw = self.link_bandwidth_map.get((src, dst))
            u = bytes_tx / (bw * self.active_time)
            if src_s in raw:
                raw[src_s] += u
            if dst_s in raw:
                raw[dst_s] += u

        return {sw: link_sum / degree for sw, link_sum in raw.items()}

    def avg_port_utilization(self) -> float:
        """
        Mean per-switch port utilization across all switches of this type.

        avg = mean( util(s) for s in switch_ids )

        Used for reporting / breakdown dicts.  Dynamic power is computed
        per-switch (see dynamic_power) so that unutilised switches do not
        inflate the result.
        """
        per_sw = self._per_switch_utilization()
        if not per_sw:
            return 0.0
        return sum(per_sw.values()) / len(per_sw)

    # ------------------------------------------------------------------
    # Power  (static / dynamic separated, no expansion multiplier)
    # ------------------------------------------------------------------

    def _get_power_params(self) -> tuple:
        """
        Get (active, idle, sleep) power per port for this switch type.

        Returns
        -------
        (active, idle, sleep) in Watts per port
        """
        type_map = {
            'nvswitch': (
                self.config.nvswitch_active_power,
                self.config.nvswitch_idle_power,
                self.config.nvswitch_sleep_power,
            ),
            'tor': (
                self.config.tor_active_power,
                self.config.tor_idle_power,
                self.config.tor_sleep_power,
            ),
            'aggregation': (
                self.config.aggregation_active_power,
                self.config.aggregation_idle_power,
                self.config.aggregation_sleep_power,
            ),
            'core': (
                self.config.core_active_power,
                self.config.core_idle_power,
                self.config.core_sleep_power,
            ),
            'nic': (
                self.config.nic_active_power,
                self.config.nic_idle_power,
                self.config.nic_sleep_power,
            ),
            'per_npu': (
                self.config.per_npu_active_power,
                self.config.per_npu_idle_power,
                self.config.per_npu_sleep_power,
            ),
        }
        # Return params for this type; fallback to tor if unknown type
        return type_map.get(self.switch_type, type_map['tor'])

    def static_power(self) -> float:
        """
        Static (baseline) power.

        LPM OFF: total_ports × P_idle
        LPM ON:  total_ports × P_sleep
        """
        N = self.total_ports()
        _, idle, sleep = self._get_power_params()
        if self.config.comm_lpm_enabled:
            return N * sleep
        return N * idle

    def dynamic_power(self) -> float:
        """
        Dynamic (utilisation-driven) power — computed per individual switch.

        For each switch s:
            util(s)     = sum(link_util_i touching s) / degree
            dynamic_s   = degree × (P_active − P_base) × min(util(s), 1.0)

        Total dynamic power = Σ dynamic_s over all switches of this type.

        Summing per-switch (rather than using a global aggregate) means
        switches with no observed traffic contribute zero dynamic power,
        which is more accurate when the topology has partially unused switches.

        Example: 2 links each at 100 % utilisation touch one switch,
        user-configured degree = 8:
            util(s)   = (1.0 + 1.0) / 8 = 0.25
            dynamic_s = 8 × delta × 0.25 = 2 × delta
        """
        degree = self._get_degree()
        per_sw = self._per_switch_utilization()
        active, idle, sleep = self._get_power_params()
        if self.config.comm_lpm_enabled:
            delta = active - sleep
        else:
            delta = active - idle
        return degree * delta * sum(min(u, 1.0) for u in per_sw.values())

    def total_power(self) -> float:
        return self.static_power() + self.dynamic_power()

    def summary(self) -> dict:
        return {
            'type':            self.switch_type,
            'count':           self.count,
            'degree':          self._get_degree(),
            'total_ports':     self.total_ports(),
            'avg_utilization': self.avg_port_utilization(),
            'static_power_W':  self.static_power(),
            'dynamic_power_W': self.dynamic_power(),
            'total_power_W':   self.total_power(),
        }


# ---------------------------------------------------------------------------
# Aggregate network model
# ---------------------------------------------------------------------------

class NetworkModel:
    """
    Aggregates power for all network components.

    Components
    ----------
    - Links      : one LinkModel per row in the link-traffic CSV
    - Switches   : one SwitchTypeModel per type found in topology_info.switches_by_type
    """

    def __init__(self,
                 network_stats: NetworkStats,
                 total_time: float,
                 config: PowerConfig,
                 comm_time: float = None):
        self.stats      = network_stats
        self.total_time = total_time
        self.config     = config
        self.active_time = comm_time if (comm_time and comm_time > 0) else total_time

        # --- link models ---
        self.link_models: List[LinkModel] = [
            LinkModel(lnk, total_time, config, active_time=self.active_time)
            for lnk in network_stats.links
        ]

        # --- shared data for switch models ---
        link_traffic: dict = {
            (lnk.src_id, lnk.dst_id): lnk.bytes_transmitted
            for lnk in network_stats.links
        }
        link_bandwidth_map: dict = {
            (lnk.src_id, lnk.dst_id): lnk.bandwidth
            for lnk in network_stats.links
        }

        topo = network_stats.topology_info

        def _make(sw_type: str, sw_ids: Set[str]) -> SwitchTypeModel:
            return SwitchTypeModel(
                switch_type=sw_type,
                switch_ids=sw_ids,
                config=config,
                switch_connections=topo.switch_connections if topo else {},
                link_traffic=link_traffic,
                link_bandwidth_map=link_bandwidth_map,
                total_time=total_time,
                active_time=self.active_time,
            )

        # --- switch models (one per type from topology) ---
        self.switch_models: Dict[str, SwitchTypeModel] = {}
        if topo and topo.switches_by_type:
            for sw_type, sw_ids in topo.switches_by_type.items():
                self.switch_models[sw_type] = _make(sw_type, sw_ids)

    # ------------------------------------------------------------------
    # Link aggregates
    # ------------------------------------------------------------------

    def total_link_power(self) -> float:
        return sum(m.power() for m in self.link_models)

    def total_link_static_power(self) -> float:
        return sum(m.static_power() for m in self.link_models)

    def total_link_dynamic_power(self) -> float:
        return sum(m.dynamic_power() for m in self.link_models)

    # ------------------------------------------------------------------
    # Switch aggregates
    # ------------------------------------------------------------------

    def total_switch_power(self) -> float:
        return sum(m.total_power() for m in self.switch_models.values())

    def total_switch_static_power(self) -> float:
        return sum(m.static_power() for m in self.switch_models.values())

    def total_switch_dynamic_power(self) -> float:
        return sum(m.dynamic_power() for m in self.switch_models.values())

    # ------------------------------------------------------------------
    # Grand totals
    # ------------------------------------------------------------------

    def total_network_power(self) -> float:
        return self.total_link_power() + self.total_switch_power()

    def total_network_energy(self) -> float:
        """Network energy (Joules), split into static and dynamic components.

        Static  : all ports/links remain powered for the full job duration.
          E_static  = (link_static + switch_static) × total_time

        Dynamic : utilisation-driven extra power exists only while the network
          is actively transmitting (the comm window, not the full wall time).
          E_dynamic = (link_dynamic + switch_dynamic) × active_time

        For uncongested links (utilisation < 1) this equals the old formula
        when active_time == total_time.  When active_time < total_time the
        dynamic term is correctly smaller, avoiding over-counting power that
        only exists during comm phases.
        """
        E_static  = (self.total_link_static_power() +
                     self.total_switch_static_power())  * self.total_time
        E_dynamic = (self.total_link_dynamic_power() +
                     self.total_switch_dynamic_power()) * self.active_time
        return E_static + E_dynamic

    # ------------------------------------------------------------------
    # Breakdown
    # ------------------------------------------------------------------

    def get_power_breakdown(self) -> Dict:
        """
        Return a comprehensive power-breakdown dictionary.

        Every switch type found in the topology gets its own set of keys:
            <type>_power_W, <type>_static_W, <type>_dynamic_W,
            <type>_count, <type>_degree, <type>_total_ports, <type>_avg_utilization
        """
        breakdown: dict = {
            'link_power_W':           self.total_link_power(),
            'link_static_W':          self.total_link_static_power(),
            'link_dynamic_W':         self.total_link_dynamic_power(),
            'total_switch_power_W':   self.total_switch_power(),
            'total_switch_static_W':  self.total_switch_static_power(),
            'total_switch_dynamic_W': self.total_switch_dynamic_power(),
        }

        # Per-type entries (generic – works for any topology)
        for sw_type, model in self.switch_models.items():
            p = sw_type
            breakdown[f'{p}_power_W']          = model.total_power()
            breakdown[f'{p}_static_W']         = model.static_power()
            breakdown[f'{p}_dynamic_W']        = model.dynamic_power()
            breakdown[f'{p}_count']            = model.count
            breakdown[f'{p}_degree']           = model._get_degree()
            breakdown[f'{p}_total_ports']      = model.total_ports()
            breakdown[f'{p}_avg_utilization']  = model.avg_port_utilization()

        # Grand totals and run metadata
        breakdown['total_network_power_W']   = self.total_network_power()
        breakdown['total_network_energy_J']  = self.total_network_energy()
        breakdown['num_links']               = len(self.link_models)
        breakdown['total_switch_count']      = sum(m.count for m in self.switch_models.values())
        breakdown['total_bytes_transmitted'] = self.stats.total_bytes_transmitted
        breakdown['total_time_s']            = self.total_time
        breakdown['active_time_s']           = self.active_time
        breakdown['lpm_enabled']             = self.config.comm_lpm_enabled

        return breakdown
