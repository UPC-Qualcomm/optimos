"""
Power Configuration Module

Defines hardware power parameters and LPM (Low Power Mode) settings
for the 4-mode power model analysis.
"""

import json
from dataclasses import dataclass, asdict, field
from typing import Dict, Optional


@dataclass
class PowerConfig:
    """
    Configuration for power modeling with LPM support.
    
    Defines the 4 operational modes:
    - Mode A: Compute LPM OFF, Communication LPM OFF (baseline)
    - Mode B: Compute LPM ON,  Communication LPM OFF
    - Mode C: Compute LPM OFF, Communication LPM ON
    - Mode D: Compute LPM ON,  Communication LPM ON (full energy-proportional)
    """
    
    # ==================== LPM Mode Flags ====================
    compute_lpm_enabled: bool = False
    comm_lpm_enabled: bool = False
    
    # ==================== GPU Power Parameters (Watts) ====================
    gpu_peak_power: float = 700.0      # H100 SXM5 TDP
    gpu_idle_power: float = 100.0      # GPU idle (compute off, but powered)
    gpu_sleep_power: float = 20.0      # Deep sleep with LPM enabled
    
    # ==================== Switch Port Power (Watts per port) ====================
    # One active/idle/sleep triple per switch type present in the topology.
    # Also drives link power: nvlink links → nvswitch params;
    #                         nic links    → nic params;
    #                         fabric links → tor params.
    #
    # HOW TO OBTAIN VALUES FROM A DATASHEET
    # ----------------------------------------
    # Datasheets publish total switch power (Watts).
    # Use switch_port_power_from_datasheet() to convert:
    #
    #   active_power = P_max_W  / N_ports   (full line-rate)
    #   idle_power   = P_idle_W / N_ports   (ports linked, no traffic)
    #   sleep_power  = omega_sport * active  (EEE 802.3az default = 0.1)
    #
    # The paper's omega_ports / omega_sport fractions are only needed when
    # P_idle is unknown and must be estimated from P_max.  With a datasheet
    # you skip that decomposition entirely.
    #
    # Example (Arista 7050CX3-32S, 32×100GbE, from datasheet):
    #   P_max=230 W, P_idle=60 W, N_ports=64 → active=3.59, idle=0.94, sleep=0.36
    #
    # Reference values used below (direct P_total / N_ports):
    #   nvswitch  NVSwitch 3.0 (Hopper): 900 W / 144 ports ≈ 6.25 W/port  [NVIDIA Hot Chips 2022]
    #   tor       Arista 7050CX3-32S (100GbE): 230 W / 64 ports ≈ 3.6 W/port  [7050CX3-32S DS]
    #   nic       ConnectX-7 400G NDR (MCX755106AS-HEAT): ~35 W / 4 lanes ≈ 8 W/port  [ConnectX-7 DS]
    #   per_npu   Intra-node GPU ring: estimated from NVLink lane power budget

    # Each triple is produced by:
    #   active, idle, sleep = switch_port_power_from_datasheet(P_max_W, P_idle_W, N_ports)
    # The arithmetic expressions are kept live so the datasheet source is always visible
    # in the code without needing to look anything up.

    # NVSwitch 3.0 (Hopper): 900 W / 144 NVLink 4.0 ports, ~225 W idle  [NVIDIA Hot Chips 34, 2022]
    # switch_port_power_from_datasheet(900, 225, 144)
    nvswitch_active_power: float = 900.0 / 144        # 6.25  W/port
    nvswitch_idle_power: float   = 225.0 / 144        # 1.56  W/port  (~25 % of P_max)
    nvswitch_sleep_power: float  = 0.1 * 900.0 / 144  # 0.63  W/port  (EEE ω_sport = 0.1)

    # Arista 7050CX3-32S (64×100GbE QSFP28): 230 W max, 60 W idle  [7050CX3-32S DS, 2021]
    # switch_port_power_from_datasheet(230, 60, 64)
    tor_active_power: float = 230.0 / 64         # 3.59  W/port
    tor_idle_power: float   =  60.0 / 64         # 0.94  W/port
    tor_sleep_power: float  = 0.1 * 230.0 / 64   # 0.36  W/port

    aggregation_active_power: float = 230.0 / 64
    aggregation_idle_power: float   =  60.0 / 64
    aggregation_sleep_power: float  = 0.1 * 230.0 / 64

    core_active_power: float = 230.0 / 64
    core_idle_power: float   =  60.0 / 64
    core_sleep_power: float  = 0.1 * 230.0 / 64

    # ConnectX-7 400G NDR (MCX755106AS-HEAT): ~35 W max, ~8 W idle, 4 physical lanes  [CX-7 DS, 2022]
    # switch_port_power_from_datasheet(35, 8, 4)
    nic_active_power: float = 35.0 / 4        # 8.75  W/port
    nic_idle_power: float   =  8.0 / 4        # 2.00  W/port
    nic_sleep_power: float  = 0.1 * 35.0 / 4  # 0.875 W/port

    # Per-NPU intra-node ring/chain — estimated from NVLink short-reach lane power budget
    # switch_port_power_from_datasheet(8, 2, 4)  [estimated, no public datasheet]
    per_npu_active_power: float = 8.0 / 4        # 2.0  W/port
    per_npu_idle_power: float   = 2.0 / 4        # 0.5  W/port
    per_npu_sleep_power: float  = 0.1 * 8.0 / 4  # 0.2  W/port

    # ==================== System size ====================
    npus_per_node: int = 8             # Number of NPUs per host node

    # ==================== Topology Node Classification ====================
    # Maps each switch-type name to the nodemap name prefix used by that topology.
    # Works for any topology (FoldedClos, Jellyfish, Dragonfly, …).
    # Override in the config JSON to match your nodemap naming scheme.
    host_prefix: str = "h"
    switch_type_prefixes: Dict[str, str] = field(default_factory=lambda: {
        "nvswitch":    "v",   # NVSwitch / intra-node GPU interconnect
        "tor":         "t",   # Top-of-Rack switch
        "aggregation": "a",   # Aggregation switch
        "core":        "c",   # Core switch
        "nic":         "n",   # NIC-level switch (ring/FC intra-node)
        "per_npu":     "p",   # Per-NPU switch (ring/FC intra-node)
    })

    # ==================== Switch Port Counts ====================
    # Maps switch-type name to the number of ports PER SWITCH.
    # Populated automatically from {type}_N_ports keys in the JSON config
    # (the same N_ports used by the datasheet converter).
    # Used by SwitchTypeModel._get_degree() for utilisation normalisation
    # and port-count calculations.  Falls back to topology-connectivity
    # degree when a type is absent from this dict.
    switch_port_counts: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        """Validate power parameters."""
        assert self.gpu_peak_power >= self.gpu_idle_power >= self.gpu_sleep_power
        # Validate per-switch-type power hierarchy (active ≥ idle ≥ sleep ≥ 0)
        for sw_type in ['nvswitch', 'tor', 'aggregation', 'core', 'nic', 'per_npu']:
            active = getattr(self, f'{sw_type}_active_power')
            idle   = getattr(self, f'{sw_type}_idle_power')
            sleep  = getattr(self, f'{sw_type}_sleep_power')
            assert active >= idle >= sleep >= 0, f"{sw_type} power hierarchy violated"

    @staticmethod
    def switch_port_power_from_datasheet(
        P_max_W: float,
        P_idle_W: float,
        N_ports: int,
        omega_sport: float = 0.1,
    ) -> tuple:
        """Convert whole-switch datasheet figures to per-port (active, idle, sleep) W.

        Datasheets publish *total* switch power; the code expects *per-port* values.
        This helper does that division and derives sleep from the EEE standard fraction.

        Parameters
        ----------
        P_max_W     : total switch power at 100 % line-rate (all ports active), Watts
        P_idle_W    : total switch power with ports linked but zero traffic, Watts
        N_ports     : number of ports on the switch
        omega_sport : fraction of active port power consumed while in EEE sleep state
                      (IEEE 802.3az default = 0.1, i.e. 10 %).
                      The paper uses this same symbol in Eq. 1.

        Returns
        -------
        (active_power, idle_power, sleep_power)  in Watts per port.
        Pass these directly as e.g. tor_active_power / tor_idle_power / tor_sleep_power.

        Relation to paper Eq. 1–2
        -------------------------
        The paper decomposes idle into fractions:
            idle_per_port = Omega_port * [1 - omega_ports * (1 - omega_sport)]
        where Omega_port = P_max_W / N_ports  and
              omega_ports = (P_max_W - P_chassis) / P_max_W.

        With a datasheet we have P_idle_W directly, so:
            idle_per_port = P_idle_W / N_ports
        Both expressions are equal — the paper's fractions are only needed when
        P_idle is *unknown* and must be estimated from P_max.

        Examples
        --------
        # Arista 7050CX3-32S  (32×100GbE QSFP28, datasheet 2021)
        active, idle, sleep = PowerConfig.switch_port_power_from_datasheet(230, 60, 64)
        # → (3.59, 0.94, 0.36)

        # Mellanox SN2700 36-port (paper Table 2 values: 180 W full, 33 W idle)
        active, idle, sleep = PowerConfig.switch_port_power_from_datasheet(180, 33, 36)
        # → (5.00, 0.92, 0.50)  — matches paper's Omega_port=5, derived idle≈0.92
        """
        active = P_max_W  / N_ports
        idle   = P_idle_W / N_ports
        sleep  = omega_sport * active
        return active, idle, sleep
    
    def get_mode_name(self) -> str:
        """Return human-readable mode name."""
        if not self.compute_lpm_enabled and not self.comm_lpm_enabled:
            return "Mode A: Baseline (No LPM)"
        elif self.compute_lpm_enabled and not self.comm_lpm_enabled:
            return "Mode B: Compute LPM Only"
        elif not self.compute_lpm_enabled and self.comm_lpm_enabled:
            return "Mode C: Communication LPM Only"
        else:
            return "Mode D: Full LPM (Energy-Proportional)"
    
    @classmethod
    def create_mode_a(cls):
        """Mode A: Compute LPM OFF, Communication LPM OFF (Baseline)"""
        return cls(compute_lpm_enabled=False, comm_lpm_enabled=False)
    
    @classmethod
    def create_mode_b(cls):
        """Mode B: Compute LPM ON, Communication LPM OFF"""
        return cls(compute_lpm_enabled=True, comm_lpm_enabled=False)
    
    @classmethod
    def create_mode_c(cls):
        """Mode C: Compute LPM OFF, Communication LPM ON"""
        return cls(compute_lpm_enabled=False, comm_lpm_enabled=True)
    
    @classmethod
    def create_mode_d(cls):
        """Mode D: Compute LPM ON, Communication LPM ON (Full Energy-Proportional)"""
        return cls(compute_lpm_enabled=True, comm_lpm_enabled=True)
    
    @classmethod
    def get_all_modes(cls):
        """Return all 4 power modes for comparison."""
        return {
            'A': cls.create_mode_a(),
            'B': cls.create_mode_b(),
            'C': cls.create_mode_c(),
            'D': cls.create_mode_d()
        }
    
    @classmethod
    def from_json(cls, json_path: str, mode: Optional[str] = None) -> 'PowerConfig':
        """Load power configuration from a JSON file.

        Switch port power is resolved in priority order for each switch type:

        1. **Datasheet inputs** (preferred) — if the JSON contains
           ``{type}_P_max_W``, ``{type}_P_idle_W`` and ``{type}_N_ports``,
           :meth:`switch_port_power_from_datasheet` is called here at load
           time to derive ``active / idle / sleep`` W/port automatically.
           An optional ``{type}_omega_sport`` key overrides the EEE 0.1
           default for that type.

        2. **Direct W/port values** (backward-compatible) — if datasheet
           keys are absent the loader falls back to
           ``{type}_active_power``, ``{type}_idle_power``,
           ``{type}_sleep_power``.

        3. **nic params** (per_npu only) — if neither form above is present
           in the JSON for ``per_npu``, the already-resolved nic params are
           used so that per-NPU intra-node switches behave like NICs by
           default without the user needing to configure them explicitly.

        4. **Python class defaults** — used for any other field absent from
           the JSON under both forms above.

        The ``{type}_N_ports`` value (when present) is also stored in
        ``switch_port_counts`` and used by ``SwitchTypeModel._get_degree()``
        for utilisation normalisation.  Switch types absent from the JSON
        fall back to topology-connectivity degree.

        Only switch types present in the topology are modelled at runtime;
        types absent from the topology are simply ignored.

        Args:
            json_path: Path to JSON configuration file.
            mode: Optional 'A'/'B'/'C'/'D' — overrides the LPM flags from
                  the JSON file.

        Returns:
            PowerConfig instance ready for power calculation.

        Datasheet-input JSON example::

            {
                "nvswitch_P_max_W": 504,  "nvswitch_P_idle_W": 126,  "nvswitch_N_ports": 96,
                "tor_P_max_W":      180,  "tor_P_idle_W":      33,   "tor_N_ports":      36,
                "nic_P_max_W":       25,  "nic_P_idle_W":       6,   "nic_N_ports":       4,
                "per_npu_P_max_W":    8,  "per_npu_P_idle_W":   2,   "per_npu_N_ports":   4
            }
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)

        defaults = cls()

        if mode is not None:
            mode = mode.upper()
            if mode == 'A':
                config_dict['compute_lpm_enabled'] = False
                config_dict['comm_lpm_enabled'] = False
            elif mode == 'B':
                config_dict['compute_lpm_enabled'] = True
                config_dict['comm_lpm_enabled'] = False
            elif mode == 'C':
                config_dict['compute_lpm_enabled'] = False
                config_dict['comm_lpm_enabled'] = True
            elif mode == 'D':
                config_dict['compute_lpm_enabled'] = True
                config_dict['comm_lpm_enabled'] = True
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be A, B, C, or D")

        # --- Per-switch-type power resolution + port-count collection ---
        # For each type: use datasheet inputs → converter if all three keys present;
        # otherwise fall back to direct W/port values or Python defaults.
        # Exception: per_npu with NO keys in JSON falls back to the resolved nic params
        # so that intra-node per-NPU switches behave like NICs by default.
        # N_ports (when available) is stored for degree lookup in _get_degree().
        sw_active: Dict[str, float] = {}
        sw_idle:   Dict[str, float] = {}
        sw_sleep:  Dict[str, float] = {}
        port_counts: Dict[str, int] = {}
        for sw_type in ('nvswitch', 'tor', 'aggregation', 'core', 'nic', 'per_npu'):
            has_datasheet = all(f'{sw_type}_{k}' in config_dict
                                for k in ('P_max_W', 'P_idle_W', 'N_ports'))
            has_direct    = any(f'{sw_type}_{k}' in config_dict
                                for k in ('active_power', 'idle_power', 'sleep_power'))
            if has_datasheet:
                omega = config_dict.get(f'{sw_type}_omega_sport', 0.1)
                n = int(config_dict[f'{sw_type}_N_ports'])
                a, i, s = cls.switch_port_power_from_datasheet(
                    config_dict[f'{sw_type}_P_max_W'],
                    config_dict[f'{sw_type}_P_idle_W'],
                    n,
                    omega_sport=omega,
                )
                port_counts[sw_type] = n
            elif has_direct:
                a = config_dict.get(f'{sw_type}_active_power',
                                    getattr(defaults, f'{sw_type}_active_power'))
                i = config_dict.get(f'{sw_type}_idle_power',
                                    getattr(defaults, f'{sw_type}_idle_power'))
                s = config_dict.get(f'{sw_type}_sleep_power',
                                    getattr(defaults, f'{sw_type}_sleep_power'))
            else:
                # Not configured in JSON at all.
                # per_npu: default to the already-resolved nic params (intra-node
                # per-NPU switches behave like NICs unless explicitly overridden).
                if sw_type == 'per_npu' and 'nic' in sw_active:
                    a, i, s = sw_active['nic'], sw_idle['nic'], sw_sleep['nic']
                else:
                    a = getattr(defaults, f'{sw_type}_active_power')
                    i = getattr(defaults, f'{sw_type}_idle_power')
                    s = getattr(defaults, f'{sw_type}_sleep_power')
            sw_active[sw_type], sw_idle[sw_type], sw_sleep[sw_type] = a, i, s

        # Merge switch_type_prefixes: defaults first, then JSON overrides
        merged_prefixes = dict(defaults.switch_type_prefixes)
        if 'switch_type_prefixes' in config_dict:
            merged_prefixes.update(config_dict['switch_type_prefixes'])

        return cls(
            # LPM flags
            compute_lpm_enabled=config_dict.get('compute_lpm_enabled', defaults.compute_lpm_enabled),
            comm_lpm_enabled=config_dict.get('comm_lpm_enabled', defaults.comm_lpm_enabled),
            # GPU power
            gpu_peak_power=config_dict.get('gpu_peak_power',  defaults.gpu_peak_power),
            gpu_idle_power=config_dict.get('gpu_idle_power',  defaults.gpu_idle_power),
            gpu_sleep_power=config_dict.get('gpu_sleep_power', defaults.gpu_sleep_power),
            # Per-switch-type power (resolved above via converter or direct fallback)
            nvswitch_active_power=sw_active['nvswitch'],
            nvswitch_idle_power=sw_idle['nvswitch'],
            nvswitch_sleep_power=sw_sleep['nvswitch'],
            tor_active_power=sw_active['tor'],
            tor_idle_power=sw_idle['tor'],
            tor_sleep_power=sw_sleep['tor'],
            aggregation_active_power=sw_active['aggregation'],
            aggregation_idle_power=sw_idle['aggregation'],
            aggregation_sleep_power=sw_sleep['aggregation'],
            core_active_power=sw_active['core'],
            core_idle_power=sw_idle['core'],
            core_sleep_power=sw_sleep['core'],
            nic_active_power=sw_active['nic'],
            nic_idle_power=sw_idle['nic'],
            nic_sleep_power=sw_sleep['nic'],
            per_npu_active_power=sw_active['per_npu'],
            per_npu_idle_power=sw_idle['per_npu'],
            per_npu_sleep_power=sw_sleep['per_npu'],
            # System size
            npus_per_node=config_dict.get('npus_per_node', defaults.npus_per_node),
            # Topology classification
            host_prefix=config_dict.get('host_prefix', defaults.host_prefix),
            switch_type_prefixes=merged_prefixes,
            # Port counts (from N_ports keys) — used as degree source by SwitchTypeModel
            switch_port_counts=port_counts,
        )
    
    def to_json(self, json_path: str, indent: int = 2) -> None:
        """Save power configuration to JSON file.
        
        Args:
            json_path: Path to save JSON file
            indent: JSON indentation (default: 2)
        """
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=indent)
        print(f"Configuration saved to: {json_path}")
