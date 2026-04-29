"""
Network Topology Nodemap Parser

Preprocesses a nodemap.json (numeric_id → named_id) to extract the network
structure for any topology type (FoldedClos, Jellyfish, Dragonfly, …).

Node classification is driven by a user-configurable prefix map so the parser
never needs to know the topology type in advance.  The caller supplies a dict
such as:

    prefix_map = {
        "nvswitch":    "v",   # nodes whose name starts with 'v'
        "tor":         "t",
        "aggregation": "a",
        "core":        "c",
        "nic":         "n",
        "per_npu":     "p",
    }

and an optional host_prefix (default "h").

The resulting TopologyInfo carries:
  - hosts                  : set of host node names
  - switches_by_type       : Dict[type_name, Set[node_name]]  ← primary data
  - switch_connections     : Dict[node_name, Set[neighbour]]  (from topology file)
  - num_* convenience attrs (derived from the sets above)
"""

import json
from collections import defaultdict
from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# Default prefix map  (matches the convention used in create_topology scripts)
# ---------------------------------------------------------------------------

DEFAULT_PREFIX_MAP: Dict[str, str] = {
    "nvswitch":    "v",   # NVSwitch / intra-node GPU interconnect
    "tor":         "t",   # Top-of-Rack switch
    "aggregation": "a",   # Aggregation switch
    "core":        "c",   # Core switch
    "nic":         "n",   # NIC-level switch (ring/FC intra-node)
    "per_npu":     "p",   # Per-NPU switch (ring/FC intra-node)
}

DEFAULT_HOST_PREFIX: str = "h"


# ---------------------------------------------------------------------------
# TopologyInfo  –  the data container
# ---------------------------------------------------------------------------

class TopologyInfo:
    """Container for parsed topology information."""

    def __init__(self):
        # ---- generic (primary) ----
        self.hosts: Set[str] = set()
        # All switch nodes, keyed by user-defined type name
        self.switches_by_type: Dict[str, Set[str]] = {}

        # ---- connectivity (from NS3 topology file) ----
        self.switch_connections: Dict[str, Set[str]] = defaultdict(set)

        # ---- metadata ----
        self.gpus_per_node: int = 8

    # --- convenience counts ---

    @property
    def num_hosts(self) -> int:
        return len(self.hosts)

    @property
    def total_switches(self) -> int:
        return sum(len(s) for s in self.switches_by_type.values())

    def to_dict(self) -> dict:
        """Serialise to a plain dict (for JSON export)."""
        return {
            "num_hosts":        self.num_hosts,
            "total_switches":   self.total_switches,
            "gpus_per_node":    self.gpus_per_node,
            "hosts":            sorted(self.hosts),
            "switches_by_type": {
                t: sorted(nodes)
                for t, nodes in self.switches_by_type.items()
            },
        }


# ---------------------------------------------------------------------------
# NodemapParser
# ---------------------------------------------------------------------------

class NodemapParser:
    """
    Parses a nodemap.json (and optionally an NS3 topology file) to produce a
    TopologyInfo.

    Node classification is driven purely by the prefix_map argument – no
    topology type needs to be specified.
    """

    def __init__(self,
                 nodemap_path: str,
                 topology_path: Optional[str] = None,
                 gpus_per_node: int = 8,
                 prefix_map: Optional[Dict[str, str]] = None,
                 host_prefix: str = DEFAULT_HOST_PREFIX):
        """
        Parameters
        ----------
        nodemap_path  : path to nodemap.json
        topology_path : optional path to the NS3 topology file (used to derive
                        switch connectivity for the degree fallback)
        gpus_per_node : assumed GPUs per host node
        prefix_map    : dict mapping type_name → name_prefix.
                        Defaults to DEFAULT_PREFIX_MAP.
        host_prefix   : prefix that identifies host nodes (default "h")
        """
        self.nodemap_path  = nodemap_path
        self.topology_path = topology_path
        self.gpus_per_node = gpus_per_node
        self.host_prefix   = host_prefix.lower()
        self.prefix_map    = {k: v.lower() for k, v in (prefix_map or DEFAULT_PREFIX_MAP).items()}

        self.nodemap: Dict[str, str] = {}
        self.topology_info = TopologyInfo()
        self.topology_info.gpus_per_node = gpus_per_node

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def load_nodemap(self) -> Dict[str, str]:
        """Load and return the nodemap.json dict."""
        try:
            with open(self.nodemap_path, 'r') as f:
                self.nodemap = json.load(f)
            print(f"✅ Loaded nodemap: {self.nodemap_path}  ({len(self.nodemap)} nodes)")
            return self.nodemap
        except (IOError, json.JSONDecodeError) as e:
            print(f"❌ Failed to load nodemap: {e}")
            return {}

    def classify_node(self, node_name: str) -> str:
        """
        Return the type name for node_name by matching against the prefix map.

        Returns 'host', a key from prefix_map, or 'unknown'.
        """
        nl = node_name.lower()
        if nl.startswith(self.host_prefix):
            return "host"
        for type_name, prefix in self.prefix_map.items():
            if nl.startswith(prefix):
                return type_name
        return "unknown"

    def parse_topology_file(self):
        """
        Parse the NS3 topology file to fill switch_connections.
        Only called when topology_path is provided.
        """
        if not self.topology_path:
            return
        try:
            with open(self.topology_path, 'r') as f:
                lines = f.readlines()

            line_idx = 0
            switch_count = 0

            # Header: "<num_nodes> <num_switches> <extra>"
            if line_idx < len(lines):
                parts = lines[line_idx].strip().split()
                if len(parts) >= 2:
                    switch_count = int(parts[1])
                line_idx += 1

            # Skip switch-ID list
            for _ in range(switch_count):
                if line_idx < len(lines):
                    line_idx += 1

            # Parse link lines until ROUTES
            while line_idx < len(lines):
                line = lines[line_idx].strip()
                line_idx += 1
                if not line or line == "ROUTES":
                    break
                parts = line.split()
                if len(parts) >= 2:
                    src, dst = parts[0], parts[1]
                    # Translate numeric IDs → named IDs if possible
                    if src.isdigit():
                        src = self.nodemap.get(src, src)
                    if dst.isdigit():
                        dst = self.nodemap.get(dst, dst)
                    self.topology_info.switch_connections[src].add(dst)
                    self.topology_info.switch_connections[dst].add(src)

            print(f"✅ Parsed topology connectivity: {self.topology_path}")
        except (IOError, ValueError) as e:
            print(f"⚠️  Could not parse topology file: {e}")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def analyze_topology(self) -> TopologyInfo:
        """
        Build and return the TopologyInfo for this nodemap.
        """
        if not self.nodemap:
            self.load_nodemap()
        if self.topology_path:
            self.parse_topology_file()

        # Classify every node
        for _numeric_id, node_name in self.nodemap.items():
            node_type = self.classify_node(node_name)
            if node_type == "host":
                self.topology_info.hosts.add(node_name)
            elif node_type != "unknown":
                self.topology_info.switches_by_type.setdefault(node_type, set())
                self.topology_info.switches_by_type[node_type].add(node_name)
            # 'unknown' nodes are silently ignored

        return self.topology_info

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(self):
        """Print a human-readable topology summary."""
        info = self.topology_info
        print("\n" + "=" * 70)
        print("TOPOLOGY PREPROCESSING SUMMARY")
        print("=" * 70)
        print(f"{'Host nodes (GPUs/NPUs)':<35} {info.num_hosts}")
        print(f"{'GPUs per node (assumed)':<35} {info.gpus_per_node}")
        print(f"{'Total switches':<35} {info.total_switches}")
        for type_name, nodes in sorted(info.switches_by_type.items()):
            label = f"  {type_name} switches"
            print(f"{label:<35} {len(nodes)}")
        print("=" * 70 + "\n")

    def save_to_json(self, output_path: str):
        """Save TopologyInfo to a JSON file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(self.topology_info.to_dict(), f, indent=2)
            print(f"✅ Saved topology info → {output_path}")
        except IOError as e:
            print(f"❌ Failed to save topology info: {e}")


# ---------------------------------------------------------------------------
# Helper: load from a previously saved JSON
# ---------------------------------------------------------------------------

def load_topology_info_from_json(json_path: str) -> Optional[TopologyInfo]:
    """Load a TopologyInfo that was previously saved with save_to_json()."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        info = TopologyInfo()
        info.gpus_per_node = data.get("gpus_per_node", 8)
        info.hosts = set(data.get("hosts", []))
        info.switches_by_type = {
            t: set(nodes) for t, nodes in data["switches_by_type"].items()
        }
        print(f"✅ Loaded topology info ← {json_path}")
        return info
    except (IOError, json.JSONDecodeError, KeyError) as e:
        print(f"❌ Error loading topology info: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: nodemap_parser.py <nodemap.json> [topology_file] [gpus_per_node]")
        sys.exit(1)
    nodemap_path   = sys.argv[1]
    topology_path  = sys.argv[2] if len(sys.argv) > 2 else None
    gpus_per_node  = int(sys.argv[3]) if len(sys.argv) > 3 else 8

    parser = NodemapParser(nodemap_path, topology_path, gpus_per_node)
    parser.analyze_topology()
    parser.print_summary()
    out = nodemap_path.replace(".json", "_topology_info.json")
    parser.save_to_json(out)


if __name__ == "__main__":
    main()
