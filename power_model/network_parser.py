"""
Network Statistics Parser

Parses the simplified link-traffic CSV output from AstraSim and integrates
with the nodemap parser to produce topology-aware NetworkStats.

The parser is topology-agnostic: node classification relies entirely on the
prefix_map provided (or the PowerConfig's switch_type_prefixes), so the same
code works for FoldedClos, Jellyfish, Dragonfly, etc.
"""

import csv
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from nodemap_parser import NodemapParser, TopologyInfo, load_topology_info_from_json


@dataclass
class LinkStats:
    """Statistics for a single directed network link."""
    src_id: str
    dst_id: str
    bytes_transmitted: float
    bandwidth: float = 0.0      # Bytes/sec (link capacity)
    link_type: str = "unknown"  # 'nvlink', 'nic', 'network', 'unknown'

    def utilization(self, total_time: float) -> float:
        """Link utilisation in [0, ∞) (may exceed 1 if over-subscribed)."""
        if total_time <= 0 or self.bandwidth <= 0:
            return 0.0
        return self.bytes_transmitted / (self.bandwidth * total_time)


class NetworkStats:
    """
    Complete network statistics:
    - Link-traffic data (from link_traffic.csv)
    - Topology information (from nodemap parser)
    """

    def __init__(self):
        self.links: List[LinkStats] = []
        self.total_bytes_transmitted: float = 0.0
        self.topology_info: Optional[TopologyInfo] = None

    # ------------------------------------------------------------------
    # Topology loading
    # ------------------------------------------------------------------

    def set_topology_info(self, topo: TopologyInfo):
        """Attach a TopologyInfo object."""
        self.topology_info = topo

    # ------------------------------------------------------------------
    # Link-type classification  (topology-aware, no hardcoded prefixes)
    # ------------------------------------------------------------------

    def _build_node_type_map(self) -> Dict[str, str]:
        """Build a name → type_name map from topology_info."""
        result: Dict[str, str] = {}
        topo = self.topology_info
        if not topo:
            return result
        for node in topo.hosts:
            result[node] = "host"
        for type_name, nodes in topo.switches_by_type.items():
            for node in nodes:
                result[node] = type_name
        return result

    def classify_link_type(self, src: str, dst: str) -> str:
        """
        Return the link-type classification for a directed link src → dst.

        Classification rules (in order):
          1. If either endpoint is 'nvswitch'  → 'nvlink'
          2. If either endpoint is 'nic'       → 'nic'
          3. If either endpoint is 'per_npu'   → 'per_npu'
          4. Fabric links – return the actual switch-type name of the network
             endpoint: prefer dst_type (link terminates at dst switch);
             fall back to src_type when dst is a host or unknown.
          5. Otherwise → 'unknown'
        """
        topo = self.topology_info
        if not topo:
            return "unknown"

        node_types = self._build_node_type_map()
        src_type = node_types.get(src, "unknown")
        dst_type = node_types.get(dst, "unknown")

        # NVSwitch wires (NVLink)
        if "nvswitch" in (src_type, dst_type):
            return "nvlink"
        # NIC wires
        if "nic" in (src_type, dst_type):
            return "nic"
        # per_npu – returned as-is; caller uses per_npu power params
        if "per_npu" in (src_type, dst_type):
            return "per_npu"
        # Fabric links: return the actual switch-type of the relevant endpoint.
        # Prefer dst_type (the link terminates at the dst switch);
        # fall back to src_type if dst is a host or unknown.
        LEAF_TYPES = {"host", "unknown"}
        if dst_type not in LEAF_TYPES:
            return dst_type
        if src_type not in LEAF_TYPES:
            return src_type

        return "unknown"

    def get_link_by_direction(self, src: str, dst: str) -> Optional[LinkStats]:
        """Return the LinkStats for the directed link src → dst, or None."""
        for lnk in self.links:
            if lnk.src_id == src and lnk.dst_id == dst:
                return lnk
        return None


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------

def parse_link_traffic_csv(csv_file_path: str,
                           topology_info: Optional[TopologyInfo] = None) -> NetworkStats:
    """
    Parse a link_traffic.csv file (columns: Source, Destination, Total_Bytes, Bandwidth [optional]).

    Parameters
    ----------
    csv_file_path     : path to link_traffic.csv
    default_bandwidth : fallback per-link capacity in Bytes/s if CSV has no Bandwidth column
    topology_info     : optional TopologyInfo for link-type classification

    Notes
    -----
    If the CSV includes a "Bandwidth (BG/s)" or similar column, it is parsed and converted
    from GB/s to Bytes/s (1024 * 1024 * 1024).  Otherwise default_bandwidth is used.
    """
    stats = NetworkStats()
    if topology_info:
        stats.set_topology_info(topology_info)

    try:
        with open(csv_file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                src      = row['Source']
                dst      = row['Destination']
                bytes_tx = float(row['Total_Bytes'])
                ltype    = stats.classify_link_type(src, dst) if topology_info else "unknown"
                
                # Try to parse per-link bandwidth from CSV; fall back to default
                link_bandwidth = 0
                for bw_col in ['Bandwidth (BG/s)', 'Bandwidth (GB/s)', 'Bandwidth', 'bandwidth']:
                    if bw_col in row and row[bw_col]:
                        try:
                            # Column is in GB/s; convert to Bytes/s
                            link_bandwidth = float(row[bw_col]) * 1024 * 1024 * 1024
                            break
                        except ValueError:
                            continue

                stats.links.append(LinkStats(
                    src_id=src,
                    dst_id=dst,
                    bytes_transmitted=bytes_tx,
                    bandwidth=link_bandwidth,
                    link_type=ltype,
                ))
                stats.total_bytes_transmitted += bytes_tx

        print(f"✅ Parsed {len(stats.links)} links  |  "
              f"total bytes: {stats.total_bytes_transmitted:.2e}")

    except (IOError, KeyError, ValueError) as e:
        print(f"❌ Error parsing link-traffic CSV: {e}")

    return stats


def parse_network_statistics_csv(csv_file_path: str,
                                 nodemap_file: Optional[str] = None,
                                 topology_file: Optional[str] = None,
                                 prefix_map: Optional[Dict[str, str]] = None,
                                 host_prefix: str = "h",
                                 gpus_per_node: int = 8) -> NetworkStats:
    """
    Parse network statistics with optional nodemap-based topology preprocessing.

    Parameters
    ----------
    csv_file_path     : path to link_traffic.csv
    default_bandwidth : per-link capacity in Bytes/s
    nodemap_file      : path to nodemap.json (enables topology-aware parsing)
    topology_file     : path to NS3 topology file (enables degree derivation)
    prefix_map        : type_name → name_prefix mapping (overrides DEFAULT_PREFIX_MAP).
                        Pass PowerConfig.switch_type_prefixes here.
    host_prefix       : prefix identifying host nodes (default 'h')
    gpus_per_node     : GPUs/NPUs per host node
    """
    topology_info = None
    if nodemap_file:
        parser = NodemapParser(
            nodemap_path=nodemap_file,
            topology_path=topology_file,
            gpus_per_node=gpus_per_node,
            prefix_map=prefix_map,
            host_prefix=host_prefix,
        )
        topology_info = parser.analyze_topology()
        parser.print_summary()

    return parse_link_traffic_csv(csv_file_path, topology_info)

