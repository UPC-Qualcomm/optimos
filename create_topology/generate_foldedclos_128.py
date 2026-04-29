#!/usr/bin/env python3
"""
Generate topology files for various 128-NPU Folded-Clos (Fat-Tree) configurations.

This script creates the necessary configuration files for different network setups,
all targeting a total of 128 NPUs.
"""
import os
import sys

# Add the parent directory to the path to import the required modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from create_topology_main import generate_topology_files

# Define output directories for the generated topology files
G2_OUTPUT = os.path.join(os.environ.get("OPTIMOS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))), "configuration", "g2", "topologies")
NS3_OUTPUT = os.path.join(os.environ.get("OPTIMOS_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))), "configuration", "ns3", "topologies")

# Ensure the output directories exist
os.makedirs(G2_OUTPUT, exist_ok=True)
os.makedirs(NS3_OUTPUT, exist_ok=True)

print("=" * 80)
print("Generating 128-NPU Folded-Clos Topology Files")
print("=" * 80)

# =============================================================================
# Option 1: Basic FoldedClos (K=8) - 128 servers, 1 NPU each
# =============================================================================
print("\n--- Generating: Basic K=8 Fat-Tree (128 servers, 1 NPU each) ---")
# A K=8 fat-tree has (K^3)/4 = (8*8*8)/4 = 128 servers/NPUs.
basic_bw = {
    'host_edge': 100,
    'edge_agg': 100,
    'agg_core': 100,
}
basic_config = {
    'K': 8,
    'bandwidth_config': basic_bw,
    'bw_unit': 'GB/s'
}
generate_topology_files(
    topology="FoldedClos",
    paths_mode="ECMP",
    config=basic_config,
    base_filename="FoldedClos_128npus_basic",
    output_dir=G2_OUTPUT
)

# =============================================================================
# Option 2: FoldedClos (K=4) with NVSwitch - 16 servers, 8 NPUs each
# =============================================================================
print("\n--- Generating: K=4 Fat-Tree with NVSwitch (16 servers, 8 NPUs each) ---")
# A K=4 fat-tree has (K^3)/4 = (4*4*4)/4 = 16 servers.
# With 8 NPUs per server, total NPUs = 16 * 8 = 128.
nvswitch_bw = {
    'host_edge': 50,
    'edge_agg': 100,
    'agg_core': 100,
    'intra_node': 200,  # Bandwidth between NPUs and the NVSwitch
}
nvswitch_config = {
    'K': 4,
    'bandwidth_config': nvswitch_bw,
    'bw_unit': 'GB/s',
    'npus_per_node': 8,
    'intra_node_topology': 'switch',
    'num_nvswitches': 2  # Using 2 NVSwitches per node for redundancy/performance
}
generate_topology_files(
    topology="FoldedClos",
    paths_mode="ECMP",
    config=nvswitch_config,
    base_filename="FoldedClos_128npus_nvswitch",
    output_dir=G2_OUTPUT
)

# =============================================================================
# Option 3: FoldedClos (K=4) with 3-Level Hierarchy
# 16 servers, 2 nodes/server, 4 NPUs/node
# =============================================================================
print("\n--- Generating: K=4 Fat-Tree with 3-Level Hierarchy ---")
# 16 servers * 2 nodes/server * 4 NPUs/node = 128 total NPUs.
threelevel_bw = {
    'host_edge': 50,
    'edge_agg': 100,
    'agg_core': 100,
    'intra_node': 200,
}
threelevel_config = {
    'K': 4,
    'bandwidth_config': threelevel_bw,
    'bw_unit': 'GB/s',
    'nodes_per_server': 2,
    'npus_per_node': 4,
    'intra_node_topology': 'switch',
    'num_nvswitches': 1
}
generate_topology_files(
    topology="FoldedClos",
    paths_mode="ECMP",
    config=threelevel_config,
    base_filename="FoldedClos_128npus_3level",
    output_dir=G2_OUTPUT
)

# =============================================================================
# Option 4: FoldedClos (K=4) with Ring Intra-node
# 16 servers, 8 NPUs each connected in a ring
# =============================================================================
print("\n--- Generating: K=4 Fat-Tree with Ring Intra-node (16 servers, 8 NPUs each) ---")
ring_bw = {
    'host_edge': 50,
    'edge_agg': 100,
    'agg_core': 100,
    'intra_node': 150,  # Bandwidth for links in the intra-node ring
}
ring_config = {
    'K': 4,
    'bandwidth_config': ring_bw,
    'bw_unit': 'GB/s',
    'npus_per_node': 8,
    'intra_node_topology': 'ring'
}
generate_topology_files(
    topology="FoldedClos",
    paths_mode="ECMP",
    config=ring_config,
    base_filename="FoldedClos_128npus_ring",
    output_dir=G2_OUTPUT
)

print("\n" + "=" * 80)
print("All 128-NPU Folded-Clos topologies generated successfully!")
print(f"Output directory: {G2_OUTPUT}")
print("=" * 80)
