from create_topology_main import generate_topology_files
import os
# Define output paths
output_path = os.environ["OPTIMOS_ROOT"] + "/configuration/topologies"

# =============================================================================
# BASIC TOPOLOGIES (without intra-node topology)
# =============================================================================

# --- Dragonfly Configuration (Basic) ---
# G=4 groups, A=4 switches/group, h=2 inter-group links, 1 node/switch
# Total Nodes = 4 * 4 * 1 = 16
# df_bw = {'host_switch': 100, 'intra_group': 50, 'inter_group': 25}
# df_config = {'G': 4, 'A': 4, 'h': 2, 'concentration': 1, 'bandwidth_config': df_bw, 'bw_unit': 'GB/s'}
# generate_topology_files(topology="Dragonfly", paths_mode="ECMP", config=df_config)

# --- Jellyfish Configuration (Basic) ---
# 16 switches, degree 4, 1 node/switch
# Total Nodes = 16 * 1 = 16
# jf_bw = {'host_switch': 100, 'switch_switch': 50}
# jf_config = {'num_switches': 16, 'degree': 4, 'num_hosts_per_switch': 1, 'bandwidth_config': jf_bw, 'bw_unit': 'GB/s'}
# generate_topology_files(topology="Jellyfish", paths_mode="Uniform", config=jf_config)

# --- Folded Clos (Basic) ---
# K=4 (4-port switches), Total Nodes = (4^3) / 4 = 16
#fc_bw = {'host_edge': 25, 'edge_agg': 25, 'agg_core': 25}
#fc_config = {'K': 4, 'bandwidth_config': fc_bw, 'bw_unit': 'GB/s'}
#generate_topology_files(topology="FoldedClos", paths_mode="ECMP", config=fc_config, base_filename="FoldedClos_16nodes_basic")

# =============================================================================
# TOPOLOGIES WITH INTRA-NODE TOPOLOGY (Multiple NPUs per node)
# =============================================================================

# --- Folded Clos with NVSwitch intra-node topology ---
# K=4 -> 16 nodes, 8 NPUs per node = 128 total NPUs
# Intra-node: switch topology with 1 NVSwitch
# Bandwidth: 75 GB/s intra-node, 25 GB/s inter-node (to edge)
fc_bw_intra = {
    'host_edge': 200,      # Inter-node bandwidth (node to edge switch)
    'edge_agg': 200,       # Edge to aggregation 
    'agg_core': 200,       # Aggregation to core
    'intra_node': 900      # Intra-node bandwidth (NPU to NVSwitch)
}
fc_config_intra = {
    'K': 4,
    'bandwidth_config': fc_bw_intra,
    'bw_unit': 'GB/s',
    'npus_per_node': 8,
    'intra_node_topology': 'switch',  # Options: 'switch', 'ring', 'fully_connected'
    'num_nvswitches': 4               # Number of NVSwitches (for 'switch' topology)
}
generate_topology_files(topology="FoldedClos", paths_mode="ECMP", config=fc_config_intra, 
                        output_dir=output_path, base_filename="FoldedClos_128npus_switch")

# --- Folded Clos with Ring intra-node topology ---
# Ring topology: NPUs connected in a ring
#fc_config_ring = {
#    'K': 4,
#    'bandwidth_config': fc_bw_intra,
#    'bw_unit': 'GB/s',
#    'npus_per_node': 8,
#    'intra_node_topology': 'ring'
#}
# generate_topology_files(topology="FoldedClos", paths_mode="ECMP", config=fc_config_ring, 
#                         base_filename="FoldedClos_128npus_ring")

# --- Folded Clos with Fully Connected intra-node topology ---
# All NPUs within a node fully connected
#fc_config_fc = {
#    'K': 4,
#    'bandwidth_config': fc_bw_intra,
#    'bw_unit': 'GB/s',
#    'npus_per_node': 4,  # Smaller for fully connected
#    'intra_node_topology': 'fully_connected'
#}
# generate_topology_files(topology="FoldedClos", paths_mode="ECMP", config=fc_config_fc, 
#                         base_filename="FoldedClos_64npus_fullmesh")

# --- Dragonfly with NVSwitch intra-node topology ---
# G=4 groups, A=4 switches/group, 2 nodes/switch, 8 NPUs/node
# Total: 4 * 4 * 2 * 8 = 256 NPUs
#df_bw_intra = {
#    'host_switch': 25,    # Inter-node bandwidth
#    'intra_group': 50,    # Intra-group switch links
#    'inter_group': 25,    # Inter-group links
#    'intra_node': 75      # Intra-node bandwidth
#}
#df_config_intra = {
#    'G': 4,
#    'A': 4,
#    'h': 2,
#    'concentration': 2,   # 2 nodes per switch
#    'bandwidth_config': df_bw_intra,
#    'bw_unit': 'GB/s',
#    'npus_per_node': 8,
#    'intra_node_topology': 'switch',
#    'num_nvswitches': 1   # 2 NVSwitches (doubles effective intra-node BW)
#}
# generate_topology_files(topology="Dragonfly", paths_mode="ECMP", config=df_config_intra,
#                         base_filename="Dragonfly_256npus_switch")

# --- Jellyfish with Ring intra-node topology ---
# 16 switches, 2 nodes/switch, 4 NPUs/node
# Total: 16 * 2 * 4 = 128 NPUs
#jf_bw_intra = {
#    'host_switch': 25,    # Inter-node bandwidth
#    'switch_switch': 50,  # Switch-to-switch links
#    'intra_node': 100     # Intra-node bandwidth
#}
#jf_config_intra = {
#    'num_switches': 16,
#    'degree': 4,
#    'num_hosts_per_switch': 2,  # 2 nodes per switch
#    'bandwidth_config': jf_bw_intra,
#    'bw_unit': 'GB/s',
#    'npus_per_node': 4,
#    'intra_node_topology': 'ring'
#}
# generate_topology_files(topology="Jellyfish", paths_mode="ECMP", config=jf_config_intra,
#                         base_filename="Jellyfish_128npus_ring")