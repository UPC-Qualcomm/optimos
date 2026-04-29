import re
import json
import matplotlib.patches as mpatches
import networkx as nx
import matplotlib.pyplot as plt

def write_g2_topology_files(links, paths, base_filename="topology", bandwidth=900, link_bandwidths=None, link_latencies=None, output_dir="./topologies/g2", path_gen=None):
    """
    Writes topology data to two files:
    1. A custom .txt file with a simple, quote-less format.
    2. A standard, machine-readable .json file.
    
    The paths in both files are filtered to only include host-to-host routes.
    Node names are renumbered using the same sequential scheme as the NS3 file
    (hosts: 0..num_hosts-1, switches: num_hosts..num_nodes-1), but the original
    type prefix is preserved (e.g., 'h0', 't128', 'v132', 'a140', 'c150').
    This means the same number in the NS3 file and these files always refers to
    the same node.

    path_gen: optional callable that returns a generator yielding
              (src_name, {dst_name: [list_of_paths]}) pairs in order.
              When provided, 'paths' is ignored for the routes section.
              For very large topologies the routes section is omitted from the
              g2 files (use the NS3 file for routing); only topology edges are
              written.

    Args:
        links (dict): A dictionary of links from the topology generator.
        paths (dict): A dictionary of all possible paths from the generator.
        base_filename (str): The base name for the output files.
        bandwidth (int): The default bandwidth for each link if not specified in link_bandwidths.
        link_bandwidths (dict): Optional dictionary mapping link IDs to bandwidth values.
        link_latencies (dict): Optional dictionary mapping link IDs to latency values.
    """
    def _get_num(name):
        match = re.search(r'\d+', name)
        return int(match.group()) if match else -1

    def _get_prefix(name):
        match = re.match(r'([a-zA-Z]+)', name)
        return match.group(1) if match else ''

    # --- 0. Build the sequential name -> labeled-name mapping (same ordering as NS3 writer) ---
    all_node_names = set()
    for _, (n1, n2) in links.items():
        all_node_names.add(n1)
        all_node_names.add(n2)

    host_names   = sorted([n for n in all_node_names if n.startswith('h')], key=_get_num)
    switch_names = sorted([n for n in all_node_names if not n.startswith('h')], key=_get_num)
    num_hosts = len(host_names)

    # Map each original name to  prefix + sequential_id  (matching NS3 numbering)
    name_to_labeled = {}
    for i, name in enumerate(host_names):
        name_to_labeled[name] = f"{_get_prefix(name)}{i}"
    for i, name in enumerate(switch_names):
        name_to_labeled[name] = f"{_get_prefix(name)}{num_hosts + i}"

    def relabel(name):
        return name_to_labeled.get(name, name)

    # --- 1. Build host-to-host paths (or note they will be streamed / skipped) ---
    streaming = path_gen is not None

    if not streaming:
        host_to_host_paths = {}
        for src, dests in (paths or {}).items():
            if src.startswith('h'):
                host_dests = {}
                for dest, path_or_paths in dests.items():
                    if dest.startswith('h') and src != dest:
                        if path_or_paths and isinstance(path_or_paths[0], list):
                            relabeled = [[relabel(hop) for hop in p] for p in path_or_paths]
                        else:
                            relabeled = [relabel(hop) for hop in path_or_paths]
                        host_dests[relabel(dest)] = relabeled
                if host_dests:
                    host_to_host_paths[relabel(src)] = host_dests

    # --- 2. Create and write the custom .txt file ---
    import os
    os.makedirs(output_dir, exist_ok=True)
    txt_filename = os.path.join(output_dir, f"{base_filename}.txt")
    with open(txt_filename, "w") as f:
        # Edges
        for link_id, (node1, node2) in links.items():
            bw = bandwidth
            if link_bandwidths and link_id in link_bandwidths:
                bw = link_bandwidths[link_id]
            lat = 0.0
            if link_latencies and link_id in link_latencies:
                lat = link_latencies[link_id]
            f.write(f"({relabel(node1)}, {relabel(node2)}, {bw}, {lat})\n")

        f.write("\n# Paths\n\n")

        if streaming:
            # Large topology: stream paths directly to file
            for src_orig, dests in path_gen():
                src_lbl = relabel(src_orig)
                for dest_orig, path_or_paths in dests.items():
                    dest_lbl = relabel(dest_orig)
                    if path_or_paths and isinstance(path_or_paths[0], list):
                        for path in path_or_paths:
                            f.write(f"{src_lbl}: {dest_lbl}: [{', '.join(relabel(h) for h in path)}]\n")
                    else:
                        f.write(f"{src_lbl}: {dest_lbl}: [{', '.join(relabel(h) for h in path_or_paths)}]\n")
        else:
            for src, dests in host_to_host_paths.items():
                for dest, path_or_paths in dests.items():
                    if path_or_paths and isinstance(path_or_paths[0], list):
                        for path in path_or_paths:
                            f.write(f"{src}: {dest}: [{', '.join(path)}]\n")
                    else:
                        f.write(f"{src}: {dest}: [{', '.join(path_or_paths)}]\n")

    print(f"Successfully wrote custom text topology to {txt_filename}")

    # --- 3. Create and write the standard .json file ---
    edges = []
    for link_id, (n1, n2) in links.items():
        bw = bandwidth
        if link_bandwidths and link_id in link_bandwidths:
            bw = link_bandwidths[link_id]
        lat = 0.0
        if link_latencies and link_id in link_latencies:
            lat = link_latencies[link_id]
        edges.append([relabel(n1), relabel(n2), bw, lat])

    json_filename = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_filename, "w") as f:
        # Write JSON incrementally to avoid building a 2+ GB dict in memory
        f.write('{\n')
        f.write(f'    "numEdges": {len(links)},\n')
        f.write('    "edges": ')
        json.dump(edges, f)
        f.write(',\n')
        f.write('    "paths": {')

        first_src = True
        if streaming:
            for src_orig, dests in path_gen():
                src_lbl = relabel(src_orig)
                if not first_src:
                    f.write(',')
                f.write(f'\n        {json.dumps(src_lbl)}: {{')
                first_dst = True
                for dest_orig, path_or_paths in dests.items():
                    dest_lbl = relabel(dest_orig)
                    if not first_dst:
                        f.write(',')
                    norm = path_or_paths if (path_or_paths and isinstance(path_or_paths[0], list)) else [path_or_paths]
                    relabeled = [[relabel(h) for h in p] for p in norm]
                    f.write(f'\n            {json.dumps(dest_lbl)}: {json.dumps(relabeled)}')
                    first_dst = False
                f.write('\n        }')
                first_src = False
        else:
            for src, dests in host_to_host_paths.items():
                if not first_src:
                    f.write(',')
                f.write(f'\n        {json.dumps(src)}: {json.dumps(dests)}')
                first_src = False

        f.write('\n    }\n}\n')
    print(f"Successfully wrote JSON topology to {json_filename}")


def write_ns3_topology_file(links, paths, filename="ns3_topology.txt", bandwidth="900GB/s", latency="0.000ms", link_bandwidths=None, link_latencies=None, bw_unit="GB/s", lat_unit="ms", output_dir="./topologies/ns3", path_gen=None):
    """
    Maps node names to sequential IDs and writes a topology file in the NS3 format,
    including pre-computed routes.
    - Hosts are mapped to IDs [0, num_hosts - 1].
    - Switches are mapped to IDs [num_hosts, num_hosts + num_switches - 1].

    Args:
        links (dict): Dictionary of links with string node names (e.g., 'h1', 's1').
        paths (dict): Dictionary of paths with string node names (ignored when path_gen
                      is provided).
        filename (str): The name of the output file.
        bandwidth (str): Default bandwidth for all links.
        latency (str): Default link latency.
        link_bandwidths (dict): Optional dictionary mapping link IDs to bandwidth values.
        link_latencies (dict): Optional dictionary mapping link IDs to latency values.
        bw_unit (str): Unit string to append to bandwidth values (e.g., "GB/s").
        lat_unit (str): Unit string to append to latency values (e.g., "ms").
        path_gen: optional callable returning a generator that yields
                  (src_name, {dst_name: [list_of_paths]}) in source-sorted order.
                  Use this for large topologies to avoid building a huge paths dict.
    """
    def get_num(name):
        """Extracts the integer part of a node name for sorting."""
        match = re.search(r'\d+', name)
        return int(match.group()) if match else -1

    # 1. Discover all unique nodes and categorize them
    all_node_names = set()
    for _, (n1, n2) in links.items():
        all_node_names.add(n1)
        all_node_names.add(n2)

    host_names = sorted([name for name in all_node_names if name.startswith('h')], key=get_num)
    switch_names = sorted([name for name in all_node_names if not name.startswith('h')], key=get_num)

    num_hosts = len(host_names)
    num_switches = len(switch_names)
    num_nodes = num_hosts + num_switches
    
    # Calculate unique links and map bandwidths and latencies
    unique_links = set()
    pair_to_bw = {}
    pair_to_lat = {}
    
    if link_bandwidths:
        for lid, (n1, n2) in links.items():
            pair = tuple(sorted((n1, n2)))
            pair_to_bw[pair] = link_bandwidths[lid]
    
    if link_latencies:
        for lid, (n1, n2) in links.items():
            pair = tuple(sorted((n1, n2)))
            pair_to_lat[pair] = link_latencies[lid]

    for n1, n2 in links.values():
        # Store links in a canonical order (smaller name first) to handle duplicates
        sorted_pair = tuple(sorted((n1, n2)))
        unique_links.add(sorted_pair)
    num_links = len(unique_links)


    # 2. Create the mapping from name to new sequential ID
    name_to_id_map = {}
    switch_ids = []
    for i, name in enumerate(host_names):
        new_id = i
        name_to_id_map[name] = new_id

    for i, name in enumerate(switch_names):
        new_id = num_hosts + i
        name_to_id_map[name] = new_id
        switch_ids.append(new_id)

    # 3. Process links using the new IDs
    processed_links_with_bw_lat = []
    for n1_str, n2_str in unique_links:
        id1 = name_to_id_map[n1_str]
        id2 = name_to_id_map[n2_str]
        
        pair_key = tuple(sorted((n1_str, n2_str)))
        bw_val = bandwidth
        if pair_key in pair_to_bw:
            val = pair_to_bw[pair_key]
            # Format if int/float to string with unit if needed, or just string
            if isinstance(val, (int, float)):
                bw_val = f"{val}{bw_unit}"
            else:
                bw_val = str(val)
        
        lat_val = latency
        if pair_key in pair_to_lat:
            val = pair_to_lat[pair_key]
            if isinstance(val, (int, float)):
                lat_val = f"{val}{lat_unit}"
            else:
                lat_val = str(val)
        
        processed_links_with_bw_lat.append((tuple(sorted((id1, id2))), bw_val, lat_val))

    # 4. Write to file
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        # Header
        f.write(f"{num_nodes} {num_switches} {num_links}\n")
        
        # Switch IDs (already sorted)
        for switch_id in switch_ids:
            f.write(f"{switch_id}\n")
            
        # Links
        for (id1, id2), bw, lat in sorted(processed_links_with_bw_lat, key=lambda x: x[0]):
            f.write(f"{id1} {id2} {bw} {lat} 0\n")

        # 5. Write paths (ROUTES section)
        f.write("\nROUTES\n")

        if path_gen is not None:
            # Streaming mode: generator yields (src_name, {dst_name: [paths]}) in order.
            # Sources are already in sequential order so no outer sort is needed.
            for src_str, dests in path_gen():
                if src_str not in name_to_id_map:
                    continue
                src_id = name_to_id_map[src_str]
                for dest_str, list_of_paths in sorted(dests.items(), key=lambda item: get_num(item[0])):
                    if dest_str not in name_to_id_map:
                        continue
                    dest_id = name_to_id_map[dest_str]
                    for path in list_of_paths:
                        path_ids = [name_to_id_map[hop] for hop in path if hop in name_to_id_map]
                        f.write(f"{src_id}:{dest_id}:[{', '.join(map(str, path_ids))}]\n")
        else:
            # Original dict-based mode
            host_to_host_paths = {}
            for src, dests in (paths or {}).items():
                if src.startswith('h'):
                    host_dests = {
                        dest: path_list for dest, path_list in dests.items()
                        if dest.startswith('h') and src != dest
                    }
                    if host_dests:
                        host_to_host_paths[src] = host_dests

            for src_str, dests in sorted(host_to_host_paths.items(), key=lambda item: get_num(item[0])):
                src_id = name_to_id_map[src_str]
                for dest_str, list_of_paths in sorted(dests.items(), key=lambda item: get_num(item[0])):
                    dest_id = name_to_id_map[dest_str]
                    for path in list_of_paths:
                        path_ids = [name_to_id_map[hop] for hop in path]
                        path_ids_str = ", ".join(map(str, path_ids))
                        f.write(f"{src_id}:{dest_id}:[{path_ids_str}]\n")

    print(f"Successfully wrote NS3 topology to {filepath}")

    # 6. Write nodemap sidecar: plain_int_id -> prefixed_name  (e.g. "32" -> "t32")
    # This lets the power model translate plain simulation IDs back to typed names.
    # Prefix key:
    #   h = NPU/host    v = NVSwitch       n = NIC switch (ring/fully_connected)
    #   p = per-NPU switch                 t = edge/ToR switch
    #   a = aggregation switch             c = core switch
    def _get_prefix(name):
        match = re.match(r'([a-zA-Z]+)', name)
        return match.group(1) if match else ''

    nodemap = {}
    for name, num_id in name_to_id_map.items():
        prefix = _get_prefix(name)
        nodemap[str(num_id)] = f"{prefix}{num_id}"

    nodemap_path = os.path.join(output_dir, f"{filename}_nodemap.json")
    sorted_nodemap = {k: nodemap[k] for k in sorted(nodemap, key=lambda k: int(k))}
    with open(nodemap_path, "w") as f:
        json.dump(sorted_nodemap, f, indent=4)
    print(f"Successfully wrote node map to {nodemap_path}")


def make_node_names_zero_indexed(links, paths):
    """
    Converts 1-indexed node names ('h1', 's1') to 0-indexed names ('h0', 's0').
    This is primarily for G2 configuration files.
    
    Args:
        links (dict): Dictionary of links with 1-indexed string node names.
        paths (dict): Dictionary of paths with 1-indexed string node names.

    Returns:
        tuple: A tuple containing:
            - dict: A new links dictionary with 0-indexed names.
            - dict: A new paths dictionary with 0-indexed names.
    """
    def convert_name(name):
        # Use regex to handle multi-digit numbers correctly
        match = re.match(r"([a-zA-Z]+)([0-9]+)", name)
        if not match:
            return name # Return name unchanged if it doesn't fit the pattern
        
        prefix = match.group(1)
        number = int(match.group(2))
        
        # Subtract 1 only if the number is greater than 0
        if number > 0:
            return f"{prefix}{number - 1}"
        return name

    # Convert links
    new_links = {
        link_id: (convert_name(n1), convert_name(n2)) 
        for link_id, (n1, n2) in links.items()
    }

    # Convert paths
    new_paths = {}
    for src, dests in paths.items():
        new_src = convert_name(src)
        new_paths[new_src] = {}
        for dest, path_list in dests.items():
            new_dest = convert_name(dest)
            new_paths[new_src][new_dest] = [convert_name(hop) for hop in path_list]
            
    return new_links, new_paths


def classify_node(node_name, topology):
    """
    Classify a node into categories for proper positioning and coloring.
    Returns: 'npu', 'nvswitch', 'npu_switch', 'nic', 'edge_switch', 'agg_switch', 'core_switch', 'main_switch'
    
    Hierarchy levels:
    - npu: Processing units (bottom)
    - nvswitch/npu_switch: Intra-node switches connecting NPUs
    - nic: NIC switch connecting node to main switch (for ring/fully_connected)
    - edge_switch: Edge/ToR switches (FoldedClos only)
    - agg_switch: Aggregation switches (FoldedClos only)
    - core_switch: Core switches (FoldedClos only)
    - main_switch: For Dragonfly/Jellyfish (all switches are ToRs)
    """
    if node_name.startswith('h'):
        return 'npu'
    
    # Check NVSwitch (for 'switch' topology)
    if hasattr(topology, 'nvswitch_ids') and node_name in topology.nvswitch_ids:
        return 'nvswitch'
    
    # Check NPU switches (for ring/fully_connected - internal switches)
    if hasattr(topology, 'npu_switches') and node_name in topology.npu_switches:
        return 'npu_switch'
    
    # Check NIC switches (for ring/fully_connected - connects to main switch)
    if hasattr(topology, 'nic_switches') and node_name in topology.nic_switches:
        return 'nic'
    
    # FoldedClos: distinguish edge, aggregation, and core switches
    if hasattr(topology, 'core_switches') and node_name in topology.core_switches:
        return 'core_switch'
    if hasattr(topology, 'agg_switches') and node_name in topology.agg_switches:
        return 'agg_switch'
    if hasattr(topology, 'edge_switches') and node_name in topology.edge_switches:
        return 'edge_switch'
    
    return 'main_switch'

def create_hierarchical_layout(G, topology, width=10, height=8):
    """
    Create a hierarchical tree-like layout.
    
    For Dragonfly/Jellyfish (4 levels):
    - NPUs (level 0) -> NVSwitches (level 1) -> NICs (level 2) -> Switches (level 3)
    
    For FoldedClos (6 levels):
    - NPUs (level 0) -> NVSwitches (level 1) -> NICs (level 2) -> Edge/ToR (level 3) -> Agg (level 4) -> Core (level 5)
    """
    pos = {}
    
    # Classify all nodes
    npus = [n for n in G.nodes() if classify_node(n, topology) == 'npu']
    nvswitches = [n for n in G.nodes() if classify_node(n, topology) in ('nvswitch', 'npu_switch')]
    nics = [n for n in G.nodes() if classify_node(n, topology) == 'nic']
    edge_switches = [n for n in G.nodes() if classify_node(n, topology) == 'edge_switch']
    agg_switches = [n for n in G.nodes() if classify_node(n, topology) == 'agg_switch']
    core_switches = [n for n in G.nodes() if classify_node(n, topology) == 'core_switch']
    main_switches = [n for n in G.nodes() if classify_node(n, topology) == 'main_switch']
    
    # Determine number of levels based on topology type
    is_folded_clos = len(edge_switches) > 0 or len(agg_switches) > 0 or len(core_switches) > 0
    
    if is_folded_clos:
        num_levels = 6
        y_levels = {
            'npu': 0,
            'nvswitch': 1,
            'nic': 2,
            'edge_switch': 3,
            'agg_switch': 4,
            'core_switch': 5
        }
    else:
        num_levels = 4
        y_levels = {
            'npu': 0,
            'nvswitch': 1,
            'nic': 2,
            'main_switch': 3
        }
    
    # Position each category
    def position_nodes(nodes, y_level):
        n = len(nodes)
        if n == 0:
            return
        sorted_nodes = sorted(nodes, key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))
        for i, node in enumerate(sorted_nodes):
            x = (i + 0.5) * width / n - width/2
            pos[node] = (x, y_level * height / num_levels)
    
    position_nodes(npus, y_levels['npu'])
    position_nodes(nvswitches, y_levels['nvswitch'])
    position_nodes(nics, y_levels['nic'])
    
    if is_folded_clos:
        position_nodes(edge_switches, y_levels['edge_switch'])
        position_nodes(agg_switches, y_levels['agg_switch'])
        position_nodes(core_switches, y_levels['core_switch'])
    else:
        position_nodes(main_switches, y_levels['main_switch'])
    
    return pos

def build_topology(topology):
    """
    Build topology by calling the appropriate methods based on topology type.
    Different topology classes have different build methods.
    """
    # For Dragonfly and Jellyfish: need to call DesignFullTopology then LinksToG2ConfFile
    # For FoldedClos: just DesignFullTopology (it builds links internally)
    
    if hasattr(topology, 'DesignFullTopology'):
        topology.DesignFullTopology()
    
    # Dragonfly and Jellyfish need LinksToG2ConfFile to add host connections
    # LinksToG2ConfFile returns the full links dict but doesn't update self.links
    if hasattr(topology, 'LinksToG2ConfFile') and topology.name in ('dragonfly', 'jellyfish'):
        links = topology.LinksToG2ConfFile()
        topology.links = links  # Update self.links with the returned value


def visualize_topology(topology, title="Network Topology", figsize=(14, 10), show_bw=True):
    """
    Visualize a network topology with nodes, edges, names, and bandwidth labels.
    Uses hierarchical layout with intra-node elements at the bottom.
    """
    # Build the topology
    build_topology(topology)
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes and edges from topology
    links = topology.links
    link_bws = topology.link_bandwidths
    
    for link_id, (src, dst) in links.items():
        G.add_edge(src, dst, bandwidth=link_bws.get(link_id, 1.0))
    
    # Create hierarchical layout
    pos = create_hierarchical_layout(G, topology, width=12, height=10)
    
    # Colors for different node types
    color_map = {
        'npu': '#4CAF50',         # Green
        'nvswitch': '#FF9800',    # Orange
        'npu_switch': '#FF9800',  # Orange (same as NVSwitch)
        'nic': '#E91E63',         # Pink (NIC connecting to main switch)
        'edge_switch': '#2196F3', # Blue (Edge/ToR - FoldedClos)
        'agg_switch': '#9C27B0',  # Purple (Aggregation - FoldedClos)
        'core_switch': '#F44336', # Red (Core - FoldedClos)
        'main_switch': '#9C27B0'  # Purple (Dragonfly/Jellyfish switches)
    }
    
    node_colors = [color_map.get(classify_node(n, topology), '#757575') for n in G.nodes()]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, ax=ax)
    
    # Draw edges with bandwidth as edge width (normalized)
    edges = G.edges()
    if edges:
        bws = [G[u][v]['bandwidth'] for u, v in edges]
        max_bw = max(bws) if bws else 1
        edge_widths = [1 + 3 * (bw / max_bw) for bw in bws]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               width=edge_widths, alpha=0.7, ax=ax,
                               connectionstyle="arc3,rad=0.1")
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
    
    # Draw edge bandwidth labels if requested
    if show_bw:
        edge_labels = {(u, v): f"{G[u][v]['bandwidth']:.1f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax,
                                      label_pos=0.3)
    
    # Build legend based on which node types are present
    is_folded_clos = any(classify_node(n, topology) in ('edge_switch', 'agg_switch', 'core_switch') for n in G.nodes())
    
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='NPU'),
        mpatches.Patch(color='#FF9800', label='NVSwitch/NPU-Switch'),
        mpatches.Patch(color='#E91E63', label='NIC Switch'),
    ]
    if is_folded_clos:
        legend_elements.extend([
            mpatches.Patch(color='#2196F3', label='Edge Switch (ToR)'),
            mpatches.Patch(color='#9C27B0', label='Aggregation Switch'),
            mpatches.Patch(color='#F44336', label='Core Switch'),
        ])
    else:
        legend_elements.append(
            mpatches.Patch(color='#9C27B0', label='Switch (ToR)')
        )
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax, G

def visualize_single_node(topology, node_idx=1, figsize=(10, 8)):
    """
    Visualize just one node's internal structure.
    Shows NPUs -> NVSwitch(es) -> Main Switch (ToR) hierarchy.
    node_idx is 1-based.
    """
    # Build the topology
    build_topology(topology)
    
    # Get NPUs for this node
    if not hasattr(topology, 'node_npus') or node_idx not in topology.node_npus:
        print(f"Node {node_idx} not found. Available nodes: {list(topology.node_npus.keys())[:5]}...")
        return
    
    npus = set(topology.node_npus[node_idx])
    
    # Get the main switch for this node (which IS the ToR)
    main_sw = topology.node_to_switch.get(node_idx) if hasattr(topology, 'node_to_switch') else None
    
    # Get NVSwitches connected to these NPUs
    nvswitches = set()
    for npu in npus:
        if npu in topology.npu_to_intra_switches:
            nvswitches.update(topology.npu_to_intra_switches[npu])
    
    # Include main switch in relevant nodes
    relevant_nodes = npus | nvswitches | ({main_sw} if main_sw else set())
    
    G = nx.DiGraph()
    for link_id, (src, dst) in topology.links.items():
        if src in relevant_nodes and dst in relevant_nodes:
            bw = topology.link_bandwidths.get(link_id, 1.0)
            G.add_edge(src, dst, bandwidth=bw)
    
    # Manual hierarchical layout
    pos = {}
    
    # NPUs at bottom
    npu_list = sorted(npus)
    for i, npu in enumerate(npu_list):
        pos[npu] = ((i + 0.5) * 2 - len(npu_list), 0)
    
    # NVSwitches in middle
    nv_list = sorted(nvswitches)
    for i, nv in enumerate(nv_list):
        pos[nv] = ((i + 0.5) * 2 - len(nv_list), 1.5)
    
    # Main switch (ToR) at top
    if main_sw:
        pos[main_sw] = (0, 3)
    
    # Colors - use classify_node for proper FoldedClos coloring
    color_map = {
        'npu': '#4CAF50',
        'nvswitch': '#FF9800',
        'npu_switch': '#FF9800',
        'nic': '#E91E63',
        'edge_switch': '#2196F3',
        'agg_switch': '#9C27B0',
        'core_switch': '#F44336',
        'main_switch': '#9C27B0'
    }
    node_colors = [color_map.get(classify_node(n, topology), '#757575') for n in G.nodes()]
    
    # Determine switch label
    sw_type = classify_node(main_sw, topology) if main_sw else 'main_switch'
    sw_label = 'Edge Switch (ToR)' if sw_type == 'edge_switch' else 'Switch (ToR)'
    sw_color = color_map.get(sw_type, '#9C27B0')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Draw edges
    edges = list(G.edges())
    if edges:
        bws = [G[u][v]['bandwidth'] for u, v in edges]
        max_bw = max(bws) if bws else 1
        edge_widths = [1 + 4 * (bw / max_bw) for bw in bws]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               width=edge_widths, alpha=0.7, ax=ax,
                               connectionstyle="arc3,rad=0.05")
    
    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['bandwidth']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10, ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='NPU'),
        mpatches.Patch(color='#FF9800', label='NVSwitch'),
        mpatches.Patch(color=sw_color, label=sw_label)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_title(f"Single Node View (Node {node_idx})\nNPUs -> NVSwitch -> {sw_label}", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax

def visualize_single_server(topology, server_idx=1, figsize=(14, 10)):
    """
    Visualize a single server's structure showing all nodes and NPUs.
    Shows the hierarchy: NPUs -> NVSwitches -> Main Switch (ToR)
    server_idx is 1-based.
    """
    # Build the topology
    build_topology(topology)
    
    # Check if server exists
    if not hasattr(topology, 'server_nodes') or server_idx not in topology.server_nodes:
        print(f"Server {server_idx} not found. Available servers: {list(topology.server_nodes.keys())[:5]}...")
        return
    
    # Get all nodes in this server
    node_ids = topology.server_nodes[server_idx]
    
    # Get the main switch for this server (which IS the ToR)
    main_sw = topology.server_to_switch.get(server_idx) if hasattr(topology, 'server_to_switch') else None
    
    # Collect all NPUs and NVSwitches for this server
    all_npus = set()
    all_nvswitches = set()
    for node_id in node_ids:
        if node_id in topology.node_npus:
            npus = topology.node_npus[node_id]
            all_npus.update(npus)
            for npu in npus:
                if npu in topology.npu_to_intra_switches:
                    all_nvswitches.update(topology.npu_to_intra_switches[npu])
    
    # All relevant nodes
    relevant_nodes = all_npus | all_nvswitches | ({main_sw} if main_sw else set())
    
    G = nx.DiGraph()
    for link_id, (src, dst) in topology.links.items():
        if src in relevant_nodes and dst in relevant_nodes:
            bw = topology.link_bandwidths.get(link_id, 1.0)
            G.add_edge(src, dst, bandwidth=bw)
    
    # Create hierarchical layout by node
    pos = {}
    num_nodes = len(node_ids)
    node_width = 12 / num_nodes if num_nodes > 0 else 12
    
    for node_offset, node_id in enumerate(node_ids):
        node_npus = topology.node_npus.get(node_id, [])
        node_center_x = (node_offset + 0.5) * node_width - 6
        
        # Position NPUs for this node
        for i, npu in enumerate(sorted(node_npus)):
            npu_x = node_center_x + (i - len(node_npus)/2 + 0.5) * 0.8
            pos[npu] = (npu_x, 0)
        
        # Position NVSwitches for this node
        node_nvswitches = set()
        for npu in node_npus:
            if npu in topology.npu_to_intra_switches:
                node_nvswitches.update(topology.npu_to_intra_switches[npu])
        
        for i, nvs in enumerate(sorted(node_nvswitches)):
            nvs_x = node_center_x + (i - len(node_nvswitches)/2 + 0.5) * 1.0
            pos[nvs] = (nvs_x, 1.5)
    
    # Main switch (ToR) at the top center
    if main_sw:
        pos[main_sw] = (0, 3)
    
    # Colors - use classify_node for proper FoldedClos coloring
    color_map = {
        'npu': '#4CAF50',
        'nvswitch': '#FF9800',
        'npu_switch': '#FF9800',
        'nic': '#E91E63',
        'edge_switch': '#2196F3',
        'agg_switch': '#9C27B0',
        'core_switch': '#F44336',
        'main_switch': '#9C27B0'
    }
    node_colors = [color_map.get(classify_node(n, topology), '#757575') for n in G.nodes()]
    
    # Determine switch label
    sw_type = classify_node(main_sw, topology) if main_sw else 'main_switch'
    sw_label = 'Edge Switch (ToR)' if sw_type == 'edge_switch' else 'Switch (ToR)'
    sw_color = color_map.get(sw_type, '#9C27B0')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1200, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
    
    # Draw edges
    edges = list(G.edges())
    if edges:
        bws = [G[u][v]['bandwidth'] for u, v in edges]
        max_bw = max(bws) if bws else 1
        edge_widths = [1 + 3 * (bw / max_bw) for bw in bws]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                               width=edge_widths, alpha=0.7, ax=ax,
                               connectionstyle="arc3,rad=0.05")
    
    # Edge labels
    edge_labels = {(u, v): f"{G[u][v]['bandwidth']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=9, ax=ax)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#4CAF50', label='NPU'),
        mpatches.Patch(color='#FF9800', label='NVSwitch'),
        mpatches.Patch(color=sw_color, label=sw_label)
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add node boundaries
    for node_offset, node_id in enumerate(node_ids):
        node_npus = topology.node_npus.get(node_id, [])
        if node_npus:
            node_center_x = (node_offset + 0.5) * node_width - 6
            rect = plt.Rectangle((node_center_x - node_width/2 + 0.3, -0.5), 
                                  node_width - 0.6, 2.5, 
                                  fill=False, linestyle='--', 
                                  edgecolor='gray', linewidth=1.5)
            ax.add_patch(rect)
            ax.text(node_center_x, -0.8, f"Node {node_id}", 
                    ha='center', va='top', fontsize=10, style='italic')
    
    ax.set_title(f"Server {server_idx}: {num_nodes} Nodes, {len(all_npus)} NPUs\n"
                 f"Hierarchy: NPUs -> NVSwitch -> {sw_label}", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_xlim(-7, 7)
    ax.set_ylim(-1.5, 4)
    plt.tight_layout()
    
    return fig, ax