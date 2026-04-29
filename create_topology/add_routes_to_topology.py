#!/usr/bin/env python3
"""
Add uniform (Al-Fares) routes to an existing FoldedClos topology file.

Uses the local RoutingEngine directly, so the generated routes are guaranteed
to be identical to what the folded-Clos engine computes at runtime.

Usage
-----
    python add_routes_to_topology.py <topology_file> [--output <output_file>]

The topology file must be in NS3 format (header + switch IDs + links + ROUTES).
A matching *_nodemap.json sidecar must exist next to the topology file.

If --output is not given, the output file name is the topology file name with
'_with_routes' appended (before any extension).

Example
-------
    python add_routes_to_topology.py topologies/FoldedClos_64npus_switch_no_route

    Reads:   FoldedClos_64npus_switch_no_route
             FoldedClos_64npus_switch_no_route_nodemap.json
    Writes:  FoldedClos_64npus_switch_no_route_with_routes
"""

import argparse
import os
import sys
import time

import networkx as nx
from routing_engine import RoutingEngine


# =========================================================================
# Topology file parsing
# =========================================================================

def parse_topology_file(filepath):
    """
    Parse an NS3-format topology file.

    Returns
    -------
    header_lines : list[str]
        Every line from the file up to and including "ROUTES".
    num_npus : int
        Number of NPU (host) nodes.
    graph : nx.Graph
        Undirected NetworkX graph with string node IDs.
    """
    with open(filepath, 'r') as f:
        raw_lines = f.readlines()

    # ── Find the ROUTES marker ──────────────────────────────────────────
    routes_idx = None
    for i, line in enumerate(raw_lines):
        if line.strip() == 'ROUTES':
            routes_idx = i
            break

    if routes_idx is None:
        print("ERROR: No 'ROUTES' marker found in the topology file.")
        sys.exit(1)

    header_lines = raw_lines[:routes_idx + 1]  # include the ROUTES line

    # ── Parse the header ─────────────────────────────────────────────────
    first_line = raw_lines[0].strip().split()
    total_nodes = int(first_line[0])
    num_switches = int(first_line[1])
    num_npus = total_nodes - num_switches

    # ── Build the graph from links ───────────────────────────────────────
    graph = nx.Graph()
    # Skip header (1 line) + switch IDs (num_switches lines)
    link_start = 1 + num_switches
    for i in range(link_start, routes_idx):
        line = raw_lines[i].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            graph.add_edge(parts[0], parts[1])

    return header_lines, num_npus, graph


# =========================================================================
# Route generation
# =========================================================================

def generate_routes(graph, num_npus, topology_file):
    """
    Use the g2 RoutingEngine (foldedclos_uniform mode) to compute a
    deterministic route for every host-pair.

    Parameters
    ----------
    graph : nx.Graph
        Topology graph with string node IDs.
    num_npus : int
        Number of NPU/host nodes (IDs 0 .. num_npus-1).
    topology_file : str
        Path to the topology file (used to locate the nodemap sidecar).

    Returns
    -------
    list[str]
        Route lines in NS3 format:  "src:dst:[n1, n2, ..., nN]"
    """
    # The routing engine lives beside this script and only supports the
    # folded-Clos route generation flow that add_routes_to_topology needs.
    engine = RoutingEngine(graph=graph, topology_file=topology_file)

    if not engine._fc_ready:
        print("ERROR: RoutingEngine failed to build the FoldedClos routing map.")
        print("       Make sure the _nodemap.json sidecar exists next to the "
              "topology file.")
        sys.exit(1)

    # ── Compute routes for all NPU pairs ─────────────────────────────────
    route_lines = []
    total_pairs = num_npus * (num_npus - 1)
    computed = 0

    t0 = time.perf_counter()
    for src in range(num_npus):
        for dst in range(num_npus):
            if src == dst:
                continue
            path = engine.calculate_route(src, dst)
            path_str = ', '.join(str(n) for n in path)
            route_lines.append(f"{src}:{dst}:[{path_str}]\n")
            computed += 1

        # Progress update every 10% of sources
        if (src + 1) % max(1, num_npus // 10) == 0:
            elapsed = time.perf_counter() - t0
            print(f"  {computed:,}/{total_pairs:,} pairs "
                  f"({100 * computed / total_pairs:.0f}%) "
                  f"in {elapsed:.1f}s")

    elapsed = time.perf_counter() - t0
    print(f"  Computed {computed:,} routes in {elapsed:.1f}s")
    return route_lines


# =========================================================================
# File writing
# =========================================================================

def write_output(header_lines, route_lines, output_path):
    """Write the topology header + computed routes to *output_path*."""
    with open(output_path, 'w') as f:
        f.writelines(header_lines)
        f.writelines(route_lines)
    print(f"Wrote {output_path}  "
          f"({len(route_lines):,} routes)")


# =========================================================================
# CLI
# =========================================================================

def default_output_path(topology_file):
    """Derive output filename by appending '_with_routes' before extension."""
    base, ext = os.path.splitext(topology_file)
    return f"{base}_with_routes{ext}"


def main():
    parser = argparse.ArgumentParser(
        description='Add Al-Fares uniform routes to a FoldedClos topology file '
                    'using the g2 RoutingEngine.')
    parser.add_argument('topology_file',
                        help='Path to the NS3-format topology file.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file path (default: <input>_with_routes).')
    args = parser.parse_args()

    topo_path = args.topology_file
    if not os.path.isfile(topo_path):
        print(f"ERROR: topology file not found: {topo_path}")
        sys.exit(1)

    nodemap_path = topo_path + '_nodemap.json'
    if not os.path.isfile(nodemap_path):
        print(f"ERROR: nodemap sidecar not found: {nodemap_path}")
        sys.exit(1)

    output_path = args.output or default_output_path(topo_path)

    print(f"Topology : {topo_path}")
    print(f"Nodemap  : {nodemap_path}")
    print(f"Output   : {output_path}")
    print()

    # 1. Parse topology
    header_lines, num_npus, graph = parse_topology_file(topo_path)
    print(f"Parsed topology: {graph.number_of_nodes()} nodes, "
          f"{graph.number_of_edges()} edges, {num_npus} NPUs")

    # 2. Generate routes
    print("Computing routes ...")
    route_lines = generate_routes(graph, num_npus, topo_path)

    # 3. Write output
    write_output(header_lines, route_lines, output_path)


if __name__ == '__main__':
    main()
