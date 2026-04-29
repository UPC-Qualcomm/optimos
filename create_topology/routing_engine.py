"""
RoutingEngine 
'uniform'
    Single deterministic path via the Al-Fares fat-tree routing algorithm
    (http://ccr.sigcomm.org/online/files/p63-alfares.pdf), extended to support
    multi-NPU servers connected via NVSwitch / ring / fully-connected
    intra-node topologies.  Requires a *_nodemap.json sidecar file alongside
    the topology file.

"""

import json
import time
from collections import deque

import networkx as nx


class RoutingEngine:
    def __init__(self, graph, topology_file=None):
        self.graph = graph
        self.topology_file = topology_file
        self._fc_ready = False

        if topology_file is not None:
            self.build_foldedclos_routing_map(topology_file)

    def _load_nodemap(self, topology_file):
        nodemap_path = str(topology_file) + '_nodemap.json'
        try:
            with open(nodemap_path, 'r') as f:
                nodemap = json.load(f)
            print(f'[RoutingEngine] Loaded nodemap from {nodemap_path} '
                  f'({len(nodemap)} nodes)')
            return nodemap
        except (OSError, json.JSONDecodeError) as exc:
            print(f'[RoutingEngine] Warning: could not load nodemap '
                  f'({nodemap_path}): {exc}')
            return None

    def build_foldedclos_routing_map(self, topology_file):
        start = time.perf_counter()
        self._fc_ready = False

        nodemap = self._load_nodemap(topology_file)
        self._id_prefix = {}
        if not nodemap:
            print('[RoutingEngine] ERROR: nodemap required for foldedclos_uniform '
                  'but not found.')
            print('[RoutingEngine]        Generate it with create_topology.py '
                  '(it creates *_nodemap.json).')
            return

        for node_id, prefixed_name in nodemap.items():
            self._id_prefix[node_id] = prefixed_name[0] if prefixed_name else ''

        all_nodes = list(self.graph.nodes())
        edge_switches = sorted(
            [node for node in all_nodes if self._id_prefix.get(node) == 't'],
            key=lambda node: int(node),
        )
        agg_switches = sorted(
            [node for node in all_nodes if self._id_prefix.get(node) == 'a'],
            key=lambda node: int(node),
        )
        core_switches = sorted(
            [node for node in all_nodes if self._id_prefix.get(node) == 'c'],
            key=lambda node: int(node),
        )
        npu_nodes = sorted(
            [node for node in all_nodes if self._id_prefix.get(node) == 'h'],
            key=lambda node: int(node),
        )

        print(f'[RoutingEngine] Node counts – NPUs: {len(npu_nodes)}, '
              f'edge: {len(edge_switches)}, agg: {len(agg_switches)}, '
              f'core: {len(core_switches)}')

        if not edge_switches:
            print("[RoutingEngine] ERROR: no edge switches ('t' prefix) found.")
            return

        sample_edge = edge_switches[0]
        agg_of_sample = [node for node in self.graph.neighbors(sample_edge)
                         if self._id_prefix.get(node) == 'a']
        k_half = len(agg_of_sample)
        if k_half == 0:
            print('[RoutingEngine] ERROR: edge switch has no agg-switch neighbours.')
            return

        self._fc_K_half = k_half
        self._fc_K = k_half * 2
        print(f'[RoutingEngine] Detected K={self._fc_K} (K/2={self._fc_K_half})')

        edge_to_agg_frozenset = {
            edge: frozenset(node for node in self.graph.neighbors(edge)
                            if self._id_prefix.get(node) == 'a')
            for edge in edge_switches
        }

        agg_set_to_pod = {}
        pod_edge_lists = {}
        for edge in edge_switches:
            agg_set = edge_to_agg_frozenset[edge]
            if agg_set not in agg_set_to_pod:
                pod_idx = len(agg_set_to_pod)
                agg_set_to_pod[agg_set] = pod_idx
                pod_edge_lists[pod_idx] = []
            pod_edge_lists[agg_set_to_pod[agg_set]].append(edge)

        for pod_idx in pod_edge_lists:
            pod_edge_lists[pod_idx].sort(key=lambda node: int(node))

        self._edge_to_pod_pos = {}
        for pod_idx, edge_list in pod_edge_lists.items():
            for pos, edge in enumerate(edge_list):
                self._edge_to_pod_pos[edge] = (pod_idx, pos)

        self._pod_agg_switches = {
            pod_idx: sorted(agg_set, key=lambda node: int(node))
            for agg_set, pod_idx in agg_set_to_pod.items()
        }

        self._agg_to_core = {
            agg: sorted(
                [node for node in self.graph.neighbors(agg)
                 if self._id_prefix.get(node) == 'c'],
                key=lambda node: int(node),
            )
            for agg in agg_switches
        }

        self._npu_to_edge_sw = {}
        self._npu_to_intra_path = {}
        intra_ok_prefixes = {'h', 't', 'v', 'n', ''}

        for npu in npu_nodes:
            visited = {npu}
            queue = deque([(npu, [npu])])
            found = False
            while queue and not found:
                node, path = queue.popleft()
                for neighbor in self.graph.neighbors(node):
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    prefix = self._id_prefix.get(neighbor, '')
                    if prefix == 't':
                        self._npu_to_edge_sw[npu] = neighbor
                        self._npu_to_intra_path[npu] = new_path
                        found = True
                        break
                    if prefix in intra_ok_prefixes and prefix not in ('a', 'c'):
                        queue.append((neighbor, new_path))

        print(f'[RoutingEngine] NPU→edge mapping: '
              f'{len(self._npu_to_edge_sw)}/{len(npu_nodes)} NPUs mapped')

        intra_node_set = frozenset(
            node for node in all_nodes if self._id_prefix.get(node, '') not in ('a', 'c')
        )
        self._intra_subgraph = self.graph.subgraph(intra_node_set)

        self._npu_to_intra_paths = {}
        for npu, edge_sw in self._npu_to_edge_sw.items():
            try:
                self._npu_to_intra_paths[npu] = list(
                    nx.all_shortest_paths(self._intra_subgraph, npu, edge_sw)
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self._npu_to_intra_paths[npu] = [self._npu_to_intra_path[npu]]

        multi_path_np_us = sum(1 for paths in self._npu_to_intra_paths.values()
                               if len(paths) > 1)
        if multi_path_np_us:
            print(f'[RoutingEngine] {multi_path_np_us}/{len(self._npu_to_intra_paths)} '
                  'NPUs have >1 intra-node path – NVSwitch uplink load-balancing active')
        else:
            print('[RoutingEngine] Single intra-node path per NPU '
                  '(no NVSwitch uplink diversity detected)')

        self._npu_server_idx = {}
        self._npu_idx_in_server = {}
        for edge in edge_switches:
            npus_here = sorted(
                [npu for npu, sw in self._npu_to_edge_sw.items() if sw == edge],
                key=lambda node: int(node),
            )
            if not npus_here:
                continue
            npus_per_server = max(1, len(npus_here) // k_half)
            for rank, npu in enumerate(npus_here):
                self._npu_server_idx[npu] = rank // npus_per_server
                self._npu_idx_in_server[npu] = rank % npus_per_server

        self._fc_ready = True
        elapsed = time.perf_counter() - start
        print(f'[RoutingEngine] FoldedClos routing map built in {elapsed * 1000:.1f} ms '
              f'({len(pod_edge_lists)} pods, K={self._fc_K})')

    def _route_single_shortest(self, src_key, dest_key):
        try:
            return list(nx.shortest_path(self.graph, src_key, dest_key))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise ValueError(f'No path found between {src_key} and {dest_key}')

    def _route_foldedclos_uniform(self, src_key, dest_key):
        if not self._fc_ready:
            return self._route_single_shortest(src_key, dest_key)

        src_edge = self._npu_to_edge_sw.get(src_key)
        dst_edge = self._npu_to_edge_sw.get(dest_key)
        if src_edge is None or dst_edge is None:
            return self._route_single_shortest(src_key, dest_key)

        src_srv_in_edge = self._npu_server_idx.get(src_key, 0)
        dst_srv_in_edge = self._npu_server_idx.get(dest_key, 0)

        src_intra_all = self._npu_to_intra_paths.get(src_key, [[src_key, src_edge]])
        dst_intra_all = self._npu_to_intra_paths.get(dest_key, [[dest_key, dst_edge]])

        lo_id = min(int(src_key), int(dest_key))
        hi_id = max(int(src_key), int(dest_key))
        intra_hash = (lo_id * 2654435761 ^ hi_id * 2246822519) & 0xFFFFFFFF
        src_intra = src_intra_all[intra_hash % len(src_intra_all)]
        dst_intra = dst_intra_all[intra_hash % len(dst_intra_all)]
        dst_intra_rev = list(reversed(dst_intra))

        if src_edge == dst_edge:
            if src_srv_in_edge == dst_srv_in_edge:
                try:
                    all_intra = list(nx.all_shortest_paths(
                        self._intra_subgraph, src_key, dest_key))
                    return all_intra[intra_hash % len(all_intra)]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return list(nx.shortest_path(self.graph, src_key, dest_key))
            return src_intra + dst_intra_rev[1:]

        k_half = self._fc_K_half
        src_pod, src_edge_pos = self._edge_to_pod_pos.get(src_edge, (0, 0))
        dst_pod, _ = self._edge_to_pod_pos.get(dst_edge, (0, 0))
        e_out_port = (dst_srv_in_edge + src_edge_pos) % k_half

        if src_pod == dst_pod:
            agg_list = self._pod_agg_switches.get(src_pod, [])
            if not agg_list or e_out_port >= len(agg_list):
                return list(nx.shortest_path(self.graph, src_key, dest_key))
            agg_sw = agg_list[e_out_port]
            return src_intra + [agg_sw, dst_edge] + dst_intra_rev[1:]

        a_out_port = (dst_srv_in_edge + e_out_port) % k_half
        src_agg_list = self._pod_agg_switches.get(src_pod, [])
        dst_agg_list = self._pod_agg_switches.get(dst_pod, [])
        if (not src_agg_list or e_out_port >= len(src_agg_list) or
                not dst_agg_list or e_out_port >= len(dst_agg_list)):
            return list(nx.shortest_path(self.graph, src_key, dest_key))

        agg_src = src_agg_list[e_out_port]
        agg_dst = dst_agg_list[e_out_port]
        core_list = self._agg_to_core.get(agg_src, [])
        if not core_list or a_out_port >= len(core_list):
            return list(nx.shortest_path(self.graph, src_key, dest_key))

        core_sw = core_list[a_out_port]
        return src_intra + [agg_src, core_sw, agg_dst, dst_edge] + dst_intra_rev[1:]

    def calculate_route(self, src, dest):
        return self._route_foldedclos_uniform(str(src), str(dest))
