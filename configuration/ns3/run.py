import argparse
import sys
import os
import itertools

# --- Determine Project Root from Environment Variable ---
try:
    project_root = os.environ["OPTIMOS_ROOT"]
    print(project_root)
    if not os.path.isdir(project_root):
        print(f"Error: OPTIMOS_ROOT environment variable '{project_root}' is not a valid directory.")
        sys.exit(1)
except KeyError:
    print("Error: Please set the 'OPTIMOS_ROOT' environment variable to the optimos project root directory.")
    sys.exit(1)


config_template="""ENABLE_QCN {enable_qcn}
USE_DYNAMIC_PFC_THRESHOLD 1
ECMP_SEED 25
USE_PRECOMPUTED_ROUTES 0

PACKET_PAYLOAD_SIZE {packet_payload}

# 16-node ring topology
TOPOLOGY_FILE {project_root}/configuration/ns3/FoldedClos_16_topology.txt
FLOW_FILE {project_root}/configuration/ns3/flow.txt
TRACE_FILE {project_root}/configuration/ns3/trace16.txt
TRACE_OUTPUT_FILE {project_root}/configuration/ns3/output/astrasim_16nodes_ring_{cc}.tr
FCT_OUTPUT_FILE {project_root}/configuration/ns3/output/astrasim_16nodes_ring_{cc}_fct.txt
PFC_OUTPUT_FILE {project_root}/configuration/ns3/output/astrasim_16nodes_ring_{cc}_pfc.txt
QLEN_MON_FILE {project_root}/configuration/ns3/output/astrasim_16nodes_ring_{cc}_qlen.txt
QLEN_MON_START 0
QLEN_MON_END 300000000000
QLEN_MON_INTERVAL 1000000000

# Extended simulation time for larger workloads
SIMULATOR_STOP_TIME 0.1

# Congestion control settings
CC_MODE {mode}
ALPHA_RESUME_INTERVAL {t_alpha}
RATE_DECREASE_INTERVAL {t_dec}
CLAMP_TARGET_RATE 0
RP_TIMER {t_inc}
EWMA_GAIN {g}
FAST_RECOVERY_TIMES 1
RATE_AI {ai}Mb/s
RATE_HAI {hai}Mb/s
MIN_RATE {min_rate}
DCTCP_RATE_AI {dctcp_ai}Mb/s

ERROR_RATE_PER_LINK 0.0000
L2_CHUNK_SIZE 4000
L2_ACK_INTERVAL 1
L2_BACK_TO_ZERO 0

HAS_WIN {has_win}
GLOBAL_T 1
VAR_WIN {vwin}
FAST_REACT {us}
U_TARGET {u_tgt}
MI_THRESH {mi}
INT_MULTI {int_multi}
MULTI_RATE 0
SAMPLE_FEEDBACK 0
PINT_LOG_BASE {pint_log_base}
PINT_PROB {pint_prob}
NIC_TOTAL_PAUSE_TIME 0

RATE_BOUND 1
ACK_HIGH_PRIO {ack_prio}
LINK_DOWN {link_down}
ENABLE_TRACE {enable_tr}

# Buffer thresholds
KMAX_MAP {kmax_map}
KMIN_MAP {kmin_map}
PMAX_MAP {pmax_map}
# Ring topology buffer 
BUFFER_SIZE {buffer_size}
"""
if __name__ == "__main__":
    bw = 200
    
    # Buffer sizes to test
    buffer_sizes = [16 * bw // 50]
    
    # Packet payload sizes to test
    packet_payload_sizes = [9000]

    # Parameter space
    # Base CC methods
    hp_params = list(itertools.product(['hp'], [0.95], [0], [(400, 100)])) # cc, utgt
    hpccPint_params = list(itertools.product(['hpccPint'], [0.95], [0], [(400, 100)]))
    dctcp_params = [('dctcp', 0.95, 0, (400, 100))]
    # PFC / original RDMA (no QCN / DCQCN / HPCC / TIMELY / DCTCP / PINT)
    # pfc_params = [('pfc', 0.95, 0, (400, 100))]
    
    # DCQCN variations
    dcqcn_variants = ['dcqcn', 'dcqcn_paper']
    dcqcn_params = list(itertools.product(dcqcn_variants, [0.95], [0], [(400, 100)]))

    params = hp_params + hpccPint_params + dctcp_params + dcqcn_params
    
    # Generate configs
    config_idx = 1
    for (cc, u_tgt_float, mi, k_base) in params:
        (kmax_base, kmin_base) = k_base
        for bfsz in buffer_sizes:
            for packet_payload in packet_payload_sizes:
                utgt_int = int(u_tgt_float * 100)
            
                # General settings
                min_rate = "1Mb/s" # Min rate as string
                
                failure = ''
                link_down = '0 0 0'
                enable_tr = 1
                pint_log_base = 1.05
                pint_prob = 1.0
                enable_qcn = 1

                config_name = f"{project_root}/configuration/ns3/configs/FoldedClos_16_config{config_idx}.txt"
            
                # CC specific settings
                if cc == "pfc":
                    # Use original RDMA behaviour + link-layer PFC: disable QCN and PINT
                    mode = 0
                    enable_qcn = 0
                    ai = 0
                    hai = 0
                    g = 0
                    dctcp_ai = 1000
                    int_multi = 1
                    has_win, vwin, us, ack_prio = 0, 0, 0, 0
                    t_alpha, t_dec, t_inc = 1, 4, 300
                    pint_prob = 0.0
                    cc_name = f"pfc_buf{bfsz}_pkt{packet_payload}"
                elif cc.startswith("dcqcn"):
                    mode = 1
                    ai = 5 * bw / 25
                    hai = 50 * bw / 25
                    g = 0.00390625
                    dctcp_ai = 1000
                    int_multi = 1
                    if cc == "dcqcn":
                        t_alpha, t_dec, t_inc, has_win, vwin, ack_prio = 1, 4, 300, 0, 0, 1
                    elif cc == "dcqcn_paper":
                        t_alpha, t_dec, t_inc, has_win, vwin, ack_prio = 50, 50, 55, 0, 0, 1
                    elif cc == "dcqcn_vwin":
                        t_alpha, t_dec, t_inc, has_win, vwin, ack_prio = 1, 4, 300, 1, 1, 0
                    elif cc == "dcqcn_paper_vwin":
                        t_alpha, t_dec, t_inc, has_win, vwin, ack_prio = 50, 50, 55, 1, 1, 0
                    us = 0
                    cc_name = f"{cc}_buf{bfsz}_pkt{packet_payload}"
                elif cc == "hp":
                    mode = 3
                    ai = 10 * bw / 25
                    hai = ai
                    int_multi = int(bw / 25)
                    if int_multi < 1: int_multi = 1
                    has_win, vwin, us, ack_prio = 1, 1, 1, 0
                    t_alpha, t_dec, t_inc, g, dctcp_ai = 1, 4, 300, 0.00390625, 1000
                    cc_name = "%s%dmi%d_buf%d_pkt%d" % (cc, utgt_int, mi, bfsz, packet_payload)
                elif cc == "dctcp":
                    mode = 8
                    ai = 10
                    hai = ai
                    dctcp_ai = 615 * bw / 25 # Scale with bw
                    has_win, vwin, us, ack_prio = 1, 1, 0, 0
                    t_alpha, t_dec, t_inc, g = 1, 4, 300, 0.0625
                    int_multi = 1
                    cc_name = f"{cc}_buf{bfsz}_pkt{packet_payload}"
                elif cc.startswith("timely"):
                    mode = 7
                    ai = 10 * bw / 10
                    hai = 50 * bw / 10
                    t_alpha, t_dec, t_inc, g, dctcp_ai = 1, 4, 300, 0.00390625, 1000
                    int_multi = 1
                    us = 0
                    if cc == "timely":
                        has_win, vwin, ack_prio = 0, 0, 1
                    elif cc == "timely_vwin":
                        has_win, vwin, ack_prio = 1, 1, 1
                    cc_name = f"{cc}_buf{bfsz}_pkt{packet_payload}"
                elif cc == "hpccPint":
                    mode = 10
                    ai = 10 * bw / 25
                    hai = ai
                    int_multi = int(bw / 25)
                    if int_multi < 1: int_multi = 1
                    has_win, vwin, us, ack_prio = 1, 1, 1, 0
                    t_alpha, t_dec, t_inc, g, dctcp_ai = 1, 4, 300, 0.00390625, 1000
                    cc_name = "%s%dmi%d_buf%d_pkt%d" % (cc, utgt_int, mi, bfsz, packet_payload)
                else:
                    continue

                # Buffer settings
                kmax_val = int(kmax_base * bw / 25) if int(400 * bw / 25) > 0 else 1
                kmin_val = int(kmin_base * bw / 25) if int(100 * bw / 25) > 0 else 1
                kmax_map = "2 %d %d %d %d" % (bw * 1000000000, kmax_val, bw * 4 * 1000000000, kmax_val * 4)
                kmin_map = "2 %d %d %d %d" % (bw * 1000000000, kmin_val, bw * 4 * 1000000000, kmin_val * 4)
                pmax_map = "2 %d %.2f %d %.2f" % (bw * 1000000000, 0.2, bw * 4 * 1000000000, 0.2)
                if cc == "dctcp":
                    kmax_val_dctcp = int(30 * bw / 10) if int(30 * bw / 10) > 0 else 1
                    kmax_map = "2 %d %d %d %d" % (bw * 1000000000, kmax_val_dctcp, bw * 4 * 1000000000, kmax_val_dctcp * 4)
                    kmin_map = kmax_map
                    pmax_map = "2 %d %.2f %d %.2f" % (bw * 1000000000, 1.0, bw * 4 * 1000000000, 1.0)

                config = config_template.format(
                project_root=project_root,
                cc=cc_name, mode=mode, t_alpha=t_alpha, t_dec=t_dec, t_inc=t_inc, g=g,
                ai=ai, hai=hai, dctcp_ai=dctcp_ai, has_win=has_win, vwin=vwin, us=us,
                u_tgt=u_tgt_float, mi=mi, int_multi=int_multi, pint_log_base=pint_log_base,
                pint_prob=pint_prob, ack_prio=ack_prio, link_down=link_down, failure=failure,
                kmax_map=kmax_map, kmin_map=kmin_map, pmax_map=pmax_map, buffer_size=bfsz,
                enable_tr=enable_tr, min_rate=min_rate, enable_qcn=enable_qcn,
                packet_payload=packet_payload
            )

                with open(config_name, "w") as file:
                    file.write(config.format(project_root=project_root))
                
                print("Generated %s" % config_name)
                config_idx += 1

    print(f"\nGenerated {config_idx - 1} configuration files.")
    print(f"To run a simulation, use: ./waf --run 'scratch/third {project_root}/configuration/ns3/configs/FoldedClos_16_configX.txt'")