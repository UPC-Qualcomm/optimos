import os
import subprocess
from typing import Dict

import sys
sys.path.insert(0, os.environ['OPTIMOS_ROOT'])
from generate_workloads import Model


def generate_workload_with_env(design_point: Dict, model, folder_name, suffix=""):
    """
    Generate workload using the correct Python environment.
    
    Args:
        design_point: Configuration dictionary with dp, mp, sp, pp, sharded
        model: Model enum
        folder_name: Output folder name
        model_data: Optional dictionary containing model-specific data
    Returns:
        True if successful, False otherwise
    """
    root = os.path.join(os.environ['OPTIMOS_ROOT'], 'workload', folder_name)
    stg_main = os.path.join(os.environ['ASTRA_SIM_ROOT'], 'extern', 'symbolic_tensor_graph', 'main.py')
    if not os.path.isfile(stg_main):
        raise FileNotFoundError(f"Workload generator entrypoint not found: {stg_main}")
    
    # Extract values from design_point dictionary
    dp = design_point['dp']
    mp = design_point['mp']
    ssp = design_point['sp']
    pp = design_point['pp']
    sharded = design_point['sharded']
    din, dout, dmodel, dff, batch, micro_batch, seq, head, num_stacks = Model.get_model_params(model)
    batch = [batch[0] * dp]
    micro_batch = batch[0] * dp
    #batch = [batch[0]]
    #micro_batch = batch[0]
    if (design_point.get('batch_size') is not None):
        batch = [design_point['batch_size'] * dp]
        micro_batch = design_point['batch_size']  * dp
        #batch = [design_point['batch_size']]
        #micro_batch = design_point['batch_size'] 
        
    model_type = Model.get_model_type(model)
    
    print("Generating workload for model:", model)
    print(f"Parameters: din={din}, dmodel={dmodel}, dff={dff}, batch={batch}, micro_batch={micro_batch}, seq={seq}, head={head}, num_stacks={num_stacks}, dp={dp}, mp={mp}, sp={ssp}, pp={pp}, sharded={sharded}, model_type={model_type}")
    # Note: Having if the micro batch is much smaller than the global batch, the generator will be much slower.
    cmd = (
        f"{os.environ['ASTRA_SIM_PYTHON']} {stg_main} "
        f"--output_dir {root} "
        f"--output_name {dp}_{mp}_{ssp}_{pp}_{1 if sharded else 0}.%d.et "
        f"--dp {dp} "
        f"--tp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--dvocal {din} "
        f"--dmodel {dmodel} "
        f"--dff {dff} "
        f"--batch '{batch}' "
        f"--micro_batch '{micro_batch}' "
        f"--seq {seq} "
        f"--head {head} "
        f"--num_stacks {num_stacks} "
        f"--weight_sharded {sharded} "
        f"--model_type {model_type} "
        f"--chakra_schema_version v0.0.4 "
        f"--suffix {suffix}"
    )
    cwd = os.path.join(os.environ['ASTRA_SIM_ROOT'], 'extern', 'symbolic_tensor_graph')
    
    print("generate_workload command:", cmd)
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    return result.returncode == 0


def get_design_space(num_npus=64, dp_range=None, mp_range=None, pp_range=None, sharded_options=None):
    """Generate all possible parallelism configurations for given NPUs"""
    if dp_range is None:
        dp_range = [1, 2, 4, 8]
    if mp_range is None:
        mp_range = [1, 2, 4, 8, 16, 32, 64, 128]
    if pp_range is None:
        pp_range = [1, 2, 4, 8, 16, 32, 64]
    if sharded_options is None:
        sharded_options = [True, False]
    
    design_space = []
    
    for dp in dp_range:
        for mp in mp_range:
            for pp in pp_range:
                for sharded in sharded_options:
                    # Calculate spatial parallelism
                    sp = num_npus // (dp * mp * pp)
                    
                    # Check if configuration is valid
                    if sp >= 1 and dp * mp * sp * pp == num_npus:
                        design_space.append((dp, mp, sp, pp, sharded))
    
    return design_space