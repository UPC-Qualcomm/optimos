#!/usr/bin/python3
import os
import subprocess
import multiprocessing
import argparse
from enum import Enum
from tqdm import tqdm


def run_command(command, cwd=None):
    result = subprocess.run(command, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"run fail! {command}")
    return True


def get_design_space(
    num_npus=64,
    dp={1, 2, 4, 8, 16},
    mp={1, 2, 4, 8, 16},
    pp={1, 2, 4, 8, 16},
    weight_sharded={True, False},
    max_ssp=64
):
    design_space = list()

    for ddp in dp:
        for mmp in mp:
            for ssharded in weight_sharded:
                for ppp in pp:
                    ssp = num_npus // (ddp * mmp * ppp)
                    if ssp < 1 or ssp > max_ssp or (num_npus != (ddp * mmp * ssp * ppp)):
                        continue
                    design_space.append((ddp, mmp, ssp, ppp, ssharded))
    return design_space

def get_design_space_no_sp(
    num_npus=64,
    dp={1, 2, 4, 8, 16},
    mp={1, 2, 4, 8, 16},
    pp={1, 2, 4, 8, 16},
    weight_sharded={True, False},
):
    design_space = list()
    ssp=1
    for ddp in dp:
        for mmp in mp:
            for ssharded in weight_sharded:
                for ppp in pp:
                    if num_npus != (ddp * mmp * ppp):
                        continue
                    design_space.append((ddp, mmp, ssp, ppp, ssharded))
    return design_space

class Model(Enum):
    T5_Small = 0
    T5_Base = 1
    T5_Large = 2
    GPT_Small = 3
    GPT_Medium = 4
    GPT_1300M = 5
    GPT_Neo_2700M = 6
    FLAN_T5_XXL_11B = 7
    GPT_13B = 8
    GPT_NeoX_20B = 9
    GPT_175B = 10
    PaLM_540B = 11
    GPT_Estimated_over_1T = 12
    Default = 13
    LLaMA_70B = 14
    Model_100B = 15
    Model_120B = 16
    LLaMA_8B = 17
    GPT_30B = 18
    GPT_40B = 19
    Simple = 20
    GPT_8B = 21
    GPT_70B = 22
    LLaMA_30B = 23
    LLaMA_405B = 24


    @staticmethod
    def get_model_type(model) -> str:
        """Returns the model architecture type string expected by main.py --model_type.

        GPT-family models use the 'gpt' graph builder; LLaMA/dense models use
        'dense'.  Everything else defaults to 'dense' as a safe fallback.
        """
        _GPT_MODELS = {
            Model.GPT_Small,
            Model.GPT_Medium,
            Model.GPT_1300M,
            Model.GPT_Neo_2700M,
            Model.GPT_13B,
            Model.GPT_NeoX_20B,
            Model.GPT_175B,
            Model.GPT_Estimated_over_1T,
            Model.GPT_30B,
            Model.GPT_40B,
            Model.GPT_8B,
            Model.GPT_70B,
            Model.Default,
            Model.Simple,
        }
        _LLAMA_MODELS = {
            Model.LLaMA_70B,
            Model.LLaMA_8B,
            Model.LLaMA_30B,
            Model.LLaMA_405B,
            Model.Model_100B,
            Model.Model_120B,
            Model.T5_Small,
            Model.T5_Base,
            Model.T5_Large,
            Model.FLAN_T5_XXL_11B,
            Model.PaLM_540B,
        }
        if model in _GPT_MODELS:
            return "gpt"
        return "llama"

    @staticmethod
    def get_model_params(model):
        """Returns parameters as
        [din, dout, dmodel, dff, batch, micro_batch, seq, head, num_stacks]
        """
        if model == Model.T5_Small:
            return [32128, 512, 512, 2048, [32], 32, 512, 8, 6]
        elif model == Model.T5_Base:
            return [32128, 768, 768, 3072, [2048], 2048, 512, 12, 12]
        elif model == Model.T5_Large:
            return [32128, 1024, 1024, 4096, [2048], 2048, 512, 16, 24]
        elif model == Model.GPT_Small:
            return [50257, 768, 768, 3072, [2048], 2048, 1024, 12, 12]
        elif model == Model.GPT_Medium:
            return [50257, 1024, 1024, 4096, [2048], 2048, 1024, 16, 24]
        elif model == Model.GPT_1300M:
            return [50257, 2048, 2048, 8192, [512], 512, 2048, 16, 24]
        elif model == Model.GPT_Neo_2700M:
            return [50257, 2560, 2560, 10240, [512], 512, 2048, 32, 32]
        elif model == Model.LLaMA_8B:
            return [128256, 4096, 4096, 14336, [32], 512, 4096, 32, 32]
        elif model == Model.FLAN_T5_XXL_11B:
            return [32128, 4096, 4096, 10240, [2048], 2048, 1024, 64, 24]
        elif model == Model.GPT_13B:
            return [50257, 5140, 5140, 20560, [2048], 2048, 2048, 40, 40]
        elif model == Model.GPT_NeoX_20B:
            return [50257, 6144, 6144, 24576, [2048], 2048, 2048, 64, 44]
        elif model == Model.GPT_8B:
            return [50257, 3072, 3072, 12288, [2048], 2048, 2048, 24, 24]
        elif model == Model.GPT_30B:
            return [50257, 6144, 6144, 24576, [2048], 2048, 2048, 48, 48]
        elif model == Model.GPT_70B:
            return [50257, 6144, 6144, 21504, [2048], 2048, 2048, 48, 72]
        elif model == Model.GPT_40B:
            return [50257, 8192, 8192, 28672, [64], 512, 2048, 32, 56]
        elif model == Model.LLaMA_30B:
            return [128256, 8192, 8192, 32768, [2048], 2048, 4096, 64, 56]
        elif model == Model.LLaMA_70B:
            return [128256, 8192, 8192, 32768, [16], 2048, 4096, 64, 80]
        elif model == Model.LLaMA_405B:
            return [128256, 16384, 16384, 67344, [2048], 2048, 4096, 128, 126]
        elif model == Model.Model_100B:
            return [32000, 32000, 9216, 36864, [2048], 2048, 72, 88]
        elif model == Model.Model_120B:
            return [32000, 32000, 10240, 40960, [2048], 2048, 2048, 80, 96]
        elif model == Model.GPT_175B:
            return [50257, 12288, 12288, 49152, [64], 2048, 2048, 96, 96]
        elif model == Model.PaLM_540B:
            return [50257, 18432, 18432, 73728, [2048], 2048, 4096, 72, 118]
        elif model == Model.GPT_Estimated_over_1T:
            return [50257, 20480, 20480, 81920, [2048], 2048, 4096, 128, 128]
        elif model == Model.Simple:
            return [1024, 1024, 1024, 4096, [32], 32, 32, 4, 4]
        else:
            return [51200, 25600, 25600, 25600 * 4, [1024], 1024, 1024, 1024, 32]

    # def get_model_params(model):
    #    din = 51200
    #    dout=25600
    #    dmodel=25600
    #    dff=25600*4
    #    batch=1024
    #    seq=1024
    #    head=1024
    #    num_stacks=32
    #    return [din, dout, dmodel, dff, batch, seq, head, num_stacks]


def generate_instance(design_point, model=Model.Default, folder_name="default",  custom_args=None):
    root = os.path.join(
        os.path.split(os.path.abspath(__file__))[0], "workload", folder_name
    )
    #root = f"/path/to/shared/storage/workload/{folder_name}"
    dp, mp, ssp, pp, weight_sharded = design_point

    din, dout, dmodel, dff, batch, micro_batch, seq, head, num_stacks = Model.get_model_params(model)
    cmd = (
        f"python main.py "
        f"--output_dir {root} "
        f"--output_name {dp}_{mp}_{ssp}_{pp}_{1 if weight_sharded else 0}.%d.et "
        #f"--comm_group {dp}_{mp}_{ssp}_{pp}_{1 if weight_sharded else 0}.json "
        f"--dp {dp} "
        f"--tp {mp} "
        f"--sp {ssp} "
        f"--pp {pp} "
        f"--dvocal {din} "
        #f"--dout {dout} "
        f"--dmodel {dmodel} "
        f"--dff {dff} "
        f"--batch '{[dp*batch[0]]}' "
        f"--micro_batch '{dp*batch[0]}' "
        f"--seq {seq} "
        f"--head {head} "
        f"--num_stacks {num_stacks} "
        f"--weight_sharded {weight_sharded} "
        f"--model_type {Model.get_model_type(model)} "
    )
    
    if custom_args is not None:
        activation_recompute = custom_args[0]
        tpsp = custom_args[1]
        mixed_precision = custom_args[2]
        print_gpu_vram = custom_args[3]
        ep = custom_args[4]
        kvhead = custom_args[5]
        experts = custom_args[6]
        kexperts = custom_args[7]
        
        cmd += (
            f"--activation_recompute {activation_recompute} "
            f"--tpsp {tpsp} "
            f"--mixed_precision {mixed_precision} "
            f"--print_gpu_vram {print_gpu_vram} "
            f"--ep {ep} "
            f"--kvhead {kvhead} "
            f"--experts {experts} "
            f"--kexperts {kexperts} "
        )
    
    cmd += f"--chakra_schema_version v0.0.4"
    cwd = os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        "astra-sim",
        "extern",
        "symbolic_tensor_graph",
    )
    print(cmd)
    run_command(cmd, cwd)

def str_to_bool(v):
    # Convert "true" to True and "false" to False
    return v.lower() in ("true", "t", "1", "yes", "y")

if __name__ == "__main__":
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=int,
        help="The model to explore",
        required=False,
        default=Model.Default,
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="The folder to dump generated files",
        required=False,
        default="Default",
    )
    
    num_npus = 32
    dp = {1, 2, 4}
    mp = {1, 2, 4}
    pp = {1, 2, 4}
    weight_sharded = {False}
    max_sp=16

    parser.add_argument(
        "--num_npus",
        type=int,
        default=num_npus,
        help="Number of NPUs"
    )
    parser.add_argument(
        "--dp",
        type=str,
        default=",".join(map(str, dp)),
        help="Data parallelism degrees, comma-separated"
    )
    parser.add_argument(
        "--mp",
        type=str,
        default=",".join(map(str, mp)),
        help="Model parallelism degrees, comma-separated"
    )
    parser.add_argument(
        "--pp",
        type=str,
        default=",".join(map(str, pp)),
        help="Pipeline parallelism degrees, comma-separated"
    )
    parser.add_argument(
        "--weight_sharded",
        type=str,
        default=",".join(map(str, weight_sharded)),
        help="whether weight sharded(True/False), comma-separated"
    )
    parser.add_argument(
        "--max_sp",
        type=int,
        default=max_sp,
        help="Maximum spatial parallelism"
    )

    # Additional custom arguments newly added in STG
    parser.add_argument(
        "--activation_recompute",
        type=str_to_bool,
        help="whether recompute activation",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--tpsp",
        type=str_to_bool,
        help="use tp+sp or tp only",
        required=False,
        default=True,
    )
    parser.add_argument("--model_type", type=str, default="dense", required=False)
    parser.add_argument(
        "--mixed_precision", type=str_to_bool, default=False, required=False
    )
    parser.add_argument(
        "--print_gpu_vram",
        type=str_to_bool,
        default=False,
        required=False,
        help="Whether to print per-GPU VRAM footprint (total / params / acts / grads) in GiB",
    )

    # These argument related to Expert Parallelism only for Mixture of Experts models.
    # We may add a preset models later.
    parser.add_argument(
        "--ep", type=int, help="expert parallel degree", required=False, default=1
    )
    parser.add_argument("--kvhead", type=int, default=8, required=False)
    parser.add_argument("--experts", type=int, default=8, required=False)
    parser.add_argument("--kexperts", type=int, default=2, required=False)

    args = parser.parse_args()
    custom_args = [
        args.activation_recompute,
        args.tpsp,
        args.mixed_precision,
        args.print_gpu_vram,
        args.ep,
        args.kvhead,
        args.experts,
        args.kexperts,
    ]
    num_npus = args.num_npus
    dp = set(map(int, args.dp.split(',')))
    mp = set(map(int, args.mp.split(',')))
    pp = set(map(int, args.pp.split(',')))
    weight_sharded = set(val.lower() == 'true' for val in args.weight_sharded.split(','))

    max_sp = args.max_sp
    model = args.model
    folder_name = args.folder_name

    design_space = get_design_space(num_npus, dp, mp, pp, weight_sharded, max_sp)
    design_space = [(1,16,2,1,0),(1,4,4,2,1),(2,4,4,1,1),(2,4,4,1,0),(1,4,2,4,0),
                    (2,4,2,2,0),(1,8,1,4,1),(1,8,1,4,0),(1,2,2,8,1),(2,2,2,4,0),
                    (2,2,2,4,1),(2,1,1,16,1)]
    func = partial(generate_instance, model=Model(int(model)), folder_name=folder_name, custom_args=custom_args)

    with multiprocessing.Pool(int(multiprocessing.cpu_count() * 0.95)) as pool:
        results = list(tqdm(pool.imap_unordered(func, design_space), total=len(design_space)))

