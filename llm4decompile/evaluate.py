import subprocess
import asyncio
from transformers import AutoTokenizer
import os
import json
from loguru import logger
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
from tqdm import tqdm
import multiprocessing
import tempfile
from vllm import LLM, SamplingParams
from datasets import load_dataset
from exebench import Wrapper, diff_io, exebench_dict_to_dict
import re

logger.add(sys.stdout, colorize=False, format="{time} {level} {message}")
os.environ["TOKENIZERS_PARALLELISM"] = "true"
zeros_pattern = r"^0+\s"  # 0000000000000...
OPT = ["O0", "O1", "O2", "O3"]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpus", type=int, default=3)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--testset_path", type=str)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    return parser.parse_args()


def evaluate_func(params) -> tuple[int, int]:
    dataset_row = params["dataset_row"]
    decompiled_c_func = params["c_func_decompile"]

    timeout = 10
    flag_compile = 0
    flag_run = 0

    synth_wrapper = Wrapper(
        c_deps=dataset_row["synth_deps"]
        + "\n"
        + dataset_row["synth_io_pairs"]["dummy_funcs"][0]
        + "\n"
        + decompiled_c_func,
        func_c_signature=dataset_row["func_head_types"].replace("extern", ""),
        func_assembly=None,
        cpp_wrapper=dataset_row["synth_exe_wrapper"],
    )

    # Check if the decompiled function can be compiled and run correctly
    test_output = synth_wrapper(
        exebench_dict_to_dict(dataset_row["synth_io_pairs"]["input"][0])
    )

    if diff_io(
        test_output, exebench_dict_to_dict(dataset_row["synth_io_pairs"]["output"][0])
    ):
        flag_run = 1

    with tempfile.TemporaryDirectory() as temp_dir:
        pid = os.getpid()
        c_file_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}.c")
        executable_onlyfunc = os.path.join(temp_dir, f"onlyfunc_{pid}")

        with open(c_file_onlyfunc, "w") as f:
            f.write(dataset_row["synth_deps"] + "\n" + decompiled_c_func)

        # Compile the C program to an assembly
        compile_command = [
            "gcc",
            "-c",
            "-S",
            c_file_onlyfunc,
            "-o",
            executable_onlyfunc,
            "-lm",
        ]
        try:
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
        except:
            return flag_compile, flag_run

    return flag_compile, flag_run


def decompile_pass_rate(testset, gen_results_repeat, args) -> int:
    all_stats = []

    for gen_index, gen_results in enumerate(gen_results_repeat):
        with multiprocessing.Pool(args.num_workers) as pool:
            tasks = [
                {
                    "dataset_row": data["exebench_dict"],
                    "c_func_decompile": output[0],
                }
                for data, output in zip(testset, gen_results)
            ]

            eval_results = list(tqdm(pool.imap(evaluate_func, tasks), total=len(tasks)))

        pool.terminate()
        pool.join()

        stats: dict[str, dict[str, int]] = {
            opt: {"compile": 0, "run": 0, "total": 0} for opt in OPT
        }

        for idx, (data, output, flag) in enumerate(
            tqdm(
                zip(testset, gen_results, eval_results),
                total=len(testset),
                desc="Evaluating",
            )
        ):
            flag_compile, flag_run = flag[0], flag[1]
            opt = data["opt"]

            stats[opt]["total"] += 1
            if flag_compile:
                stats[opt]["compile"] += 1
            if flag_run:
                stats[opt]["run"] += 1

        all_stats.append(stats)

    # average
    avg_stats: dict[str, dict[str, float]] = {
        opt: {"compile": 0, "run": 0, "total": 0} for opt in OPT
    }

    for stats in all_stats:
        for opt in OPT:
            avg_stats[opt]["compile"] += stats[opt]["compile"]
            avg_stats[opt]["run"] += stats[opt]["run"]
            avg_stats[opt]["total"] += stats[opt]["total"]

    for opt in OPT:
        avg_stats[opt]["compile"] /= len(gen_results_repeat)
        avg_stats[opt]["run"] /= len(gen_results_repeat)
        avg_stats[opt]["total"] /= len(gen_results_repeat)

    for opt, data in avg_stats.items():
        compile_rate = data["compile"] / data["total"] if data["total"] > 0 else 0
        run_rate = data["run"] / data["total"] if data["total"] > 0 else 0
        print(
            f"Optimization {opt}: Compile Rate: {compile_rate:.4f}, Run Rate: {run_rate:.4f}"
        )

    return 0


def compile_and_write(input_text) -> dict[str, str]:
    asm_all = {}
    if "/* Variables and functions */" in input_text:
        # Exclude macro and types
        input_text = input_text.split("/* Variables and functions */")[-1]
        input_text = "\n\n".join(input_text.split("\n\n")[1:])  # Exclude variables
        ##### begin of remove __attribute__
        input_text = input_text.replace("__attribute__((used)) ", "")
        ##### end of remove __attribute__

    input_file_name = "tmp.c"
    with open(input_file_name, "w") as f:
        f.write(input_text)

    try:
        for opt_state in OPT:
            obj_output = input_file_name + "_" + opt_state + ".o"
            asm_output = input_file_name + "_" + opt_state + ".s"

            # Compile the C program to object file
            subprocess.run(
                ["gcc", "-c", "-o", obj_output, input_file_name, "-" + opt_state],
                check=True,
            )

            # Generate assembly code from object file using objdump
            subprocess.run(
                f"objdump -d {obj_output} > {asm_output}",
                shell=True,  # Use shell to handle redirection
                check=True,
            )

            with open(asm_output) as f:
                asm = f.read()
                ##### start of clean up
                asm_clean = ""
                asm = asm.split("Disassembly of section .text:")[-1].strip()
                for tmp in asm.split("\n"):
                    tmp_asm = tmp.split("\t")[-1]  # remove the binary code
                    tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
                    asm_clean += tmp_asm + "\n"
                if len(asm_clean.split("\n")) < 4:
                    raise ValueError("compile fails")
                asm = asm_clean
                ##### end of clean up

                ##### start of filter digits and attribute
                asm = re.sub(zeros_pattern, "", asm)
                asm = asm.replace("__attribute__((used)) ", "")
                ##### end of filter digits

                asm_all[opt_state] = asm

            # Remove the object file
            if os.path.exists(obj_output):
                os.remove(obj_output)

    except Exception as e:
        print(f"Error {e}")

    finally:
        # Remove the assembly output files
        for opt_state in OPT:
            asm_output = input_file_name + "_" + opt_state + ".s"
            if os.path.exists(asm_output):
                os.remove(asm_output)

        if os.path.exists(input_file_name):
            os.remove(input_file_name)

    return asm_all


def run_eval_pipeline(args: Namespace) -> int:
    # load the model
    model_path = Path(args.model_path)
    if not model_path.exists() or not model_path.is_dir():
        logger.error(f"Invalid model {model_path}")
        return -1
    print(f"Model loaded from {model_path}")

    try:
        dataset = load_dataset("jordiae/exebench", split="test_synth")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        stop_sequences = [tokenizer.eos_token]

        # prompt templates
        before = "# What is the source code?\n"
        after = "\n# What is the source code?\n"
        inputs = []
        testset = []

        for row in tqdm(dataset, desc="Processing compilation"):
            # compile the C program to assembly
            c_source_code = (
                row["synth_deps"]
                + "\n"
                + row["synth_io_pairs"]["dummy_funcs"][0]
                + "\n"
                + row["func_def"]
            )
            asm_all: dict[str, str] = compile_and_write(c_source_code)

            # Prepare the prompt
            for opt, asm in asm_all.items():
                prompt = before + asm + after
                inputs.append(prompt)

                data = {"opt": opt, "prompt": prompt, "exebench_dict": row}
                testset.append(data)

        # Prepare the model
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.gpus,
            max_model_len=args.max_total_tokens,
            # max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens,
            stop=stop_sequences,
        )

        gen_results_repeat = []
        logger.info(f"The exp will loop for {args.repeat} times....")
        for i in range(args.repeat):
            logger.info(f"The {i+1} loop...")
            gen_results = llm.generate(inputs, sampling_params)
            gen_results = [[output.outputs[0].text] for output in gen_results]
            gen_results_repeat.append(gen_results)

    except Exception as e:
        logger.error(e)
        traceback.print_exc()
        return -1

    ret = decompile_pass_rate(testset, gen_results_repeat, args)
    return ret


def main():
    args = parse_args()
    ret = run_eval_pipeline(args)
    sys.exit(ret)


if __name__ == "__main__":
    main()
