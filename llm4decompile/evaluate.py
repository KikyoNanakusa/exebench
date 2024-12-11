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
compile_error_count = 0


def parse_args() -> Namespace:
    """
    The function `parse_args` defines a parser for command line arguments with various options and
    default values.
    :return: The function `parse_args()` is returning an object of type `Namespace` which contains the
    parsed arguments provided through the command line or default values if not provided.
    """
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--gpus", type=int, default=2)  # Should be factor of 32
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.82)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max_total_tokens", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--debug", type=bool, default=False)
    return parser.parse_args()


def evaluate_func(params) -> tuple[int, int]:
    """
    The function `evaluate_func` compiles and runs a decompiled C function, checking for successful
    compilation and execution within a specified timeout period.

    :param params: The `evaluate_func` function takes in a dictionary `params` as input, which should
    contain the following keys:
    :return: The function `evaluate_func` returns a tuple containing two integers: `flag_compile` and
    `flag_run`.
    """
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    dataset_row = params["dataset_row"]
    decompiled_c_func = params["c_func_decompile"]

    # print(f"decompiled c func: \n{decompiled_c_func}") # for debugging

    timeout = 10
    flag_compile = 0
    flag_run = 0

    try:
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
            test_output,
            exebench_dict_to_dict(dataset_row["synth_io_pairs"]["output"][0]),
        ):
            flag_run = 1
    except Exception as e:
        logging.error(f"Error in Wrapper execution: {e}")
        return flag_compile, flag_run

    try:
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
            subprocess.run(compile_command, check=True, timeout=timeout)
            flag_compile = 1
    except subprocess.CalledProcessError as e:
        logging.error(f"GCC compilation failed: {e}")
    except subprocess.TimeoutExpired as e:
        logging.error(f"GCC compilation timed out: {e}")
    except Exception as e:
        logging.error(f"Unexpected error during GCC compilation: {e}")

    return flag_compile, flag_run


def safe_evaluate_func(task):
    """
    The `safe_evaluate_func` function attempts to evaluate a task using `evaluate_func` and logs any
    errors that occur, returning (0, 0) in case of an exception.

    :param task: It seems like you have provided a function `safe_evaluate_func` that wraps around
    another function `evaluate_func` to handle any exceptions that may occur during its execution.
    However, the definition of the `evaluate_func` function is missing in your code snippet
    :return: If an error occurs during the evaluation of the task, a tuple (0, 0) will be returned.
    """
    try:
        return evaluate_func(task)
    except Exception as e:
        logging.error(f"Error in task {task}: {e}")
        return (0, 0)


def decompile_pass_rate(testset, gen_results_repeat, args) -> int:
    """
    This function calculates and prints the average compile and run rates for different optimization
    levels based on evaluation results.

    :param testset: The `testset` parameter in the `decompile_pass_rate` function is likely a list of
    dictionaries containing information about test data. Each dictionary in the `testset` list probably
    has keys like "exebench_dict" and "opt" that store specific data related to the test cases
    :param gen_results_repeat: Gen_results_repeat is a list of results generated by a genetic algorithm
    for multiple iterations. Each iteration contains results for different optimizations applied to a
    test dataset
    :param args: The `args` parameter in the `decompile_pass_rate` function seems to be a custom object
    or dictionary that contains at least one key called `num_workers`. This key is used to specify the
    number of workers to be used in the multiprocessing pool for parallel processing
    :return: The function `decompile_pass_rate` is returning an integer value of 0.
    """
    import logging

    logging.basicConfig(
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

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

            eval_results = list(
                tqdm(pool.imap(safe_evaluate_func, tasks), total=len(tasks))
            )

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


def compile_and_write(function_name, input_text, error_counter) -> dict[str, str]:
    """
    The function `compile_and_write` compiles a C program, generates assembly code, cleans and filters
    the assembly code, and returns the cleaned assembly code for different optimization states.
    """
    asm_all = {}

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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Generate assembly code from object file using objdump
            subprocess.run(
                f"objdump -d --disassemble={function_name} {obj_output} > {asm_output}",
                shell=True,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            with open(asm_output) as f:
                asm = f.read()
                asm_clean = ""
                asm = asm.split("Disassembly of section .text:")[-1].strip()
                for tmp in asm.split("\n"):
                    tmp_asm = tmp.split("\t")[-1]  # remove the binary code
                    tmp_asm = tmp_asm.split("#")[0].strip()  # remove the comments
                    asm_clean += tmp_asm + "\n"
                if len(asm_clean.split("\n")) < 4:
                    raise ValueError("compile fails")
                asm = asm_clean

                # Filter digits and attributes
                asm = re.sub(zeros_pattern, "", asm)
                asm = asm.replace("__attribute__((used)) ", "")

                asm_all[opt_state] = asm

            # Remove the object file
            if os.path.exists(obj_output):
                os.remove(obj_output)

    except Exception:
        # カウントのインクリメント
        with error_counter.get_lock():
            error_counter.value += 1

    finally:
        # Remove the assembly output files
        for opt_state in OPT:
            asm_output = input_file_name + "_" + opt_state + ".s"
            if os.path.exists(asm_output):
                os.remove(asm_output)

        if os.path.exists(input_file_name):
            os.remove(input_file_name)

    return asm_all


def is_exebench_function_valid(row) -> bool:
    synth_wrapper = Wrapper(
        c_deps=row["synth_deps"]
        + "\n"
        + row["synth_io_pairs"]["dummy_funcs"][0]
        + "\n"
        + row["func_def"],
        func_c_signature=row["func_head_types"].replace("extern", ""),
        func_assembly=None,
        cpp_wrapper=row["synth_exe_wrapper"],
    )

    # Check if the decompiled function can be compiled and run correctly
    test_output = synth_wrapper(
        exebench_dict_to_dict(row["synth_io_pairs"]["input"][0])
    )

    return diff_io(
        test_output,
        exebench_dict_to_dict(row["synth_io_pairs"]["output"][0]),
    )


def run_eval_pipeline(args: Namespace) -> int:
    """
    The function `run_eval_pipeline` loads a model, processes compilation tasks, generates outputs using
    a language model, and calculates a decompile pass rate based on the generated results.

    :param args: The `args` parameter in the `run_eval_pipeline` function is of type `Namespace`, which
    is typically used to hold command-line arguments parsed by the `argparse` module in Python. It
    contains the arguments and their values provided by the user when running the script. These
    arguments can be accessed
    :type args: Namespace
    :return: The function `run_eval_pipeline` returns an integer value.
    """
    compile_error_counter = multiprocessing.Value("i", 0)

    # load the model
    model_path = Path(args.model_path)
    print(f"Model is {model_path}")

    # Prepare the model
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.gpus,
        max_model_len=args.max_total_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        stop=stop_sequences,
    )

    print(f"Model loaded: {model_path}")
    try:
        dataset = load_dataset("jordiae/exebench", split="test_synth")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        stop_sequences = [tokenizer.eos_token]

        # prompt templates
        before = "# This is the assembly code:\n"
        after = "\n# What is the source code?\n"
        inputs = []
        testset = []

        compile_count = 0
        progress_bar = tqdm(
            dataset,
            desc=f"Processing compilation, error count {compile_error_counter.value}",
        )

        count = 0  # for debugging
        for row in progress_bar:
            # Check exebench function itself is testable or not
            if not is_exebench_function_valid(row):
                break

            # Compile the C program to assembly
            c_source_code = (
                row["synth_deps"]
                + "\n"
                + row["synth_io_pairs"]["dummy_funcs"][0]
                + "\n"
                + row["func_def"]
            )
            asm_all: dict[str, str] = compile_and_write(
                row["fname"], c_source_code, compile_error_counter
            )

            # Prepare the prompt
            for opt, asm in asm_all.items():
                prompt = before + asm + after
                inputs.append(prompt)

                data = {"opt": opt, "prompt": prompt, "exebench_dict": row}
                testset.append(data)

            compile_count += 1
            # update the progress bar
            progress_bar.set_description(
                f"Processing compilation, error count {compile_error_counter.value} / {compile_count}"
            )

            if args.debug:
                count += 1
                if count > 5:
                    break

        gen_results_repeat = []
        logger.info(f"The exp will loop for {args.repeat} times....")
        for i in range(args.repeat):
            logger.info(f"The {i+1} loop...")
            gen_results = llm.generate(inputs, sampling_params)
            gen_results = [[output.outputs[0].text] for output in gen_results]
            if args.debug:
                print(f"gen_results: \n{gen_results}")
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
