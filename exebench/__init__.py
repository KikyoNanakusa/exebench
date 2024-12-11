import math
import json
from pathlib import Path
import subprocess
from typing import Optional, Tuple
import tempfile
import contextlib
import os
import stat
import shutil
import glob
import re
from ast import literal_eval

__all__ = ["diff_io", "Wrapper", "exebench_dict_to_dict"]

__version__ = 0.1

# UTILS (in a self-contained file to ease deployment)

_DEFAULT_CMD_TIMEOUT = 5
_ROOT_PATH_FOR_JSON_HPP = os.path.dirname(__file__)
_SYNTH_LIBS_PATH = os.path.dirname(__file__)


def _run_command(
    command: str,
    stdin: Optional[str] = None,
    timeout: Optional[int] = _DEFAULT_CMD_TIMEOUT,
) -> Tuple[str, str]:
    try:
        output = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            input=stdin,
            timeout=timeout,
            encoding="utf-8",
        )
        stdout = (
            output.stdout.decode("utf-8")
            if isinstance(output.stdout, bytes)
            else output.stdout
        )
        stderr = (
            output.stderr.decode("utf-8")
            if isinstance(output.stderr, bytes)
            else output.stderr
        )
        if stderr:
            print(f"Command error: {stderr}")
        return stdout, stderr
    except subprocess.TimeoutExpired as e:
        print(f"Command timeout: {e}")
        raise
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}: {e.output}")
        raise
    except Exception as e:
        print(f"Unexpected error while running command: {e}")
        raise


def _get_host_process_id():
    process_id = "exebench_" + os.uname()[1] + "_" + str(os.getpid())
    return process_id


def _cleanup(path_pattern):
    for path in glob.glob(path_pattern):
        try:
            shutil.rmtree(path)
        except:
            pass


@contextlib.contextmanager
def _get_tmp_path(
    content: Optional[str] = None, suffix: Optional[str] = None, delete=True
) -> str:
    tmp_dir = "/work/nas/tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    prefix = _get_host_process_id()
    try:
        with tempfile.NamedTemporaryFile(
            dir=tmp_dir, prefix=prefix, suffix=suffix, delete=False, mode="w+"
        ) as ntf:
            if content:
                ntf.write(content)
                ntf.flush()

            os.chmod(
                ntf.name,
                stat.S_IRUSR
                | stat.S_IWUSR
                | stat.S_IXUSR  # 所有者
                | stat.S_IRGRP
                | stat.S_IXGRP  # グループ
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
            yield ntf.name
    except Exception as e:
        print(f"Error creating temporary file: {e}")
        raise
    finally:
        if delete and os.path.exists(ntf.name):
            try:
                os.remove(ntf.name)
            except Exception as e:
                print(f"Error cleaning up temporary file: {e}")


class _Assembler:
    def __call__(self, c_deps, func_c_signature, func_assembly, cpp_wrapper) -> Path:
        raise NotImplemented


class _DefaultAssembler(_Assembler):
    def __call__(self, c_deps, func_c_signature, func_assembly, cpp_wrapper) -> Path:
        with _get_tmp_path(content=None, suffix=".x", delete=False) as executable_path:
            c_deps += f"\nextern {func_c_signature};\n"

            with _get_tmp_path(content=c_deps, suffix=".c") as c_deps_path:
                cpp_wrapper = re.sub(
                    r"extern\s\"C\"\s\{\s.*\s\}",
                    'extern "C" \n{\n#include "' + c_deps_path + '"\n}\n',
                    cpp_wrapper,
                )
                with _get_tmp_path(content=cpp_wrapper, suffix=".cpp") as cpp_path:
                    cmd = f"g++ -fpermissive -O0 -o {executable_path} {cpp_path} -I {_ROOT_PATH_FOR_JSON_HPP} -I{_SYNTH_LIBS_PATH}"

                    stdout, stderr = _run_command(cmd)

                    if stderr:
                        print(f"stderr: {stderr}")

        return Path(executable_path)


def _compile_exe_path(
    c_deps, func_c_signature, func_assembly, cpp_wrapper, assembler_backend
):
    return assembler_backend(c_deps, func_c_signature, func_assembly, cpp_wrapper)


# API
class Wrapper:
    def __init__(
        self,
        c_deps,
        func_c_signature,
        func_assembly,
        cpp_wrapper,
        assembler_backend=_DefaultAssembler(),
    ):
        self._compiled_exe_path = self._compile_exe_path(
            c_deps, func_c_signature, func_assembly, cpp_wrapper, assembler_backend
        )

    @staticmethod
    def _compile_exe_path(
        c_deps, func_c_signature, func_assembly, cpp_wrapper, assembler_backend
    ):
        return _compile_exe_path(
            c_deps, func_c_signature, func_assembly, cpp_wrapper, assembler_backend
        )

    def __call__(self, inp, return_stdout_and_stderr=False):
        executable = self._compiled_exe_path

        with _get_tmp_path(content=None, suffix=".json") as input_tmp_json_path:
            output_file = "".join(input_tmp_json_path.split(".")[:1]) + "-out.json"

            with open(input_tmp_json_path, "w") as f:
                json.dump(inp, f)

            stdout, stderr = _run_command(
                f"{executable} {input_tmp_json_path} {output_file}"
            )

            with open(output_file, "r") as f:
                output = json.load(f)
            os.remove(output_file)

        if return_stdout_and_stderr:
            return output, stdout, stderr

        return output


def diff_io(observed_output, expected_output) -> bool:
    if type(observed_output) is not type(expected_output):
        return False
    if isinstance(observed_output, list):
        if len(observed_output) != len(expected_output):
            return False
        for e1, e2 in zip(observed_output, expected_output):
            ok = diff_io(e1, e2)
            if not ok:
                return False
    elif isinstance(observed_output, dict):
        for key in observed_output:
            if key not in expected_output:
                return False
            ok = diff_io(observed_output[key], expected_output[key])
            if not ok:
                return False
    elif isinstance(observed_output, float):
        ok = math.isclose(observed_output, expected_output)
        if not ok:
            return False
    else:
        ok = observed_output == expected_output
        if not ok:
            return False
    return True


def _fix_nested_dict(inp):  # hack
    if isinstance(inp, dict):
        for k in inp:
            inp[k] = _fix_nested_dict(inp[k])
    elif isinstance(inp, list):
        for idx, e in enumerate(inp):
            inp[idx] = _fix_nested_dict(e)
    else:
        return literal_eval(inp)
    return inp


def exebench_dict_to_dict(exebench_dict):
    keys = exebench_dict["var"]
    values = exebench_dict["value"]
    return _fix_nested_dict({k: v for k, v in zip(keys, values)})
