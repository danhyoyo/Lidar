"""Execute real IPython shell escapes in an isolated notebook namespace."""
import json
import os
from pathlib import Path

from IPython.core.interactiveshell import InteractiveShell

ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(marker):
    notebook = json.loads((ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb").read_text())
    return next("".join(c["source"]) for c in notebook["cells"]
                if c["cell_type"] == "code" and marker in "".join(c["source"]))


def execute_shell_cell(source, namespace, cwd=ROOT):
    shell = InteractiveShell.instance()
    previous_namespace, previous_cwd = shell.user_ns, Path.cwd()
    shell.user_ns = namespace
    namespace["get_ipython"] = lambda: shell
    try:
        os.chdir(cwd)
        transformed = shell.input_transformer_manager.transform_cell(source)
        exec(compile(transformed, "notebook_cell", "exec"), namespace)
    finally:
        shell.user_ns = previous_namespace
        os.chdir(previous_cwd)
