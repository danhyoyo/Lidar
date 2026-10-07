"""Execute the notebook runner with actual children and Python output capture."""

import ast
import io
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def notebook_runner(cwd):
    notebook = json.loads((ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb").read_text())
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code" and "def run_command(" in "".join(cell["source"]))
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "run_command")
    namespace = {"REPO_DIR": cwd, "os": os, "subprocess": subprocess, "sys": sys}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "notebook_runner", "exec"), namespace)
    return namespace["run_command"]


def test_stdout_and_stderr_reach_notebook_python_capture(tmp_path, capsys):
    notebook_runner(tmp_path)([sys.executable, "-c",
                              "import sys; print('stdout line'); print('stderr line', file=sys.stderr)"])
    output = capsys.readouterr().out
    assert "stdout line" in output.splitlines() and "stderr line" in output.splitlines()
    assert "[RUN]" in output and "[PID]" in output


def test_child_output_is_visible_before_exit_without_an_explicit_child_flush(tmp_path, monkeypatch):
    acknowledgement = tmp_path / "ack"

    class Receiver(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if text.strip() == "READY":
                acknowledgement.write_text("received")
            return result

    output = Receiver()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("PYTHONUNBUFFERED", "0")
    child = """
import sys, time
from pathlib import Path
print("READY")
ack = Path(sys.argv[1])
deadline = time.monotonic() + 5
while not ack.is_file():
    if time.monotonic() > deadline:
        raise SystemExit("Notebook never acknowledged live output")
    time.sleep(.01)
print("ACKNOWLEDGED")
"""
    notebook_runner(tmp_path)([sys.executable, "-c", child, acknowledgement])
    assert acknowledgement.is_file() and "ACKNOWLEDGED" in output.getvalue()
    assert os.environ["PYTHONUNBUFFERED"] == "0"


def test_cwd_and_literal_arguments_are_preserved(tmp_path, capsys):
    working = tmp_path / "directory with spaces"
    working.mkdir()
    argument = "literal $(no-command) `no-command` tiếng Việt"
    notebook_runner(tmp_path)([sys.executable, "-c",
                              "import os,sys; print(os.getcwd()); print(sys.argv[1])", argument], cwd=working)
    output = capsys.readouterr().out
    assert str(working) in output.splitlines() and argument in output.splitlines()


def test_failed_child_preserves_exit_code_and_a_bounded_output_tail(tmp_path, capsys):
    child = "import sys; [print('line '+str(i)) for i in range(250)]; print('failure detail',file=sys.stderr); sys.exit(7)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        notebook_runner(tmp_path)([sys.executable, "-c", child])
    output = capsys.readouterr().out
    assert "failure detail" in output
    assert error.value.returncode == 7
    assert "failure detail" in (error.value.output or "")
    assert len(error.value.output.splitlines()) <= 100


@pytest.mark.skipif(os.name != "posix", reason="POSIX signal exit semantics")
def test_sigkill_is_reported_with_last_child_output(tmp_path, capsys):
    child = "import os,signal; print('before termination',flush=True); os.kill(os.getpid(),signal.SIGKILL)"
    with pytest.raises(subprocess.CalledProcessError) as error:
        notebook_runner(tmp_path)([sys.executable, "-c", child])
    assert error.value.returncode == -signal.SIGKILL
    assert "before termination" in (error.value.output or "")
    assert "SIGKILL" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix", reason="POSIX child-process liveness")
def test_notebook_interrupt_terminates_the_direct_child(tmp_path, monkeypatch):
    pid_file = tmp_path / "child.pid"

    class InterruptedReceiver(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if text.strip() == "INTERRUPT_READY":
                raise KeyboardInterrupt
            return result

    monkeypatch.setattr(sys, "stdout", InterruptedReceiver())
    child = """
import os,sys,time
from pathlib import Path
Path(sys.argv[1]).write_text(str(os.getpid()))
print("INTERRUPT_READY", flush=True)
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    time.sleep(.01)
"""
    with pytest.raises(KeyboardInterrupt):
        notebook_runner(tmp_path)([sys.executable, "-c", child, pid_file])
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
