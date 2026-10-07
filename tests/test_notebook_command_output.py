"""Exercise the notebook's direct shell commands with actual CPU training."""
import json
import math
import shlex
import sys
import tempfile
from pathlib import Path

import pytest

from notebook_test_utils import ROOT, notebook_cell, execute_shell_cell


def smoke_namespace(tmp_path):
    from test_ap_training import setup
    from common import create_experiment_config, write_json
    config, _, _, _, _ = setup(tmp_path, primary='loss', epochs=1)
    return dict(RUN_SMOKE_TEST=True, config_dict=config, create_experiment_config=create_experiment_config,
        write_json=write_json, tempfile=tempfile, Path=Path, json=json, math=math,
        q=shlex.quote, sys=sys, DEVICE='cpu', SMOKE_TRAIN_BATCHES=1, SMOKE_VAL_BATCHES=1)


def test_direct_smoke_command_produces_finite_metrics_and_checkpoint(tmp_path):
    namespace = smoke_namespace(tmp_path)
    execute_shell_cell(notebook_cell('SMOKE_ROOT ='), namespace)
    assert namespace['_exit_code'] == 0
    assert namespace['smoke_checkpoint'].is_file()
    assert namespace['rows'][0]['optimizer_updates'] > 0
    assert json.loads(namespace['SMOKE_CONFIG'].read_text())['train']['checkpoint_selection']['primary'] == 'loss'


def test_failed_direct_command_stops_cell_before_metrics_are_read(tmp_path):
    namespace = smoke_namespace(tmp_path)
    namespace['config_dict']['data']['kitti']['location'] = str(tmp_path / 'missing data')
    with pytest.raises(RuntimeError, match='Smoke training failed'):
        execute_shell_cell(notebook_cell('SMOKE_ROOT ='), namespace)
    assert namespace['_exit_code'] != 0
    assert 'rows' not in namespace


def test_notebook_quotes_paths_and_literal_arguments_in_real_ipython_shell(tmp_path):
    output = tmp_path / "folder with spaces; literal $(no-command) ' quote" / 'argument.txt'
    output.parent.mkdir()
    value = 'literal $(no-command) `no-command` tiếng Việt'
    namespace = dict(q=shlex.quote, sys=sys, output=output, value=value)
    execute_shell_cell('''!{q(sys.executable)} -u -c 'import sys; from pathlib import Path; Path(sys.argv[1]).write_text(sys.argv[2])' {q(str(output))} {q(value)}
if _exit_code:
    raise RuntimeError("Command failed")
''', namespace)
    assert output.read_text() == value and namespace['_exit_code'] == 0
