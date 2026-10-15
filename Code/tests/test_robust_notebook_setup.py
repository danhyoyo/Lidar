"""Execute the real setup cell against both supported checkout layouts."""
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from IPython.core.interactiveshell import InteractiveShell

from notebook_test_utils import execute_shell_cell

ROOT = Path(__file__).resolve().parents[1]


def setup_source():
    notebook = json.loads((ROOT / '3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'DEFAULT_REPO_DIR' in ''.join(cell['source']))
    return source.split('# 2. Base directory configuration')[0]


def checkout(path, layout, *, stage_support=True):
    code = path / 'Code' if layout == 'nested' else path
    (code / 'configs').mkdir(parents=True)
    (code / 'configs/config.json').write_text('{}')
    (code / 'detector').mkdir()
    (code / 'tools/kitti_training_pipeline').mkdir(parents=True)
    (code / 'requirements-kitti.txt').touch()
    if stage_support:
        for relative in ('tools/kitti_training_pipeline/notebook_config.py',
                         'detector/core/backbone_config.py',
                         'detector/core/models/backbones/mobilepixornext.py',
                         'detector/core/models/backbones/mobilepixornext_blocks.py'):
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, target)
    return code.resolve()


def run_setup(tmp_path, monkeypatch, *, colab=True, clone_layout='nested', cwd=None, clone_stage_support=True):
    content = tmp_path / 'content'
    content.mkdir(exist_ok=True)
    commands = []

    def fake_clone(args, **kwargs):
        assert args[:3] == ['git', 'clone', '-b']
        commands.append(list(args))
        checkout(Path(args[-1]), clone_layout, stage_support=clone_stage_support)
        return SimpleNamespace(returncode=0)

    def fake_system(command):
        args = shlex.split(InteractiveShell.instance().var_expand(command))
        if args[:2] == ['git', 'clone']:
            return fake_clone(args).returncode
        assert args[:2] == ['pip', 'install']
        assert Path(args[-1]).is_file()
        commands.append(args)
        return 0

    monkeypatch.setattr(subprocess, 'run', fake_clone)
    monkeypatch.setattr(InteractiveShell.instance(), 'system', fake_system)
    if colab:
        google = ModuleType('google')
        module = ModuleType('google.colab')
        module.drive = SimpleNamespace(mount=lambda path: None)
        google.colab = module
        monkeypatch.setitem(sys.modules, 'google', google)
        monkeypatch.setitem(sys.modules, 'google.colab', module)
    else:
        monkeypatch.delitem(sys.modules, 'google.colab', raising=False)
    source = setup_source().replace('"/content', '"' + str(content))
    namespace = {}
    execute_shell_cell(source, namespace, cwd=cwd or content)
    return namespace, commands


@pytest.mark.parametrize('layout', ['nested', 'legacy'])
def test_existing_colab_checkout_uses_code_root_and_requirements(tmp_path, monkeypatch, layout):
    path = tmp_path / 'content/Lidar'
    code = checkout(path, layout)
    marker = path / 'user_file.txt'
    marker.write_text('preserve')
    namespace, commands = run_setup(tmp_path, monkeypatch)
    assert namespace['DEFAULT_REPO_DIR'] == code
    assert not any(args[:2] == ['git', 'clone'] for args in commands)
    assert commands[-1][-1] == str(code / 'requirements-kitti.txt')
    assert marker.read_text() == 'preserve'


def test_fresh_colab_clone_recognizes_reorganized_code_directory(tmp_path, monkeypatch):
    namespace, commands = run_setup(tmp_path, monkeypatch)
    code = (tmp_path / 'content/Lidar/Code').resolve()
    assert namespace['DEFAULT_REPO_DIR'] == code
    clones = [args for args in commands if args[:2] == ['git', 'clone']]
    assert len(clones) == 1
    assert clones[0][3] == 'research/mobilepixornext-under1m'
    assert commands[-1][-1] == str(code / 'requirements-kitti.txt')


@pytest.mark.parametrize('location', ['Lidar', 'drive/MyDrive/Lidar/Code'])
@pytest.mark.parametrize('stale_file', ['tools/kitti_training_pipeline/notebook_config.py',
    'detector/core/backbone_config.py', 'detector/core/models/backbones/mobilepixornext.py',
    'detector/core/models/backbones/mobilepixornext_blocks.py'])
def test_stale_colab_source_is_preserved_and_replaced_by_fresh_runtime_checkout(tmp_path, monkeypatch, location, stale_file):
    path = tmp_path / 'content' / location
    stale = checkout(path, 'nested')
    source = stale / stale_file
    source.write_text('def resolve_notebook_config(): pass\n')
    marker = path / 'user_file.txt'
    marker.write_text('preserve')
    namespace, commands = run_setup(tmp_path, monkeypatch, cwd=stale)
    assert namespace['DEFAULT_REPO_DIR'] != stale
    assert namespace['DEFAULT_REPO_DIR'].is_relative_to(tmp_path / 'content')
    assert len([args for args in commands if args[:2] == ['git', 'clone']]) == 1
    assert source.read_text() == 'def resolve_notebook_config(): pass\n'
    assert marker.read_text() == 'preserve'


def test_clone_without_attention_source_is_rejected_before_dependency_install(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match='stage|attention|source'):
        run_setup(tmp_path, monkeypatch, clone_stage_support=False)


def test_setup_switches_cached_pipeline_and_detector_modules_to_selected_source(tmp_path):
    old = checkout(tmp_path / 'old', 'nested')
    new = checkout(tmp_path / 'new', 'nested')
    for code in (old, new):
        shutil.copyfile(ROOT / 'tools/kitti_training_pipeline/common.py',
                        code / 'tools/kitti_training_pipeline/common.py')
        for relative in ('core/models/model.py', 'core/datasets/dataset.py', 'core/losses/loss_fn.py'):
            path = code / 'detector' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    stale = old / 'tools/kitti_training_pipeline/notebook_config.py'
    stale.write_text('def resolve_notebook_config(): pass\n')
    notebook = json.loads((ROOT / '3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    setup = ''.join(notebook['cells'][1]['source'])
    block = setup.split('# 3. Path & Module Configuration')[1].split('# 4. Device & Hardware Inspection')[0]
    script = '''
import importlib, os, sys
from pathlib import Path
old, new = map(Path, sys.argv[1:])
sys.path[:0] = [str(old), str(old / "tools/kitti_training_pipeline"), str(old / "detector")]
import common, notebook_config
stale_package = importlib.import_module("tools.kitti_training_pipeline.notebook_config")
old_resolver = importlib.import_module("core.backbone_config")
namespace = {"os": os, "sys": sys, "Path": Path, "REPO_DIR": new}
exec(BLOCK, namespace)
fresh = importlib.import_module("notebook_config")
package = importlib.import_module("tools.kitti_training_pipeline.notebook_config")
resolver = importlib.import_module("core.backbone_config")
assert Path(fresh.__file__).is_relative_to(new)
assert Path(package.__file__).is_relative_to(new)
assert Path(resolver.__file__).is_relative_to(new)
assert callable(fresh.attention_placement_overrides)
assert callable(package.attention_placement_summary)
'''.replace('BLOCK', repr(block))
    result = subprocess.run([sys.executable, '-c', script, str(old), str(new)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_local_launch_from_checkout_root_resolves_code(tmp_path, monkeypatch):
    path = tmp_path / 'Lidar'
    code = checkout(path, 'nested')
    namespace, commands = run_setup(tmp_path, monkeypatch, colab=False, cwd=path)
    assert namespace['DEFAULT_REPO_DIR'] == code
    assert not commands


def test_incomplete_existing_checkout_is_reported_without_deleting_files(tmp_path, monkeypatch):
    path = tmp_path / 'content/Lidar'
    path.mkdir(parents=True)
    marker = path / 'user_file.txt'
    marker.write_text('preserve')
    with pytest.raises(FileNotFoundError, match='existing|Existing|incomplete|Incomplete'):
        run_setup(tmp_path, monkeypatch)
    assert marker.read_text() == 'preserve'


def test_generator_produces_the_same_setup_and_extraction_cells(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'tools/generate_robust_notebook.py')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    generated = json.loads((tmp_path / '3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    setup = next(''.join(cell['source']) for cell in generated['cells']
                 if cell['cell_type'] == 'code' and 'DEFAULT_REPO_DIR' in ''.join(cell['source']))
    assert setup.split('# 2. Base directory configuration')[0] == setup_source()
    notebook = json.loads((ROOT / '3D_Lidar_Detection_Pipeline_Robust.ipynb').read_text())
    extraction = lambda cells: next(''.join(cell['source']) for cell in cells
                                   if 'archives_spec =' in ''.join(cell['source']))
    assert extraction(generated['cells']) == extraction(notebook['cells'])
