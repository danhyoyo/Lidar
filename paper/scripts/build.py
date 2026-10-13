"""Build the supplied manuscript sources; no detector experiments are run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('auto', 'latexmk', 'tectonic'), default='auto')
    parser.add_argument('--tectonic', default='tectonic', help='Tectonic executable path')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    engine = args.engine
    if engine == 'auto':
        engine = 'latexmk' if shutil.which('latexmk') else 'tectonic'
    executable = 'latexmk' if engine == 'latexmk' else args.tectonic
    if not shutil.which(executable):
        parser.error(f'Executable not available: {executable}')
    logs = root / 'build_logs'
    logs.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault('XDG_CACHE_HOME', str(root / '.build_cache'))
    for name in ('main', 'supplementary'):
        command = ([executable, '-pdf', '-interaction=nonstopmode', '-halt-on-error', name + '.tex']
                   if engine == 'latexmk' else
                   [executable, '--keep-logs', '--keep-intermediates', name + '.tex'])
        result = subprocess.run(command, cwd=root, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (logs / (name + '.txt')).write_text(result.stdout)
        print(result.stdout)
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f'Built {root / (name + ".pdf")}')


if __name__ == '__main__':
    main()
