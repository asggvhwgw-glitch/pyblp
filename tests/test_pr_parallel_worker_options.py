"""Check that spawn multiprocessing workers receive the parent's PyBLP options."""
import json
import os
import subprocess
import sys
import textwrap

def test_spawn_inherits_option_snapshot(tmp_path):
    probe = tmp_path / 'spawn_probe.py'
    probe.write_text(textwrap.dedent("""
        import json
        import multiprocessing
        import pyblp
        from pyblp.utilities import basics

        def snapshot(_):
            return [
                pyblp.options.micro_computation_chunks,
                pyblp.options.pseudo_inverses,
            ]

        if __name__ == '__main__':
            multiprocessing.set_start_method('spawn', force=True)
            pyblp.options.verbose = False
            pyblp.options.micro_computation_chunks = 7
            pyblp.options.pseudo_inverses = False
            with pyblp.parallel(2):
                print('OBSERVED:' + json.dumps(basics.pool.map(snapshot, [0, 1])))
    """))
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, timeout=60,
        env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    text = next(line[len('OBSERVED:'):] for line in result.stdout.splitlines() if line.startswith('OBSERVED:'))
    assert json.loads(text) == [[7, False], [7, False]]
