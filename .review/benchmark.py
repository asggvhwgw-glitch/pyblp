"""Reproduce fixed-parameter Problem.solve timing and memory comparisons for PR 201."""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import textwrap
import time

import numpy as np

BASE = '16536807004b16c24fb40b931531b45b33884e00'
ORIGINAL_PR = '14f086c247c19b796df3b0ce08c72e11ab27953c'
FIELDS = ('objective', 'gradient', 'delta', 'beta', 'micro_values', 'micro_covariances', 'parameter_covariances')


def source_at(commit: str, path: str) -> str:
    return subprocess.check_output(['git', 'show', f'{commit}:{path}'], text=True)


def run_one(mode: str, directory: Path) -> None:
    sys.path.insert(0, str(Path.cwd()))
    import pyblp
    market_module = importlib.import_module('pyblp.markets.market')
    Market = market_module.Market
    pyblp.options.verbose = False
    if mode == 'baseline':
        source = source_at(BASE, 'pyblp/markets/market.py')
        market_node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Market')
        for name in ['compute_probabilities_by_xi_tensor', 'update_probabilities_by_parameter_tangent_mapping']:
            node = next(n for n in market_node.body if isinstance(n, ast.FunctionDef) and n.name == name)
            namespace = dict(vars(market_module))
            exec(compile(textwrap.dedent(ast.get_source_segment(source, node)), '<upstream reference>', 'exec'), namespace)
            setattr(Market, name, namespace[name])

    kernel_times = []
    original_update = Market.update_probabilities_by_parameter_tangent_mapping

    def timed_update(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original_update(*args, **kwargs)
        finally:
            kernel_times.append(time.perf_counter() - start)

    Market.update_probabilities_by_parameter_tangent_mapping = timed_update
    source = source_at(ORIGINAL_PR, 'tests/test_pr_probability_tangents.py')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'build_micro_problem')
    factory_source = ast.get_source_segment(source, node)
    assert factory_source.count('T, J, I = 3, 12, 80') == 1
    factory_source = factory_source.replace('T, J, I = 3, 12, 80', 'T, J, I = 3, 200, 1000')
    namespace = {'np': np, 'pyblp': pyblp}
    exec(compile(factory_source, '<benchmark fixture>', 'exec'), namespace)
    problem, options = namespace['build_micro_problem']()

    # Warm up once; each subsequent solve evaluates the same objective and analytic gradient at fixed theta.
    problem.solve(**options)
    elapsed, kernels, calls = [], [], []
    for _ in range(3):
        kernel_times.clear()
        start = time.perf_counter()
        results = problem.solve(**options)
        elapsed.append(time.perf_counter() - start)
        assert kernel_times, 'The benchmark did not exercise probability tangent updates.'
        kernels.append(sum(kernel_times))
        calls.append(len(kernel_times))
    np.savez(directory / f'{mode}.npz', **{field: getattr(results, field) for field in FIELDS})
    report = {
        'mode': mode, 'T': 3, 'J': 200, 'I': 1000, 'repeats': 3,
        'solve_seconds': elapsed, 'median_solve_seconds': statistics.median(elapsed),
        'kernel_seconds': kernels, 'median_kernel_seconds': statistics.median(kernels),
        'kernel_calls': calls, 'peak_process_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'python': platform.python_version(), 'numpy': np.__version__, 'pyblp': pyblp.__version__,
        'threads': {key: os.environ.get(key) for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']},
        'scope': 'Fixed-parameter Problem.solve objective/gradient evaluation with an inside-purchase micro moment; not full parameter optimization.',
    }
    (directory / f'{mode}.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    mode, destination = sys.argv[1:]
    directory = Path(destination).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if mode in {'baseline', 'optimized'}:
        run_one(mode, directory)
        return
    if mode != 'compare':
        raise ValueError(mode)
    for implementation in ['baseline', 'optimized']:
        subprocess.run([sys.executable, __file__, implementation, str(directory)], check=True)
    baseline = json.loads((directory / 'baseline.json').read_text())
    optimized = json.loads((directory / 'optimized.json').read_text())
    differences = {}
    with np.load(directory / 'baseline.npz') as old, np.load(directory / 'optimized.npz') as new:
        for field in FIELDS:
            np.testing.assert_allclose(new[field], old[field], rtol=3e-8, atol=3e-9, equal_nan=True, err_msg=field)
            differences[field] = float(np.nanmax(np.abs(new[field] - old[field]))) if old[field].size else 0.0
    summary = {
        'numerical_equivalence': 'passed', 'max_absolute_differences': differences,
        'solve_speedup': baseline['median_solve_seconds'] / optimized['median_solve_seconds'],
        'kernel_speedup': baseline['median_kernel_seconds'] / optimized['median_kernel_seconds'],
        'rss_reduction_mib': baseline['peak_process_rss_mib'] - optimized['peak_process_rss_mib'],
        'baseline': baseline, 'optimized': optimized,
        'memory_note': 'Process-wide peak RSS in separate processes, including imports, warmup, and solve; not a kernel-only allocation measurement.',
    }
    (directory / 'comparison.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
