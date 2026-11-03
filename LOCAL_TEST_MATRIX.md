# PyBLP fork: full local testing of the integration branch

This branch stages all currently prepared optimization PRs. The fork's
master remains untouched. These changes are **not** certified by full
upstream PyBLP tests and are not an official PyBLP release.

## Installation and tests

```bash
git fetch origin
git switch integration/all-p0p1-hybrid
python -m pip install -e ".[tests]"
python -m pytest -q tests/test_pr_*.py
python -m pytest -q tests
```

Set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` for controlled
performance measurements. Use independent processes for peak RSS.

## Opt-in configurations

```python
import pyblp
pyblp.options.implicit_gradient_solver = 'dense'  # unchanged default
pyblp.options.implicit_gradient_solver = 'cg'
pyblp.options.implicit_gradient_solver = 'hybrid' # graph precondition for select markets
result = problem.solve(fp_type='outside')          # ordinary RC logit only
```

The outside-share mapping is stacked on the nonlinear-dispatch fix.
IV/GMM optimization is stacked on the simulation-inference fix.
Graph-Newton share inversion is intentionally excluded because it did
not consistently beat outside-share SQUAREM.

## Required gates before upstream merge

Test demand-only and supply-side BLP, nested models, micro moments,
second choices, epsilon scaling, fixed effects, resampling, two-step GMM,
and multiprocessing (spawn, fork, forkserver, optional pathos).
Compare estimates, objective, gradients, standard errors, convergence
flags and actual log-share residuals. Benchmark wall time and **peak RSS**
at fixed data, seed, optimization settings and accuracy.

Draft PRs in the fork:
#1 simulation covariance, #2 block IV/GMM, #3 worker options,
#4 outside-share contraction, #5 experimental implicit derivatives.
Earlier fixes are in contrib/share-contraction and
contrib/probability-tangents branches.
