"""Use a central-difference step that avoids cancellation in log-share contractions."""
import ast
from pathlib import Path
import sys

if sys.argv[1] == '200':
    path = Path(sys.argv[2]) / 'tests/test_blp.py'
    source = path.read_text()
    old = '            approximate = compute_finite_differences(lambda x: contraction(x)[0], initial)\n'
    new = (
        '            # enlarge the central-difference step to avoid cancellation when delta is far from zero\n'
        '            approximate = compute_finite_differences(lambda x: contraction(x)[0], initial, epsilon_scale=1000)\n'
    )
    assert source.count(old) == 1
    source = source.replace(old, new, 1)
    ast.parse(source)
    path.write_text(source)
