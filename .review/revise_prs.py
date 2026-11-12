"""Apply the narrowly scoped maintainer revisions to exact PR snapshots."""
import argparse
import ast
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f'Expected one occurrence of {old[:100]!r}; got {text.count(old)}')
    return text.replace(old, new, 1)


CONTRACTION_TEST = '''@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('fp_type', ['linear', 'safe_linear'])
@pytest.mark.parametrize('clipping', ['none', 'lower', 'upper', 'both'])
def test_contraction_jacobian(
    simulated_problem: SimulatedProblemFixture,
    fp_type: str,
    clipping: str,
) -> None:
    """Use central finite differences to test linear contraction Jacobians with scaling and share clipping."""
    _, _, problem, _, results = simulated_problem

    # there is no fixed point to differentiate without random coefficients
    if problem.K2 == 0:
        return pytest.skip("Mean utilities have a closed-form solution without random coefficients.")

    # only do the test for a single market
    market = Market(
        problem, problem.unique_market_ids[0], results._parameters, results.sigma, results.pi, results.rho,
        beta=results.beta, gamma=results.gamma, delta=results.delta,
    )
    assert market.delta is not None

    # test the fixture's scale and scales on either side of one when nesting permits scaling
    epsilon_scales = [problem.epsilon_scale] if market.H > 0 else [problem.epsilon_scale, 0.5, 2.0]
    for epsilon_scale in epsilon_scales:
        market.epsilon_scale = epsilon_scale
        shares = market.compute_probabilities(market.delta)[0] @ market.agents.weights

        # put bounds strictly between distinct shares to stay away from nondifferentiable clipping points
        unique_shares = np.unique(shares)
        assert unique_shares.size >= 3
        lower = float(unique_shares[:2].mean()) if clipping in {'lower', 'both'} else -np.inf
        upper = float(unique_shares[-2:].mean()) if clipping in {'upper', 'both'} else np.inf
        clipped = ((shares < lower) | (shares > upper)).ravel()
        assert clipped.any() == (clipping != 'none')
        if clipping in {'lower', 'both'}:
            assert (shares < lower).any()
        if clipping in {'upper', 'both'}:
            assert (shares > upper).any()

        def test_jacobian(initial: Array, contraction: Callable, _: Callable) -> Tuple[Array, bool]:
            """Check the Jacobian without attempting to solve an artificially clipped fixed point."""
            exact = contraction(initial)[2]
            assert exact is not None
            approximate = compute_finite_differences(lambda x: contraction(x)[0], initial)
            np.testing.assert_allclose(exact, approximate, atol=1e-7, rtol=1e-6)

            # a locally constant clipped share leaves only the identity contribution in its row
            np.testing.assert_array_equal(exact[clipped], np.eye(market.J)[clipped])
            contraction(initial)
            return initial, True

        # check both the analytic derivative and the clipping indicator returned by the actual contraction
        _, actual_clipped, _, errors = market.compute_delta(
            market.delta, Iteration(test_jacobian, compute_jacobian=True), fp_type, (lower, upper),
        )
        assert not errors
        np.testing.assert_array_equal(actual_clipped.ravel(), clipped)


'''

TANGENT_TEST = '''@pytest.mark.usefixtures('simulated_problem')
@pytest.mark.parametrize('keep_conditionals', [False, True])
def test_probability_tangents(simulated_problem: SimulatedProblemFixture, keep_conditionals: bool) -> None:
    """Use central finite differences to test the contribution of xi to probability tangents."""
    _, _, problem, _, results = simulated_problem
    if results._parameters.P == 0:
        return pytest.skip("There are no unfixed parameters to differentiate.")

    # only do the test for a single market, using the existing simulated parameters and choice probabilities
    market = Market(
        problem, problem.unique_market_ids[0], results._parameters, results.sigma, results.pi, results.rho,
        beta=results.beta, gamma=results.gamma, delta=results.delta,
    )
    delta = market.delta
    assert delta is not None
    probabilities, conditionals = market.compute_probabilities()
    probability_tangents, conditional_tangents = market.compute_probabilities_by_parameter_tangent_mapping(
        probabilities, conditionals, keep_conditionals=keep_conditionals,
    )
    original_probability_tangents = copy.deepcopy(probability_tangents)
    original_conditional_tangents = copy.deepcopy(conditional_tangents)

    # arbitrary directions test the update independently of the implicit mean utility derivative calculation
    xi_jacobian = np.random.RandomState(0).normal(size=(market.J, results._parameters.P))
    market.update_probabilities_by_parameter_tangent_mapping(
        probability_tangents, conditional_tangents, probabilities, conditionals, xi_jacobian,
    )

    for p, parameter in enumerate(results._parameters.unfixed):
        probability_tangent = probability_tangents[p]
        conditional_tangent = conditional_tangents[p]

        # beta's total derivative is zero, whereas gamma and phi leave existing tangents unchanged
        if isinstance(parameter, BetaParameter):
            np.testing.assert_array_equal(probability_tangent, np.zeros_like(probability_tangent))
            if conditional_tangent is not None:
                np.testing.assert_array_equal(conditional_tangent, np.zeros_like(conditional_tangent))
        elif isinstance(parameter, (GammaParameter, PhiParameter)):
            np.testing.assert_array_equal(probability_tangent, original_probability_tangents[p])
            if conditional_tangent is not None:
                np.testing.assert_array_equal(conditional_tangent, original_conditional_tangents[p])
        else:
            direction = xi_jacobian[:, [p]]

            def compute_perturbed_probabilities(shift: Array) -> Array:
                """Perturb mean utility in the selected direction, holding other parameters fixed."""
                return market.compute_probabilities(delta + shift.item() * direction)[0]

            approximate = compute_finite_differences(compute_perturbed_probabilities, np.zeros(1))
            np.testing.assert_allclose(
                probability_tangent, original_probability_tangents[p] + approximate.reshape(probabilities.shape),
                atol=1e-7, rtol=1e-6,
            )
            if conditional_tangent is not None:
                assert conditionals is not None
                original_conditional_tangent = original_conditional_tangents[p]
                assert original_conditional_tangent is not None

                def compute_perturbed_conditionals(shift: Array) -> Array:
                    """Perturb mean utility and compute within-nest choice probabilities."""
                    perturbed = market.compute_probabilities(delta + shift.item() * direction)[1]
                    assert perturbed is not None
                    return perturbed

                approximate = compute_finite_differences(compute_perturbed_conditionals, np.zeros(1))
                np.testing.assert_allclose(
                    conditional_tangent,
                    original_conditional_tangent + approximate.reshape(conditionals.shape),
                    atol=1e-7, rtol=1e-6,
                )

        # omitting conditional tangents must not cause the update to create them
        if not keep_conditionals:
            assert conditional_tangent is None


'''


def revise(root: Path, number: int) -> None:
    market_path = root / 'pyblp/markets/market.py'
    test_path = root / 'tests/test_blp.py'
    market = market_path.read_text()
    tests = test_path.read_text()
    if number == 200:
        market = replace_once(
            market,
            'header.extend([("Delta" if fp_type in {\'linear\', \'safe_linear\'} else "Exp Delta", "Max Norm"), '
            '("Max Norm", "Improvement")])',
            'header.extend([("Delta" if \'linear\' in fp_type else "Exp Delta", "Max Norm"), '
            '("Max Norm", "Improvement")])',
        )
        market = replace_once(market, "        if fp_type in {'linear', 'safe_linear'}:\n", "        if 'linear' in fp_type:\n")
        market = replace_once(
            market,
            '                    weighted_ratios = self.agents.weights * probability_ratios.T\n'
            '                    jacobian = (x / share_ratios) * (probability_ratios @ weighted_ratios)\n',
            '                    shares = x0 * share_ratios\n'
            '                    probabilities = x0 * probability_ratios\n'
            '                    weighted_probabilities = self.agents.weights * probabilities.T\n'
            '                    jacobian = x / x0.T * (probabilities @ weighted_probabilities) / shares\n',
        )
        market = replace_once(
            market,
            '                    if self.epsilon_scale != 1:\n                        jacobian /= self.epsilon_scale\n',
            '\n                    # account for the scale of the idiosyncratic preference shock\n'
            '                    if self.epsilon_scale != 1:\n                        jacobian /= self.epsilon_scale\n',
        )
        old = '                    if clipped_shares.any():\n'
        assert market.count(old) == 2
        market = market.replace(old, '\n                    # clipped shares are locally constant in mean utility\n' + old)
        tests = replace_once(tests, 'from pyblp.utilities.basics import (',
                             'from pyblp.markets.market import Market\nfrom pyblp.utilities.basics import (')
        anchor = "@pytest.mark.usefixtures('simulated_problem')\ndef test_probabilities("
        tests = replace_once(tests, anchor, CONTRACTION_TEST + anchor)
        (root / 'tests/test_pr_share_contraction.py').unlink()
    elif number == 201:
        market = replace_once(market,
            '        """Add xi\'s contribution directly, without constructing a J by J by I tensor."""',
            '        """Update tangents of probabilities with respect to parameters to account for the contribution of xi."""',
        )
        market = replace_once(market,
            '            conditionals_tangent = conditionals_tangent_mapping[p]\n            if isinstance(parameter, BetaParameter):',
            '            conditionals_tangent = conditionals_tangent_mapping[p]\n\n'
            '            # total derivatives are zero for beta parameters\n            if isinstance(parameter, BetaParameter):',
        )
        market = replace_once(market,
            '                continue\n            if isinstance(parameter, (GammaParameter, PhiParameter)):',
            '                continue\n\n            # derivatives remain zero for gamma and phi parameters\n'
            '            if isinstance(parameter, (GammaParameter, PhiParameter)):',
        )
        market = replace_once(market,
            '                continue\n            direction = xi_jacobian[:, [p]]',
            '                continue\n\n            # add the contribution of xi to unconditional probability tangents\n'
            '            direction = xi_jacobian[:, [p]]',
        )
        market = replace_once(market,
            '            probabilities_tangent += probabilities * (direction - mean) / self.epsilon_scale\n'
            '            if self.H > 0:',
            '            probabilities_tangent += probabilities * (direction - mean) / self.epsilon_scale\n\n'
            '            # account for within-nest substitution and update conditional probability tangents\n'
            '            if self.H > 0:',
        )
        start = market.index('    def compute_probabilities_by_xi_tensor(')
        end = market.index('    def compute_shares_by_variable_jacobian(', start)
        market = market[:start] + market[end:]
        (root / 'tests/test_pr_probability_tangents.py').unlink()
        for path in (root / 'pyblp').rglob('*.py'):
            source = market if path == market_path else path.read_text()
            assert 'compute_probabilities_by_xi_tensor' not in source, path
        tests = replace_once(tests, 'from pyblp.utilities.basics import (',
            'from pyblp.markets.market import Market\n'
            'from pyblp.parameters import BetaParameter, GammaParameter, PhiParameter\n'
            'from pyblp.utilities.basics import (',
        )
        anchor = "@pytest.mark.usefixtures('simulated_problem')\ndef test_surplus("
        tests = replace_once(tests, anchor, TANGENT_TEST + anchor)
    else:
        raise ValueError(number)
    ast.parse(market)
    ast.parse(tests)
    market_path.write_text(market)
    test_path.write_text(tests)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('number', type=int, choices=[200, 201])
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    revise(args.root, args.number)
