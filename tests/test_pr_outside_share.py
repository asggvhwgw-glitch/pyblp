"""Opt-in outside-share contraction: ordinary random coefficients logit only."""
import numpy as np
import pytest
import pyblp


@pytest.mark.parametrize('outside_share', [0.15, 0.005])
@pytest.mark.parametrize('method', ['squarem', 'lm'])
def test_outside_share_matches_standard_inversion(outside_share, method):
    rng = np.random.default_rng(111)
    T, J, I = 4, 8, 50
    N = T * J
    ids = np.repeat(np.arange(T), J)
    prices = rng.uniform(0.8, 2.0, N)
    nodes = rng.normal(size=(T * I, 1))
    sigma = 0.4
    delta = np.empty(N)
    shares = np.empty(N)
    for t in range(T):
        p = slice(t * J, (t + 1) * J)
        a = slice(t * I, (t + 1) * I)
        x = prices[p]
        utilities = -x[:, None] - 3 + sigma * x[:, None] * nodes[a].T
        # Shift the utilities so that the outside share is in the target range.
        low, high = -20.0, 20.0
        for _ in range(70):
            shift = (low + high) / 2
            exponentials = np.exp(utilities + shift)
            model_outside = np.mean(1 / (1 + exponentials.sum(axis=0)))
            if model_outside > outside_share:
                low = shift
            else:
                high = shift
        shift = (low + high) / 2
        exponentials = np.exp(utilities + shift)
        delta[p] = -x - 3 + shift
        shares[p] = np.mean(exponentials / (1 + exponentials.sum(axis=0)), axis=1)

    product_data = {
        'market_ids': ids, 'prices': prices, 'shares': shares,
        'demand_instruments0': prices**2,
    }
    agent_data = {
        'market_ids': np.repeat(np.arange(T), I), 'nodes': nodes,
        'weights': np.full(T * I, 1 / I),
    }
    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices'), pyblp.Formulation('0 + prices')],
        product_data, agent_data=agent_data,
    )
    iteration = (
        pyblp.Iteration('squarem', {'atol': 1e-13})
        if method == 'squarem'
        else pyblp.Iteration('lm', {'xtol': 1e-12, 'ftol': 1e-12, 'gtol': 1e-12}, compute_jacobian=True)
    )
    common = dict(
        sigma=np.array([[sigma]]),
        sigma_bounds=(np.array([[sigma]]), np.array([[sigma]])),
        iteration=iteration, method='1s', error_behavior='raise',
        initial_update=False,
    )
    outside = problem.solve(fp_type='outside', **common)
    baseline = problem.solve(fp_type='safe_linear', **common)
    np.testing.assert_allclose(outside.delta, baseline.delta, atol=5e-8, rtol=0)
    np.testing.assert_allclose(outside.delta.ravel(), delta, atol=5e-8, rtol=0)
    np.testing.assert_allclose(outside.compute_shares(), problem.products.shares, rtol=1e-8, atol=1e-12)


@pytest.mark.parametrize('nested,epsilon', [(True, 1.0), (False, 0.5)])
def test_outside_map_rejects_unsupported_models(nested, epsilon):
    products = {
        'market_ids': [0, 0, 0, 1, 1, 1],
        'prices': [1.0, 1.2, 1.4, 0.8, 1.3, 1.7],
        'shares': [0.10, 0.11, 0.12, 0.11, 0.12, 0.13],
        'demand_instruments0': [1, 2, 3, 4, 5, 6],
    }
    if nested:
        products['nesting_ids'] = [0, 1, 0, 0, 1, 0]
    agents = {'market_ids': [0, 0, 1, 1], 'nodes': [[-0.5], [0.5], [-0.5], [0.5]],
              'weights': [0.5, 0.5, 0.5, 0.5]}
    problem = pyblp.Problem(
        [pyblp.Formulation('1 + prices'), pyblp.Formulation('0 + prices')],
        products, agent_data=agents, epsilon_scale=epsilon,
    )
    with pytest.raises(ValueError, match="outside"):
        problem._validate_fp_type('outside')
