"""Standard statistical routines."""

from typing import List, Optional, Tuple

import numpy as np
import scipy.linalg

from .algebra import approximately_invert, duplication_matrix, elimination_matrix
from .basics import Array, Error, Groups
from .. import exceptions


def _align_iv_rows(matrices: List[Array], row_counts: List[int]) -> List[Array]:
    """Align stacked design rows to instrument blocks without block diagonal matrices."""
    if not matrices:
        raise ValueError("At least one IV design block is required.")
    columns = matrices[0].shape[1]
    if any(x.ndim != 2 or x.shape[1] != columns for x in matrices):
        raise ValueError("IV design blocks must have the same number of columns.")
    if any(n < 0 for n in row_counts) or sum(x.shape[0] for x in matrices) != sum(row_counts):
        raise ValueError("IV design and instrument blocks must have identical total row counts.")
    blocks: List[Array] = []
    index = offset = 0
    for count in row_counts:
        remaining = count
        pieces: List[Array] = []
        while remaining:
            while index < len(matrices) and offset == matrices[index].shape[0]:
                index += 1
                offset = 0
            take = min(remaining, matrices[index].shape[0] - offset)
            pieces.append(matrices[index][offset:offset + take])
            offset += take
            remaining -= take
        blocks.append(matrices[0][:0] if not pieces else pieces[0] if len(pieces) == 1 else np.vstack(pieces))
    return blocks


class IV(object):
    """Generalized instrumental variables with cached blockwise projections.

    The caller must create a new instance whenever X, Z, or W changes.
    """

    covariances: Array
    errors: List[Error]

    def __init__(self, X_list: List[Array], Z_list: List[Array], W: Array) -> None:
        X_blocks = _align_iv_rows(X_list, [z.shape[0] for z in Z_list])
        product = np.vstack([z.T @ x for z, x in zip(Z_list, X_blocks)])
        covariances_inverse = product.T @ W @ product
        self.covariances, replacement = approximately_invert(covariances_inverse)
        self._projection = self.covariances @ product.T @ W
        self.errors = []
        if replacement:
            self.errors.append(exceptions.LinearParameterCovariancesInversionError(covariances_inverse, replacement))

    def estimate(
        self,
        X_list: List[Array],
        Z_list: List[Array],
        W: Array,
        y_list: List[Array],
        jacobian_list: List[Array],
        convert_jacobians: bool,
        fixed_parameters: Optional[Array] = None,
    ) -> Tuple[Array, List[Array], List[Array]]:
        """Return IV estimates, residuals, and optionally converted Jacobians."""
        if not (len(Z_list) == len(y_list) == len(jacobian_list)):
            raise ValueError("IV instrument, outcome, and Jacobian partitions must match.")
        if any(z.shape[0] != y.shape[0] for z, y in zip(Z_list, y_list)):
            raise ValueError("IV instrument and outcome block lengths differ.")
        X_blocks = _align_iv_rows(X_list, [y.shape[0] for y in y_list])
        if fixed_parameters is None:
            parameters = self._projection @ np.vstack([z.T @ y for z, y in zip(Z_list, y_list)])
        else:
            if convert_jacobians:
                raise ValueError("Fixed IV coefficients cannot be used for Jacobian conversion.")
            parameters = np.asarray(fixed_parameters)
            if parameters.shape != (self._projection.shape[0], y_list[0].shape[1]):
                raise ValueError("Fixed IV coefficients have an unexpected shape.")
            parameters = parameters.copy()
        residuals_list = [y - x @ parameters for x, y in zip(X_blocks, y_list)]
        if convert_jacobians:
            with np.errstate(under='ignore'):
                coefficient_jacobian = self._projection @ np.vstack([
                    z.T @ jac for z, jac in zip(Z_list, jacobian_list)
                ])
                jacobian_list = [
                    jac - x @ coefficient_jacobian for x, jac in zip(X_blocks, jacobian_list)
                ]
        return parameters, residuals_list, jacobian_list

def compute_gmm_weights(S: Array) -> Tuple[Array, List[Error]]:
    """Compute a GMM weighting matrix."""
    errors: List[Error] = []

    # invert the matrix and handle any errors
    W, replacement = approximately_invert(S)
    if replacement:
        errors.append(exceptions.GMMMomentCovariancesInversionError(S, replacement))
    if np.isnan(W).any():
        errors.append(exceptions.InvalidMomentCovariancesError())

    # enforce shape and symmetry
    return np.c_[W + W.T] / 2, errors


def compute_gmm_moment_covariances(
    u_list: List[Array],
    Z_list: List[Array],
    covariance_type: str,
    clustering_ids: Array,
    center_moments: bool,
) -> Array:
    """Compute covariances between moments."""

    # count dimensions
    N = u_list[0].shape[0]

    # compute the moment covariances
    if covariance_type == 'unadjusted':
        pairs = list(zip(u_list, Z_list))
        S = np.block([[compute_gmm_error_covariance(u1, u2) * (Z1.T @ Z2) for u2, Z2 in pairs] for u1, Z1 in pairs]) / N
    else:
        g = compute_gmm_moments(u_list, Z_list)
        if center_moments:
            g -= g.mean(axis=0)
        if covariance_type == 'clustered':
            g = Groups(clustering_ids).sum(g)
        S = g.T @ g / N

    # enforce shape and symmetry
    return np.c_[S + S.T] / 2


def compute_gmm_parameter_covariances(W: Array, S: Array, mean_G: Array, se_type: str) -> Tuple[Array, List[Error]]:
    """Estimate GMM parameter covariances."""
    errors: List[Error] = []

    # attempt to compute the covariance matrix
    covariances_inverse = mean_G.T @ W @ mean_G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the robust covariance matrix
    if se_type != 'unadjusted':
        with np.errstate(invalid='ignore'):
            covariances = covariances @ mean_G.T @ W @ S @ W @ mean_G @ covariances

    # enforce shape and symmetry
    return np.c_[covariances + covariances.T] / 2, errors


def compute_gmm_parameter_sensitivity(W: Array, mean_G: Array) -> Tuple[Array, List[Error]]:
    """Compute the sensitivity measure of Andrews, Gentzkow, and Shapiro (2017)."""
    errors: List[Error] = []

    # attempt to compute the covariance matrix
    covariances_inverse = mean_G.T @ W @ mean_G
    covariances, replacement = approximately_invert(covariances_inverse)
    if replacement:
        errors.append(exceptions.GMMParameterCovariancesInversionError(covariances_inverse, replacement))

    # compute the sensitivity measure
    sensitivity = -covariances @ mean_G.T @ W
    return sensitivity, errors


def compute_gmm_error_covariance(u1: Array, u2: Array) -> Array:
    """Compute the covariance between two error terms."""
    return np.cov(u1.flatten(), u2.flatten(), bias=True)[0][1]


def compute_gmm_moments(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Compute GMM moments."""
    return np.hstack([u * Z for u, Z in zip(u_list, Z_list)])


def compute_gmm_moments_mean(u_list: List[Array], Z_list: List[Array]) -> Array:
    """Average moment blocks without allocating an N by M elementwise product."""
    return np.vstack([Z.T @ u / u.shape[0] for u, Z in zip(u_list, Z_list)])


def compute_gmm_moments_jacobian_mean(jacobian_list: List[Array], Z_list: List[Array]) -> Array:
    """Average moment Jacobian blocks without an N by M by P temporary tensor."""
    return np.vstack([Z.T @ jac / jac.shape[0] for jac, Z in zip(jacobian_list, Z_list)])


def compute_sigma_squared_vector_covariances(sigma: Array, sigma_vector_covariances: Array) -> Array:
    """Use the delta method to transform the asymptotic covariance matrix of vech(sigma) into the asymptotic covariance
    matrix of vech(sigma * sigma') where sigma is a lower triangular Cholesky root of parameters. See Section 10.5.4 in
    the Handbook of Matrices (Lutkepohl and Lutkepohl).
    """
    if sigma_vector_covariances.size == 0:
        return sigma_vector_covariances

    k = sigma.shape[0]
    L = elimination_matrix(k)
    D = duplication_matrix(k)
    I = np.eye(k)
    jacobian = 2 * scipy.linalg.pinv(D) @ np.kron(sigma, I) @ L.T
    return jacobian @ sigma_vector_covariances @ jacobian.T
