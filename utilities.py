import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def meyer_coeffs(X):
    """
    Find the meyer coefficients of a set of force-diameter indentation results

    Parameters
    ----------
    X : DataFrame - shape of {n_samples, n_forces}
        Indentation diameters. The columns must be the force values.

    Returns
    -------
    X : DataFrame - shape of {n_samples, n_forces + 3}
        Returns the same input DataFrame with additional three columns - k, m, which are
        the meyer coefficients, and Rsquared, which is the coefficient of determination of
        the meyer fit.
    """

    def least_squares(x):
        y = np.log(forces)
        y = y[~np.isnan(x)]  # Disregard NaN values
        x = np.log(x[~np.isnan(x)])

        # The least squares fitting scheme
        N = (len(x) * sum(x * y) - sum(x) * sum(y)) / (
            len(x) * sum(x**2) - sum(x) ** 2
        )
        K = 1 / len(x) * (sum(y) - N * sum(x))

        # Compute R^2
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - K - N * x) ** 2)
        r_squared = 1 - ss_res / ss_total

        return np.array([np.exp(K), N, r_squared])

    forces = X.columns.drop("Hardness").astype(float)
    X[["k", "m", "Rsquared"]] = X.apply(
        lambda row: least_squares(row[forces]), axis=1, result_type="expand"
    )
    return X


def construct_graphs(Y, strains):
    """
    Constructs an array of graphs given Ludwig coefficients and strains vector.

    Parameters
    ----------
    Y : DataFrame - shape of {n_samples, 3}
        A DataFrame which contains the Ludwig coefficients of the materials - Y, K and n.
    strains : 1D array
        The strain vectors for which the graph will be constructed.

    Returns
    -------
    Y_graphs : DataFrame - shape of {n_samples, len(strains)}
        A DataFrame which contains the graphical representation of the materials.
        It's columns are the strain values.
    """
    strains_matrix = np.row_stack([strains] * len(Y.index))
    Y_graphs = pd.DataFrame(
        data=Y["Y"].values[:, np.newaxis]
        + Y["K"].values[:, np.newaxis] * strains_matrix ** Y["n"].values[:, np.newaxis],
        index=Y.index,
        columns=strains,
    )
    return Y_graphs.astype(float)
