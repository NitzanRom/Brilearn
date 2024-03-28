import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def re(y1, y2):
    """
    Calculates the average relative error between two graphs, 
    for every graph in a DataFrame of n_samples graphs.

    Parameters
    ----------
    y1, y2 : DataFrame - shape of {n_samples, n_points}

    Results
    ----------
    1D array - shape of {n_samples,}
    """
    if len(y1.shape) == 1 or y1.shape[1] == 1:
        return 100 * abs(y1.values - y2.values) / y1.values
    return np.mean(100 * abs(y1.values - y2.values) / y1.values, axis=1)


def rmse(y1, y2):
    """
    Calculates the root mean squared error between two graphs, 
    for every graph in a DataFrame of n_samples graphs.

    Parameters
    ----------
    y1, y2 : DataFrame - shape of {n_samples, n_points}

    Results
    ----------
    1D array - shape of {n_samples,}
    """
    if len(y1.shape) == 1 or y1.shape[1] == 1:
        return np.abs(y1.values - y2.values)
    return np.sqrt(np.mean((y1.values - y2.values) ** 2, axis=1))


def get_errors(Y_pred, Y_true):
    """
    Calculates relative error and root mean squared error between predicted and true graphs, and plots
    a histogram of those errors for both the yield stress and the entire graph.

    Parameters
    ----------
    Y_pred : DataFrame of shape {n_samples, strain vector}
        A DataFrame of predicted materials' graphs. The columns are the strain vector of the graphs.
    Y_true : DataFrame of shape {n_samples, strain vector}
        A DataFrame of materials' actual graphs. The columns are the strain vector of the predicted graphs.

    Returns
    -------
    errors : DataFrame
        relative error and root mean squared error of both the yield stress and the entire graph,
        for all inspected materials.
    """
    if not np.array_equal(Y_pred.columns, Y_true.columns):
        raise ValueError(
            "Make sure the strain vector is the same for both Y_pred and Y_true"
        )

    re_vec = re(Y_true, Y_pred)
    rmse_vec = rmse(Y_true, Y_pred)
    yield_re_vec = re(Y_pred.iloc[:, 0], Y_true.iloc[:, 0])
    yield_rmse_vec = rmse(Y_pred.iloc[:, 0], Y_true.iloc[:, 0])

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), dpi=300)
    labels = [
        "Relative error [%]",
        "Mean Squared error [MPa]",
        "Yield Relative error [%]",
        "Yield Absolute error [MPa]",
    ]
    data_sets = [re_vec, rmse_vec, yield_re_vec, yield_rmse_vec]
    units = 2 * ["%", "MPa"]

    for ax, data, label, unit in zip(axes.flatten(), data_sets, labels, units):
        n, _, patches = ax.hist(
            data,
            facecolor="#2ab0ff",
            weights=100 * np.ones(len(data)) / len(data),
            edgecolor="#e0e0e0",
            linewidth=0.5,
            alpha=0.7,
        )
        n = n.astype("int")
        for i in range(len(patches)):
            patches[i].set_facecolor(plt.cm.Blues(0.4 + 0.6 * n[i] / max(n)))
        ax.set_xlabel(label)
        ax.set_ylabel("(%) of dataset")
        ax.grid(axis="y")
        ax.set_title(f"Error - {np.mean(data):.2f} ± {np.std(data):.2f} {unit}")
    fig.tight_layout()

    errors = pd.DataFrame(
        data=np.array((re_vec, rmse_vec, yield_re_vec, yield_rmse_vec)).T,
        columns=["RE", "MSE", "RE_Yield", "MSE_Yield"],
        index=Y_true.index,
    )

    return errors
