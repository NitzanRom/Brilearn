import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


class Model:
    """
    A model for predicting a material's flow stress curve based on a set of simple brinell indentations.
    The model utilized two submodels - an XGBoost model for predicting the yield stress, and an in-house model
    for predicting the hardening curve.
    """

    def __init__(self):
        """Initialize yield and hardening models"""
        self.yld = self.__YieldSubmodel()
        self.hardening = self.__HardeningSubmodel()

    def load_model(self):
        self.yld.load()
        self.hardening.load()

    def save_model(self):
        self.yld.save()
        self.hardening.save()

    def predict(self, X):
        """
        Predict the flow stress curve of a material or a set of materials from a given database.

        Parameters
        ----------
        X : DataFrame
            Data to predict with.
            The columns of X are the strain vector used for the prediction.
            The rows are the materials to predict.

        Returns
        -------
        Y_pred : DataFrame
            Predicted result
        """
        # Checking if needed columns exist
        if ("k" not in X.columns) and ("m" not in X.columns):
            raise NameError(
                """No meyer coefficients have been identified.
                Make sure to use utilities.meyer_coeffs on the dataframe before making a prediction."""
            )

        if (self.hardening.model is None) or (self.yld.model is None):
            raise ValueError(
                """Yield or hardening model were not initiallized. Make sure to either fit the models
                             using a training database, or load an existing model."""
            )

        # Calling the yield and hardening submodels to make a prediction
        yld_prediction = self.yld.predict(X[["k", "m", "Hardness"]])
        hardening_prediction = self.hardening.predict(X[["k", "m", "Hardness"]])

        return pd.merge(
            left=yld_prediction,
            right=hardening_prediction,
            left_index=True,
            right_index=True,
        )

    def fit(self, X_train, Y_train, strains, X_val=None, Y_val=None, cv=5):
        """
        Fit both yield and hardening models on an existing database. Refer to the readme file to see how the database
        should be constructed.

        Parameters
        ----------
        X_train : DataFrame of shape {n_samples, 3}
            Training data. It should have three features: 'k', 'm' and 'Hardness',
            refering to the meyer coefficients and the Brinell Hardness
        Y_train : DataFrame of shape {n_samples, 3}
            Validation target values. It should have three features: 'Y', 'K' and 'n',
            refering to the Ludwig coefficients
        strains : 1D numpy array
            The strain vector based on which model fitting occurs.
        X_val : DataFrame of shape {n_samples, 3} : Optional
            Validation data. Has the same features as the training data.
        Y_val : DataFrame of shape {n_samples, 3} : Optional
            Validation target values. Has the same features as the validation target values.
        cv : int : Optional
            Number of subsets for K-fold cross-validation. Default value is 5.
            Irrelevant if no validation data is supplied.

        Returns
        -------
        None
        """
        # Checking if needed columns exist
        for X in [X_train, X_val]:
            if set(["k", "m", "Hardness"]) != set(X.columns):
                raise NameError("X dataframes have wrong column names.")
        for Y in [Y_train, Y_val]:
            if set(["Y", "K", "n"]) != set(Y.columns):
                raise NameError("Y dataframes have wrong column names.")

        # Calling the yield and hardening submodels to make a prediction
        self.yld.fit(X_train, Y_train, X_val, Y_val, cv)
        self.hardening.fit(X_train, Y_train, strains)

    class __HardeningSubmodel:
        def __init__(self):
            self.model = None
            self.strains = None

        def load(self):
            self.model = pd.read_csv("hardening_parameters.csv", index_col="strain")
            self.strains = self.model.index.astype(float)

        def save(self):
            self.model.to_csv("hardening_parameters.csv")

        def fit(self, X_train, Y_train, strains):
            # Defining the indentation strain vector
            strains_ind = np.linspace(0.1, 2, 50)

            # Pre-allocating the dataframe which will hold the fitted model.
            self.model = pd.DataFrame(
                index=strains, columns=["strain_ind", "a0", "a1", "a2"]
            )
            self.model.index.name = "strain"

            self.strains = strains

            # This loop searches for the true_strain-indentation_strain pairs which generate
            # the best true_stress-indentation_stress correlation for all inspected materials.
            for e_true in strains:
                # Set up the true stress vector
                s_true = (
                    Y_train.loc[:, "Y"]
                    + Y_train.loc[:, "K"] * e_true ** Y_train.loc[:, "n"]
                )
                prev_res = np.inf
                for e_ind in strains_ind:
                    # Set up the indentation stress vector
                    s_ind = X_train.loc[:, "k"] * e_ind ** X_train.loc[:, "m"]
                    # Find the 2nd polynomial fit for the two vectors
                    coeffs, res, *_ = np.polyfit(s_ind, s_true, 2, full=True)
                    # Check if the residual is lower than the previously found match
                    if res < prev_res:
                        prev_res = res
                    else:
                        # Store the best-fit (e_true, e_ind) pair.
                        self.model.loc[e_true, "strain_ind"] = e_ind
                        self.model.loc[e_true, ["a0", "a1", "a2"]] = coeffs
                        break

        def predict(self, X):
            def row_prediction(row):
                # A function which generates the best-fit s_true curve given an s_ind curve
                s_ind = row["k"] * e_ind ** row["m"]
                return coeffs[:, 0] * s_ind**2 + coeffs[:, 1] * s_ind + coeffs[:, 2]

            # Call for the strain vector and the model parameters needed for the prediction
            strains = self.model.index
            e_ind = self.model.loc[strains, "strain_ind"]
            coeffs = np.array(self.model.loc[strains, ["a0", "a1", "a2"]])

            # Construct the Y_pred DataFrame
            Y_pred = pd.DataFrame(
                data=X.apply(row_prediction, axis=1),  # Apply the prediction function
                index=X.index,
                columns=[strain for strain in strains],
            )
            return Y_pred

    class __YieldSubmodel:
        def __init__(self):
            self.model = None

        def load(self):
            """Load yield model"""
            self.model = xgb.XGBRegressor()
            self.model.load_model("yield_hyperparameters.json")

        def save(self):
            self.model.save_model("yield_hyperparameters.json")

        def fit(self, X_train, Y_train, X_val, Y_val, cv):
            # If no validation data given, do a simple model fit using the xgboost fit function
            if (X_val is None) and (Y_val is None):
                self.model.fit(X_train, Y_train)

            # If validation data is given, perform a K-Fold cross validation.
            else:
                # The hyperparameters grid to be checked
                param_grid = {
                    "n_estimators": [10000],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "max_depth": [3, 7, 10],
                    "min_child_weight": [0, 5, 10],
                    "subsample": [0.5, 0.7, 1.0],
                    "colsample_bytree": [0.5, 1.0],
                    "gamma": [0, 5, 10],
                    "reg_alpha": [0.2, 0.8],
                    "reg_lambda": [0.2, 0.8],
                    "eval_metric": ["rmse"],
                    "early_stopping_rounds": [5],
                }

                # Call the cross-validation class from the xgboost library.
                random_search = GridSearchCV(
                    xgb.XGBRegressor(),
                    param_grid=param_grid,
                    cv=cv,
                    scoring="neg_mean_squared_error",
                    n_jobs=-1,
                    verbose=1,
                    refit=True,
                )
                random_search.fit(X_train, Y_train["Y"], eval_set=[(X_val, Y_val["Y"])])

                # Store the best estimated model.
                self.model = random_search.best_estimator_

        def predict(self, X):
            Y_pred = self.model.predict(X[["k", "m", "Hardness"]])
            return pd.DataFrame(data=Y_pred, index=X.index, columns=[0])
