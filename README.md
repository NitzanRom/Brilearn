# Brilearn
A machine learning model for predicting mechanical behavior of metals using Brinell force-trace diameter indentation measurements.
This model is consisted of two submodels - an XGBoost regression model for predicting the material's yield stress, and a collection of residual-minimizing-polynomials for predicting the material's hardening curve.
The theory behind this model is outlayed in a paper about to be published.

An example for how the model works is attached to this repository.

When feeding the model with measurements to be used for prediction, you need to provide normalized force-normalized trace diameter in the following form:

f = p / D^2

e = d / D

Where p is the indentation load, d is the trace diameter, and D is the diameter of the indenter.

Also make sure that the dataframe feeded to the model has the normalized force values as the dataframe's columns. Refer to the 'prediction_data.xlsx' file for further clarification.
