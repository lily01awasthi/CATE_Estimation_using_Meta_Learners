import pandas as pd
from sklearn.linear_model import LassoCV, Ridge
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import cross_val_predict

def r_fit(data, treatment_col, outcome_col, covariate_cols):
    """
    Train an R-learner model to estimate the Conditional Average Treatment Effect (CATE).

    Parameters:
    - data: DataFrame, the preprocessed dataset
    - treatment_col: str, name of the treatment column
    - outcome_col: str, name of the outcome column
    - covariate_cols: list of str, names of the covariate columns

    Returns:
    - tau_model: trained model for estimating the treatment effect
    """
    model = Ridge(alpha=10.0)
    X = data[covariate_cols]
    T = data[treatment_col]
    y = data[outcome_col]

    # Fit outcome model
    y_model = cross_val_predict(model, X, y, cv=5)

    # Fit treatment model
    t_model = cross_val_predict(model, X, T, cv=5)

    # Calculate residuals
    y_residual = y - y_model
    t_residual = T - t_model

    # Regularization of residuals
    t_residual_clipped = np.clip(t_residual, a_min=0.001, a_max=None)
    y_residual = (y_residual - np.mean(y_residual)) / np.std(y_residual)
    t_residual_clipped = (t_residual_clipped - np.mean(t_residual_clipped)) / np.std(t_residual_clipped)

    # Add a small constant to prevent division by zero
    epsilon = 1e-3 * np.std(t_residual)
    t_residual_regularized = t_residual_clipped + epsilon

    #     # Use RidgeCV or LassoCV for additional regularization in the final model
    #     regularization = 'ridge'
    #     tau_model = RidgeCV(alphas=[0.1, 1.0, 10.0])
    regularization = 'lasso'
    tau_model = Ridge(alpha=0.1)
    # Fit the treatment effect model
    division_result = y_residual / t_residual_regularized
    print(f"division result : {division_result.describe()}")

    tau_model.fit(X, y_residual / t_residual_regularized)
    print(f"tau model coefficient :{tau_model.coef_}")
    # print(f"Best alpha chosen by cross-validation: {tau_model.alpha_}")

    return tau_model, y_model, t_model, y_residual,t_residual

def predict_outcomes_r(X, tau_model, y_model, t_model):
    """
    Predict potential outcomes for both treatment and control groups.

    Parameters:
    - X: pd.DataFrame, feature matrix
    - tau_model: trained model for estimating the treatment effect
    - y_model: trained model for predicting the outcome
    - t_model: trained model for predicting the treatment

    Returns:
    - pd.Series with treatment effect estimates.
    """
    # Predict residuals
    y_residual = y_model
    t_residual = t_model

    # Predict treatment effect
    tau_pred = tau_model.predict(X)

    return pd.DataFrame({'tau_pred': tau_pred, 'y_residual': y_residual, 't_residual': t_residual})

def estimate_CATE_r(df):
    """
    Estimate the Conditional Average Treatment Effect (CATE) using R-learner.

    Parameters:
    - df: pd.DataFrame, dataframe containing predictions for treatment effect and residuals

    Returns:
    - pd.Series with CATE estimates.
    """
    return df['tau_pred']