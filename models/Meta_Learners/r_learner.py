import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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
    # Create the covariates matrix and the outcome vector
    X = data[covariate_cols]
    T = data[treatment_col]
    y = data[outcome_col]

    # Step 1: Fit a model to predict the outcome using covariates
    y_model = RandomForestRegressor(n_estimators=100, random_state=42)
    y_model.fit(X, y)

    # Step 2: Fit a model to predict the treatment using covariates
    t_model = RandomForestRegressor(n_estimators=100, random_state=42)
    t_model.fit(X, T)

    # Compute residuals
    y_residual = y - y_model.predict(X)
    t_residual = T - t_model.predict(X)

    # Clip t_residual to avoid extremely small values
    t_residual_clipped = t_residual.clip(lower=0.01)


    # Step 3: Fit a model on the residuals to estimate the treatment effect
    tau_model = RandomForestRegressor(n_estimators=100, random_state=42)
    tau_model.fit(X, y_residual / (t_residual_clipped + 1e-10))  # Adding a small value to avoid division by zero

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
    y_residual = y_model.predict(X)
    t_residual = t_model.predict(X)

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
