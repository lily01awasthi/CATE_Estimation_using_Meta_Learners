from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import cross_val_predict
import pandas as pd

def r_fit(data, treatment_col, outcome_col, covariate_cols):
    """
    Train an R-learner model to estimate the Conditional Average Treatment Effect (CATE) using RandomForest with specified parameters.
    
    Parameters:
    - data: DataFrame, the preprocessed dataset
    - treatment_col: str, name of the treatment column
    - outcome_col: str, name of the outcome column
    - covariate_cols: list of str, names of the covariate columns
    
    Returns:
    - tau_model: trained RandomForest model for estimating the treatment effect
    """
    X = data[covariate_cols]
    T = data[treatment_col]
    y = data[outcome_col]

    # Use RandomForest with best parameters to calculate residuals (instead of Ridge)
    residual_model = RandomForestRegressor(max_depth=10, max_features=None, min_samples_split=10, n_estimators=500, random_state=42)
    y_model = cross_val_predict(residual_model, X, y, cv=5)
    t_model = cross_val_predict(residual_model, X, T, cv=5)

    # Calculate residuals
    y_residual = y - y_model
    t_residual = T - t_model

    # Regularize treatment residuals to avoid division by zero
    epsilon = 1e-3 * np.std(t_residual)
    t_residual_regularized = np.clip(t_residual, a_min=epsilon, a_max=None)

    # Fit the RandomForest model on residuals for CATE estimation
    tau_model = RandomForestRegressor(
        max_depth=10,
        max_features=None,
        min_samples_split=10,
        n_estimators=500,
        random_state=42
    )
    tau_model.fit(X, y_residual / t_residual_regularized)

    # Return the tau model
    return tau_model

def predict_outcomes_r(X, tau_model):
    """
    Predict the treatment effect using the R-Learner's tau model.
    
    Parameters:
    - X: pd.DataFrame, feature matrix for the test data
    - tau_model: trained RandomForest model for estimating the treatment effect
    
    Returns:
    - pd.DataFrame containing 'tau_pred' predictions for the test data
    """
    # Predict the treatment effect on the test data
    tau_pred = tau_model.predict(X)
    return pd.DataFrame({'tau_pred': tau_pred})

def estimate_CATE_r(predictions_df):
    """
    Estimate the Conditional Average Treatment Effect (CATE) using R-learner.
    
    Parameters:
    - predictions_df: pd.DataFrame, dataframe containing predictions for treatment effect
    
    Returns:
    - pd.Series with CATE estimates
    """
    return predictions_df['tau_pred']
