import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


def x_fit(data, treatment_col, outcome_col, covariate_cols):
    """
    Train an X-learner model to estimate the Conditional Average Treatment Effect (CATE).

    Parameters:
    - data: DataFrame, the preprocessed dataset
    - treatment_col: str, name of the treatment column
    - outcome_col: str, name of the outcome column
    - covariate_cols: list of str, names of the covariate columns

    Returns:
    - cate_estimates: DataFrame, containing the CATE estimates for each instance
    """
    # Split the data into treatment and control groups
    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]

    # Create the covariates matrix and the outcome vector for both groups
    X_treated = treated_data[covariate_cols]
    y_treated = treated_data[outcome_col]
    X_control = control_data[covariate_cols]
    y_control = control_data[outcome_col]

    # Train separate models on the treated and control groups
    model_treated = Ridge(alpha=10.0, random_state=42)
    model_control = Ridge(alpha=10.0, random_state=42)

    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)

    return model_treated, model_control


def predict_outcomes_x(X, model_treated, model_control):
    """
    Predict potential outcomes for both treatment and control groups.

    Parameters:
    - X: pd.DataFrame, feature matrix excluding the treatment variable
    - model_treated: trained model for treated data
    - model_control: trained model for control data

    Returns:
    - pd.DataFrame with columns 'pred_treated' and 'pred_control'.
    """
    pred_treated = model_treated.predict(X)
    pred_control = model_control.predict(X)

    return pd.DataFrame({'pred_treated': pred_treated, 'pred_control': pred_control})


def estimate_CATE_x(df):
    """
    Estimate the Conditional Average Treatment Effect (CATE) using X-learner.

    Parameters:
    - df: pd.DataFrame, dataframe containing predictions for treated and control groups

    Returns:
    - pd.Series with CATE estimates.
    """
    return df['pred_treated'] - df['pred_control']
