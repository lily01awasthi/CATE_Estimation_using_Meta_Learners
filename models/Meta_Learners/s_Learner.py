"""
S-learner can help determine how different educational treatments affect various student outcomes, 
considering both student and school-level characteristics.The S-learner approach integrates the treatment
information directly into the feature space of a single predictive model.
"""
"""
STEPS:
1. Combine Treatment and covariates::Assuming Dataset D, with features(covariates) X, treatment variable T and outcome variable Y. Here in our dataset
X- student level features(demographics and achievements) and school-level features (school characteristics).

2. Augment Feature Space:: Treatment variable T is included as an additional feature in  the feature set X, so feature set becomes X^ = [X,T]

3. Train a Predictive Model:: Train a single predictive model f(e.g. a regression model, random forest or neural network) to predict the outcome Y,
using the augmented feature set X^. 

4. Estimate CATE:: for each individual i, you estimate the potential outcomes under different treatments. Predict the outcomes for individual i with 
treatment T=t(eg T=0, T=1 for binary treatment)
Y^(0)=f(Xi,0), Y^(1)=f(Xi,1). CATE for for individual i is then computed as the difference in these predicted outcomes: CATE(Xi)= f(Xi,1)-f(Xi,0)
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

"""
The S-learner uses a single model to estimate the treatment effect by including the treatment as a feature in the model.
"""


def s_fit(data, treatment_col, outcome_col, covariate_cols):
    """
    Train an S-learner model to estimate the Conditional Average Treatment Effect (CATE).

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

    # Create the covariates matrix and the outcome vector
    X = data[covariate_cols + [treatment_col]]
    Y = data[outcome_col]

    # Train a model on the combined data
    model = GradientBoostingRegressor(n_estimators=200, random_state=42,learning_rate= 0.1, max_depth= 3 )
    model.fit(X, Y)

    return model


def predict_outcomes(X, model, treatment_col):
    """
    Predict potential outcomes for both treatment and control groups.

    Parameters:
    - X: pd.DataFrame, feature matrix excluding the treatment variable
    - model: trained model
    - treatment_col: str, name of the treatment column

    Returns:
    - pd.DataFrame with columns 'pred_0' and 'pred_1' for control and treatment predictions.
    """
    X_control = X.copy()
    X_control[treatment_col] = 0
    pred_0 = model.predict(X_control)

    X_treatment = X.copy()
    X_treatment[treatment_col] = 1
    pred_1 = model.predict(X_treatment)

    return pd.DataFrame({'pred_0': pred_0, 'pred_1': pred_1})


def estimate_CATE(df):
    """
    Estimate the Conditional Average Treatment Effect (CATE).

    Parameters:
    - X: pd.DataFrame, feature matrix excluding the treatment variable
    - model: trained model
    - treatment_col: str, name of the treatment column

    Returns:
    - pd.Series with CATE estimates.
    """

    return df['pred_1'] - df['pred_0']
