import numpy as np
from sklearn.model_selection import cross_val_predict
import pandas as pd
from models.Meta_Learners.meta_learner_models import R_learner_model

def r_fit(data, treatment_col, outcome_col, covariate_cols):
    X = data[covariate_cols]
    T = data[treatment_col]
    y = data[outcome_col]

    # Useing Ridge regression for residual modeling with best parameters
    y_model = cross_val_predict(R_learner_model, X, y, cv=5)
    t_model = cross_val_predict(R_learner_model, X, T, cv=5)

    # Calculate residuals
    y_residual = y - y_model
    t_residual = T - t_model

    # Regularizing treatment residuals to avoid division by zero
    epsilon = 1e-3 * np.std(t_residual)
    t_residual_regularized = np.clip(t_residual, a_min=epsilon, a_max=None)

    # Fitting the Ridge model on residuals for CATE estimation
    tau_model = R_learner_model
    tau_model.fit(X, y_residual / t_residual_regularized)

    # Returning the tau model
    return tau_model

def predict_outcomes_r(X, tau_model):
    # Predicting the treatment effect on the test data
    tau_pred = tau_model.predict(X)
    return pd.DataFrame({'tau_pred': tau_pred})

def estimate_CATE_r(predictions_df):
    return predictions_df['tau_pred']
