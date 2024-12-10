import pandas as pd
from models.Meta_Learners.meta_learner_models import S_learner_model

def s_fit(data, treatment_col, outcome_col, covariate_cols):
    # Create the covariates matrix and the outcome vector
    X = data[covariate_cols + [treatment_col]]
    Y = data[outcome_col]

    model = S_learner_model  
    model.fit(X, Y)

    return model


def predict_outcomes_s(X, model, treatment_col):
    
    X_control = X.copy()
    X_control[treatment_col] = 0
    pred_0 = model.predict(X_control)

    X_treatment = X.copy()
    X_treatment[treatment_col] = 1
    pred_1 = model.predict(X_treatment)

    return pd.DataFrame({'pred_treated': pred_1, 'pred_control': pred_0})


def estimate_CATE_s(df):
    return df['pred_treated'] - df['pred_control']