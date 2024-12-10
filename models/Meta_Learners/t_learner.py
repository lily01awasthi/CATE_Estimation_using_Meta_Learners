from models.Meta_Learners.meta_learner_models import T_learner_model
import pandas as pd

def t_fit(data, treatment_col, outcome_col, covariate_cols):
    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]

    X_treated = treated_data[covariate_cols]
    y_treated = treated_data[outcome_col]
    X_control = control_data[covariate_cols]
    y_control = control_data[outcome_col]

    T_learner_model_control, T_learner_model_treated = T_learner_model
    
    # T-Learner: AdaBoost
    model_control = T_learner_model_control
    model_treated = T_learner_model_treated

    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)

    return model_treated, model_control

def predict_outcomes_t(X, model_treated, model_control):
    pred_treated = model_treated.predict(X)
    pred_control = model_control.predict(X)

    return pd.DataFrame({'pred_treated': pred_treated, 'pred_control': pred_control})

def estimate_CATE_t(predictions_df):
    return predictions_df['pred_treated'] - predictions_df['pred_control']
