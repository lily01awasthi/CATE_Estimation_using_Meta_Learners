import pandas as pd
from models.Meta_Learners.meta_learner_models import X_learner_model
from sklearn.ensemble import AdaBoostRegressor

def x_fit(data, treatment_col, outcome_col, covariate_cols):
    
    # Split the data into treatment and control groups
    treated_data = data[data[treatment_col] == 1]
    control_data = data[data[treatment_col] == 0]

    # Create the covariates matrix and the outcome vector for both groups
    X_treated = treated_data[covariate_cols]
    y_treated = treated_data[outcome_col]
    X_control = control_data[covariate_cols]
    y_control = control_data[outcome_col]

    X_learner_model_control,X_learner_model_treated = X_learner_model

    # Train separate models on the treated and control groups
    model_treated = X_learner_model_treated
    model_control = X_learner_model_control


    model_treated.fit(X_treated, y_treated)
    model_control.fit(X_control, y_control)

    # Estimate pseudo outcomes for CATE model training
    tau_control = y_control - model_treated.predict(X_control)
    tau_treated = model_control.predict(X_treated) - y_treated

    # Combine pseudo outcomes for final CATE model training
    X_combined = pd.concat([X_control, X_treated])
    tau_combined = pd.concat([pd.Series(tau_control, index=control_data.index),
                              pd.Series(tau_treated, index=treated_data.index)])

    # Train CATE model
    cate_model = AdaBoostRegressor(learning_rate=0.01,n_estimators=50,random_state=42)
    cate_model.fit(X_combined, tau_combined)

    return model_treated, model_control, cate_model

def predict_outcomes_x(X, model_treated, model_control, cate_model):
    
    pred_treated = model_treated.predict(X)
    pred_control = model_control.predict(X)
    cate_estimates = cate_model.predict(X)

    return pd.DataFrame({
        'pred_treated': pred_treated,
        'pred_control': pred_control,
        'cate_estimates': cate_estimates
    })

def estimate_CATE_x(df):
    return df['cate_estimates']

