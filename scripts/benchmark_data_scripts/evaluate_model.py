
import numpy as np
from scripts.analysis_data_scripts.evaluate_models import compute_rmse, bootstrap_emse

# Function to compute RMSE and EMSE
def evaluate_model(X_train, y_train, treatment_train, model, meta_learner):
    """
    Evaluate a given model using RMSE and bootstrap EMSE.
    """
    # Compute RMSE for each meta-learner
    if meta_learner == 'S-Learner':
        X_combined = np.column_stack([X_train, treatment_train])
        y_pred = model.predict(X_combined)
    elif meta_learner == 'T-Learner':
        model_control, model_treat = model
        y_pred = model_treat.predict(X_train) - model_control.predict(X_train)
    elif meta_learner == 'X-Learner':
        model_control, model_treat = model[:2]  # First two models
        tau_control = y_train[treatment_train == 0] - model_treat.predict(X_train[treatment_train == 0])
        tau_treat = model_control.predict(X_train[treatment_train == 1]) - y_train[treatment_train == 1]
        y_pred = np.hstack([tau_control, tau_treat])
    elif meta_learner == 'R-Learner':
        y_residual = y_train - model.predict(X_train)
        y_pred = model.predict(X_train)
    else:
        raise ValueError(f"Unknown meta-learner: {meta_learner}")

    rmse = compute_rmse(y_train, y_pred)

    # Compute EMSE using bootstrapping
    emse = bootstrap_emse(X_train, y_train, treatment_train, model, meta_learner)

    return rmse, emse