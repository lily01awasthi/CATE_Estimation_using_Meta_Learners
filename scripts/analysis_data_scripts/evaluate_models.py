from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import numpy as np
import pandas as pd
# Evaluation Metric Functions
def compute_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared= False )


def bootstrap_emse_no_groundtruth(X_train, treatment_train, model, meta_learner, n_bootstraps=10):
    """
    Evaluate the variability of treatment effect estimates using bootstrap EMSE.
    
    Parameters:
    - X_train: np.ndarray, covariates for training.
    - treatment_train: np.ndarray, treatment assignments for training.
    - model: trained model or model pipeline.
    - meta_learner: str, meta-learner type ('S-Learner', 'T-Learner', 'X-Learner', 'R-Learner').
    - n_bootstraps: int, number of bootstrap resamples.

    Returns:
    - float, the mean bootstrap variance (variability in predictions).
    """
    cate_predictions = []

    for _ in range(n_bootstraps):
        # Resample data
        X_boot, t_boot = resample(X_train, treatment_train)

        # Estimate CATE based on meta-learner type
        if meta_learner == 'S-Learner':
            X_boot_combined = np.column_stack([X_boot, t_boot])
            model.fit(X_boot_combined, np.random.rand(len(X_boot)))  # Dummy outcomes
            y_treated = model.predict(np.column_stack([X_boot, np.ones_like(t_boot)]))
            y_control = model.predict(np.column_stack([X_boot, np.zeros_like(t_boot)]))
            cate_pred = y_treated - y_control

        elif meta_learner == 'T-Learner':
            model_control, model_treat = model

            X_treat = X_boot[t_boot == 1]
            X_control = X_boot[t_boot == 0]

            model_treat.fit(X_treat, np.random.rand(len(X_treat)))  # Dummy outcomes
            model_control.fit(X_control, np.random.rand(len(X_control)))  # Dummy outcomes

            y_treated = model_treat.predict(X_boot)
            y_control = model_control.predict(X_boot)
            cate_pred = y_treated - y_control

        elif meta_learner == 'X-Learner':
            model_control, model_treat = model[:2]

            X_treat = X_boot[t_boot == 1]
            X_control = X_boot[t_boot == 0]

            tau_control = np.random.rand(len(X_control)) - model_treat.predict(X_control)
            tau_treat = model_control.predict(X_treat) - np.random.rand(len(X_treat))

            cate_pred = np.hstack([tau_control, tau_treat])

        elif meta_learner == 'R-Learner':
            model_y = model.__class__(**model.get_params())
            model_t = model.__class__(**model.get_params())
            model_y.fit(X_boot, np.random.rand(len(X_boot)))  # Dummy outcomes
            model_t.fit(X_boot, t_boot)

            y_residual = np.random.rand(len(X_boot)) - model_y.predict(X_boot)
            t_residual = t_boot - model_t.predict(X_boot)

            model.fit(t_residual.reshape(-1, 1), y_residual)
            cate_pred = model.predict(t_residual.reshape(-1, 1))

        cate_predictions.append(cate_pred)

    # Compute variance across bootstraps for each instance
    cate_predictions = np.array(cate_predictions)  # Shape: (n_bootstraps, n_samples)
    variance_per_instance = np.var(cate_predictions, axis=0)  # Variance per sample

    # Return the mean variance as a summary metric
    return np.mean(variance_per_instance)


def bootstrap_emse(X_train, y_train, treatment_train, model, meta_learner, n_bootstraps=10):
    mse_list = []
    for _ in range(n_bootstraps):
        X_boot, y_boot, t_boot = resample(X_train, y_train, treatment_train)
        
        if meta_learner == 'S-Learner':
            X_boot_combined = np.column_stack([X_boot, t_boot])
            model.fit(X_boot_combined, y_boot)
            cate_pred = model.predict(X_boot_combined)

        elif meta_learner == 'T-Learner':
            # Ensure model is a tuple of control and treated models
            if isinstance(model, tuple) and len(model) == 2:
                model_control, model_treat = model
            else:
                raise ValueError("Expected model to be a tuple (model_control, model_treat) for T-Learner.")

            # Create bootstrap samples for treated and control groups
            X_treat, y_treat = X_boot[t_boot == 1], y_boot[t_boot == 1]
            X_control, y_control = X_boot[t_boot == 0], y_boot[t_boot == 0]
            
            # Fit models on bootstrap samples
            model_treat.fit(X_treat, y_treat)
            model_control.fit(X_control, y_control)
            
            # Predict treatment effects using both models
            treat_pred = model_treat.predict(X_boot)
            control_pred = model_control.predict(X_boot)
            cate_pred = treat_pred - control_pred

        elif meta_learner == 'X-Learner':
            # Ensure model includes both control and treated models
            if isinstance(model, tuple) and len(model) >= 2:
                model_control, model_treat = model[:2]  # Use first two models
            else:
                raise ValueError("Expected model to include (model_control, model_treat) for X-Learner.")

            # Bootstrap samples for treated and control groups
            X_treat, y_treat = X_boot[t_boot == 1], y_boot[t_boot == 1]
            X_control, y_control = X_boot[t_boot == 0], y_boot[t_boot == 0]
            
            # Fit models on bootstrap samples
            model_treat.fit(X_treat, y_treat)
            model_control.fit(X_control, y_control)
            
            # Estimate treatment effects
            tau_control = y_control - model_treat.predict(X_control)  # Counterfactual for control
            tau_treat = model_control.predict(X_treat) - y_treat      # Counterfactual for treated
            
            # Combine treatment effects to form the CATE prediction
            cate_pred = np.hstack([tau_control, tau_treat])

        elif meta_learner == 'R-Learner':
            model_y = model.__class__(**model.get_params())
            model_t = model.__class__(**model.get_params())
            model_y.fit(X_boot, y_boot)
            model_t.fit(X_boot, t_boot)
            
            y_residual = y_boot - model_y.predict(X_boot)
            t_residual = t_boot - model_t.predict(X_boot)
            
            model.fit(t_residual.reshape(-1, 1), y_residual)
            cate_pred = model.predict(t_residual.reshape(-1, 1))
        
        mse_boot = mean_squared_error(y_boot, cate_pred, squared=False)
        mse_list.append(mse_boot)
    
    return np.mean(mse_list)

def evaluate_meta_learner(predictions, y_test, treatment_test, learner_name):
    """
    Evaluate predictions of a meta-learner by comparing predicted outcomes with actual outcomes.

    Parameters:
    - predictions: pd.DataFrame, predicted outcomes for treated and control groups
    - y_test: np.array, actual observed outcomes
    - treatment_test: np.array, treatment assignments in the test set
    - learner_name: str, name of the meta-learner (e.g., 'S-Learner')

    Returns:
    - metrics: dict, containing RMSE, bias, and variance
    """
    # Select predicted outcomes based on treatment assignment
    predicted_outcomes = np.where(
        treatment_test == 1,
        predictions['pred_treated'],  # Use treated predictions for treated individuals
        predictions['pred_control']   # Use control predictions for control individuals
    )
    
    # Compute evaluation metrics rmse, bias, and variance
    rmse = np.sqrt(mean_squared_error(y_test, predicted_outcomes))
    bias = np.mean(predicted_outcomes - y_test)
    variance = np.var(predicted_outcomes)

    # Store metrics in a dictionary
    metrics = {
        f'{learner_name}_RMSE': rmse,
        f'{learner_name}_Bias': bias,
        f'{learner_name}_Variance': variance
    }
    
    return metrics


