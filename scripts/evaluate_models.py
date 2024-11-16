from sklearn.metrics import mean_squared_error
from sklearn.utils import resample
import numpy as np

# Evaluation Metric Functions
def compute_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared= False )


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