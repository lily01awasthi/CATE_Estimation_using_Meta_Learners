from data.data_preprocessing import preprocessor,load_data
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import resample
import pandas as pd

def get_param_grid():
    return {
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [100, 200, 500],  # Number of trees
                'max_depth': [None, 10, 20],  # Maximum depth of the tree
                'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
                'max_features': ['sqrt', 'log2', None]  # Number of features to consider when looking for the best split
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(),
            'params': {
                'n_estimators': [100, 200, 500],  # Number of boosting stages
                'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage used in each boosting step
                'max_depth': [3, 5, 10],  # Maximum depth of individual estimators
                'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
                'subsample': [0.8, 1.0]  # Fraction of samples used for fitting the individual estimators
            }
        },
        'NeuralNetwork': {
            'model': MLPRegressor(max_iter=1000),  # Neural network regressor
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (100, 100)],  # Size of the hidden layers
                'activation': ['relu', 'tanh'],  # Activation functions
                'solver': ['adam', 'lbfgs'],  # Solver for weight optimization
                'learning_rate_init': [0.001, 0.01]  # Initial learning rate
            }
        },
        'Ridge': {
            'model': Ridge(),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]  # Regularization strength
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'kernel': ['linear', 'rbf'],  # Type of kernel
                'C': [0.1, 1.0, 10.0],  # Regularization parameter
                'epsilon': [0.1, 0.2, 0.5]  # Epsilon in the epsilon-SVR model
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(),
            'params': {
                'n_estimators': [50, 100, 200],  # Number of boosting stages
                'learning_rate': [0.01, 0.1, 0.5]  # Shrinks the contribution of each estimator
            }
        },
        'ExtraTrees': {
            'model': ExtraTreesRegressor(),
            'params': {
                'n_estimators': [100, 200, 500],  # Number of trees in the forest
                'max_depth': [None, 10, 20],  # Maximum depth of the tree
                'min_samples_split': [2, 5, 10]  # Minimum number of samples required to split a node
            }
        }
    }

def bootstrap_emse(X_train, y_train, treatment_train, model, n_bootstraps=10, meta_learner='S-Learner'):
    mse_list = []
    
    for _ in range(n_bootstraps):
        # Bootstrap resampling
        X_boot, y_boot, t_boot = resample(X_train, y_train, treatment_train)
        
        if meta_learner == 'S-Learner':
            X_boot = np.column_stack([X_boot, t_boot])
            model.fit(X_boot, y_boot)
            cate_pred = model.predict(X_boot)
        
        elif meta_learner == 'T-Learner':
            # Separate models for treatment and control groups
            X_treat, y_treat = X_boot[t_boot == 1], y_boot[t_boot == 1]
            X_control, y_control = X_boot[t_boot == 0], y_boot[t_boot == 0]
            
            model_treat = model.__class__(**model.get_params())  # New instance of the model
            model_control = model.__class__(**model.get_params())  # New instance of the model
            
            model_treat.fit(X_treat, y_treat)
            model_control.fit(X_control, y_control)
            
            # Predict for all data (X_treat + X_control) so that shapes match when calculating CATE
            treat_pred = model_treat.predict(X_boot)
            control_pred = model_control.predict(X_boot)
            cate_pred = treat_pred - control_pred

        elif meta_learner == 'X-Learner':
            # X-Learner first fits the T-Learner, then fits pseudo outcomes
            model_treat = model.__class__(**model.get_params())
            model_control = model.__class__(**model.get_params())
            X_treat, y_treat = X_boot[t_boot == 1], y_boot[t_boot == 1]
            X_control, y_control = X_boot[t_boot == 0], y_boot[t_boot == 0]
            
            model_treat.fit(X_treat, y_treat)
            model_control.fit(X_control, y_control)
            
            # Compute pseudo-outcomes
            tau_treat = y_treat - model_control.predict(X_treat)
            tau_control = model_treat.predict(X_control) - y_control
            
            # Refit the models using the pseudo-outcomes
            model_treat.fit(X_control, tau_control)
            model_control.fit(X_treat, tau_treat)
            cate_pred = np.concatenate([
                model_treat.predict(X_control),
                model_control.predict(X_treat)
            ])

        elif meta_learner == 'R-Learner':
            # Step 1: Residualize the outcome and treatment
            model_y = model.__class__(**model.get_params())  # New instance for outcome regression
            model_t = model.__class__(**model.get_params())  # New instance for treatment regression
            model_y.fit(X_boot, y_boot)
            model_t.fit(X_boot, t_boot)
            
            y_residual = y_boot - model_y.predict(X_boot)
            t_residual = t_boot - model_t.predict(X_boot)
            
            # Step 2: Regress the outcome residuals on treatment residuals
            model.fit(t_residual.reshape(-1, 1), y_residual)
            cate_pred = model.predict(t_residual.reshape(-1, 1))
        
        # Compute the MSE for this bootstrap sample
        mse_boot = mean_squared_error(y_boot, cate_pred)
        mse_list.append(mse_boot)
    
    # Compute the Expected MSE (EMSE)
    emse = np.mean(mse_list)
    return emse


def grid_search_cv(X_train, y_train, treatment_train, model, params, meta_learner):
    """
    Perform GridSearchCV for hyperparameter tuning and return the best model, EMSE, MAE, and R².
    """
    y_train = y_train.ravel()  # Ensure y_train is 1D
    
    # For S-Learner, combine covariates and treatment into a single dataset
    if meta_learner == 'S-Learner':
        X = np.column_stack([X_train, treatment_train])
    else:
        X = X_train
    
    # Run GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y_train)
    
    # Get the best model from the search
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    if meta_learner == 'R-Learner':
        # Residualize the outcome and treatment for R-Learner
        model_y = model.__class__(**model.get_params())  # New instance for outcome regression
        model_t = model.__class__(**model.get_params())  # New instance for treatment regression
        model_y.fit(X_train, y_train)
        model_t.fit(X_train, treatment_train)
        
        y_residual = y_train - model_y.predict(X_train)
        t_residual = treatment_train - model_t.predict(X_train)

        # Final CATE regression on residualized treatment
        best_model.fit(t_residual.reshape(-1, 1), y_residual)
        y_pred = best_model.predict(t_residual.reshape(-1, 1))
    
    else:
        # For S-Learner and others
        if meta_learner == 'S-Learner':
            X_best = np.column_stack([X_train, treatment_train])
        else:
            X_best = X_train
        
        y_pred = best_model.predict(X_best)
    
    # Calculate evaluation metrics
    best_mae = mean_absolute_error(y_train, y_pred)
    best_r2 = r2_score(y_train, y_pred)
    
    # Compute EMSE using bootstrapping
    emse = bootstrap_emse(X_train, y_train, treatment_train, best_model, n_bootstraps=100, meta_learner=meta_learner)
    
    return best_model, emse, best_mae, best_r2, best_params

def apply_grid_search():
    results = []
    meta_learners = ['S-Learner', 'T-Learner', 'X-Learner', 'R-Learner']
    param_grids = get_param_grid()

    # call data preprocessing 
    X_train_processed, y_train, treatment_train, test_data = preprocessor(load_data())
    for model_name, config in param_grids.items():
        for learner in meta_learners:
            print(f"Running GridSearchCV for {model_name} using {learner}...")
            # Run the grid search using the preprocessed data
            best_model, best_emse, best_mae, best_r2,best_params = grid_search_cv(X_train_processed, y_train, 
                                                                    treatment_train, 
                                                                    config['model'], config['params'], learner)
            
            # Store results
            result = {
                'model': model_name,
                'learner': learner,
                'best_emse': best_emse,
                'best_mae': best_mae,
                'best_r2': best_r2,
                'best_params': best_params
            }
            results.append(result)
            
            print(f"{learner} {model_name} Results: Best EMSE={best_emse:.4f}, Best MAE={best_mae:.4f}, Best R²={best_r2:.4f}")
            
    # Convert results to a DataFrame for easy viewing
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
    results_df = apply_grid_search()
    # Save the results as a CSV file
    results_df.to_csv('grid_search_results.csv', index=False)
    print(results_df)
    