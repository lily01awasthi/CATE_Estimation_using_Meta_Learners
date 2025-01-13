from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from scripts.analysis_data_scripts.data_preprocessor import preprocessor, load_data
from scripts.analysis_data_scripts.evaluate_models import compute_rmse, bootstrap_emse_no_groundtruth
import time


def get_param_grid():
    return {
        'RandomForest': {
            'model': RandomForestRegressor(),
            'params': {
                'n_estimators': [100, 200, 300],  # Number of trees
                'max_depth': [None, 4, 8],  # Maximum depth of the tree
                'min_samples_split': [2, 10, 20],  # Minimum number of samples required to split a node
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

# Meta-Learner Helper Functions
def s_learner_grid_search(X_train, y_train, treatment_train, model, params):
    X_combined = np.column_stack([X_train, treatment_train])
    grid_search = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_combined, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Updated `t_learner_grid_search` to ensure it returns two values: (models, params)
def t_learner_grid_search(X_train, y_train, treatment_train, model, params):
    X_control, y_control = X_train[treatment_train == 0], y_train[treatment_train == 0]
    X_treated, y_treated = X_train[treatment_train == 1], y_train[treatment_train == 1]

    # Grid search for control model
    grid_search_control = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_control.fit(X_control, y_control)
    best_model_control = grid_search_control.best_estimator_

    # Grid search for treated model
    grid_search_treated = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_treated.fit(X_treated, y_treated)
    best_model_treated = grid_search_treated.best_estimator_

    # Return the two models as a tuple, and combine parameters into a dictionary
    return (best_model_control, best_model_treated), {'control_params': grid_search_control.best_params_, 'treated_params': grid_search_treated.best_params_}

# Updated `x_learner_grid_search` to return two values consistently
def x_learner_grid_search(X_train, y_train, treatment_train, model, params):
    X_control, y_control = X_train[treatment_train == 0], y_train[treatment_train == 0]
    X_treated, y_treated = X_train[treatment_train == 1], y_train[treatment_train == 1]

    # Train initial models for control and treated groups
    grid_search_control = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_control.fit(X_control, y_control)
    best_model_control = grid_search_control.best_estimator_

    grid_search_treated = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_treated.fit(X_treated, y_treated)
    best_model_treated = grid_search_treated.best_estimator_

    # Estimate treatment effects for control and treated groups
    tau_control = y_control - best_model_treated.predict(X_control)
    tau_treated = best_model_control.predict(X_treated) - y_treated

    # Combine pseudo-outcomes
    X_combined = np.vstack([X_control, X_treated])
    tau_combined = np.hstack([tau_control, tau_treated])

    # Train final CATE model on pseudo-outcomes
    cate_model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    cate_model.fit(X_combined, tau_combined)

    # Return both models and final CATE model as tuple, and parameter dictionary
    return (best_model_control, best_model_treated, cate_model.best_estimator_), {'control_params': grid_search_control.best_params_, 'treated_params': grid_search_treated.best_params_, 'cate_params': cate_model.best_params_}

def r_learner_grid_search(X_train, y_train, treatment_train, model, params):
    model_y = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    model_t = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    model_y.fit(X_train, y_train)
    model_t.fit(X_train, treatment_train)

    y_residual = y_train - model_y.predict(X_train)
    t_residual = treatment_train - model_t.predict(X_train)

    cate_model = GridSearchCV(estimator=model, param_grid=params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    cate_model.fit(t_residual.reshape(-1, 1), y_residual)

    return cate_model.best_estimator_, cate_model.best_params_

def grid_search_cv(X_train, y_train, treatment_train, model, params, meta_learner):
    if meta_learner == 'S-Learner':
        best_model, best_params = s_learner_grid_search(X_train, y_train, treatment_train, model, params)
    elif meta_learner == 'T-Learner':
        # T-Learner returns two models, package them as a tuple with parameters
        best_models, best_params = t_learner_grid_search(X_train, y_train, treatment_train, model, params)
        best_model = best_models  # Maintain two models as tuple for T-Learner
    elif meta_learner == 'X-Learner':
        # X-Learner can return two models plus an optional CATE model
        best_models, best_params = x_learner_grid_search(X_train, y_train, treatment_train, model, params)
        best_model = best_models  # Keep as tuple if needed in `apply_grid_search`
    elif meta_learner == 'R-Learner':
        best_model, best_params = r_learner_grid_search(X_train, y_train, treatment_train, model, params)
    else:
        raise ValueError(f"Unknown meta-learner: {meta_learner}")
    
    # Return two items consistently, with best_model as a tuple if needed
    return best_model, best_params

# Main function to apply grid search across meta-learners and models
def apply_grid_search(param_grids, X_train, y_train, treatment_train):
    results = []
    meta_learners = ['S-Learner','T-Learner', 'X-Learner', 'R-Learner'] # all meta_learners
    # meta_learners = ['S-Learner'] #, 'S-Learner' only
    # meta_learners = ['T-Learner'] #, 'T-Learner' only
    # meta_learners = ['X-Learner'] #, 'X-Learner' only
    # meta_learners = ['R-Learner'] #, 'R-Learner' only

    for model_name, config in param_grids.items():
        for learner in meta_learners:
            print(f"Running GridSearchCV for {model_name} using {learner}...")
            best_model, best_params = grid_search_cv(X_train, y_train, treatment_train, config['model'], config['params'], learner)

            # Prepare data for predictions depending on the meta-learner
            if learner == 'S-Learner':
                X_combined = np.column_stack([X_train, treatment_train])  # Add treatment as feature
                predictions = best_model.predict(X_combined)
                
            elif learner == 'T-Learner':
                # For T-Learner, we need to use two separate models for control and treated groups
                (best_model_control, best_model_treated) = best_model  # Unpack the tuple

                # Get predictions separately for control and treated groups
                predictions_control = best_model_control.predict(X_train[treatment_train == 0])
                predictions_treated = best_model_treated.predict(X_train[treatment_train == 1])

                # Combine predictions back into a single array matching the order of `y_train`
                predictions = np.zeros_like(y_train)
                predictions[treatment_train == 0] = predictions_control
                predictions[treatment_train == 1] = predictions_treated

            elif learner == 'X-Learner':
                # Unpack the tuple of models for X-Learner (includes control, treated, and cate_model)
                best_model_control, best_model_treated, cate_model = best_model

                # Calculate treatment effects for control and treated groups
                tau_control = y_train[treatment_train == 0] - best_model_treated.predict(X_train[treatment_train == 0])
                tau_treated = best_model_control.predict(X_train[treatment_train == 1]) - y_train[treatment_train == 1]

                # Combine treatment effect predictions back into a single array matching the order of `y_train`
                predictions = np.zeros_like(y_train, dtype=np.float64)
                predictions[treatment_train == 0] = tau_control
                predictions[treatment_train == 1] = tau_treated

            elif learner == 'R-Learner':
                # For R-Learner, residualize the treatment and outcome before prediction
                model_y = config['model'].__class__(**config['model'].get_params())
                model_t = config['model'].__class__(**config['model'].get_params())
                
                # Fit models on y and treatment for residual calculation
                model_y.fit(X_train, y_train)
                model_t.fit(X_train, treatment_train)

                y_residual = y_train - model_y.predict(X_train)
                t_residual = treatment_train - model_t.predict(X_train)

                # Predict using residualized treatment only (1 feature)
                predictions = best_model.predict(t_residual.reshape(-1, 1))
            else:
                raise ValueError(f"Unknown meta-learner: {learner}")
            
            # Calculate metrics using functions from evaluate.py
            if learner != 'R-Learner':
                rmse = compute_rmse(y_train, predictions)
            else:
                rmse = None
                
            emse = bootstrap_emse_no_groundtruth(X_train, treatment_train, best_model, learner)
            

            # Append the result
            results.append({
                'model': model_name,
                'learner': learner,
                'best_rmse': rmse,
                'best_emse': emse,
                'best_params': best_params
                
            })

    return pd.DataFrame(results)


if __name__ == '__main__':

    # Start timing
    start_time = time.time()

    # Load and preprocess data
    df = load_data(path='data/analysis_data/dataset.csv')
    X_train, y_train, treatment_train, X_test, y_test, treatment_test = preprocessor(df)
    
    # Define parameter grids and apply grid search
    param_grid = get_param_grid()
    results_df = apply_grid_search(param_grid, X_train, y_train, treatment_train)
    
    # Save and print results
    results_df.to_csv('results/analysis_data_results/grid_search_results/grid_search_results.csv', index=False) # all_meta_learners 
    # results_df.to_csv('results/analysis_data_results/grid_search_results/grid_search_results_S_learner.csv', index=False) # S_learner
    # results_df.to_csv('results/analysis_data_results/grid_search_results/grid_search_results_T_learner.csv', index=False) # T_learner
    # results_df.to_csv('results/analysis_data_results/grid_search_results/grid_search_results_X_learner.csv', index=False) # X_learner
    # results_df.to_csv('results/analysis_data_results/grid_search_results/grid_search_results_R_learner.csv', index=False) # R_learner

    # print(results_df)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
