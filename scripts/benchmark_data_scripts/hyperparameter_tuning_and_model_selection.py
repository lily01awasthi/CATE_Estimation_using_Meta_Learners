from data_preprocessing import load_data, preprocess_data, split_data_for_test
from scripts.analysis_data_scripts.hyperparameter_tuning_and_model_selection import get_param_grid, apply_grid_search

# Main function for hyperparameter tuning and model selection
if __name__ == '__main__':
    # Load and preprocess benchmark data
    training_data, ground_truth = load_data(path="data/benchmark_data")
    meta_learner_data = preprocess_data(training_data, ground_truth)
    (X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p,X_train_k, 
     X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k) = split_data_for_test(meta_learner_data)

    # # Define parameter grids and apply grid search
    # param_grid = get_param_grid()

    # # Apply grid search for the main model for hypothesis 1 (ate_p_1__)
    # results_df = apply_grid_search(param_grid, X_train_p, y_train_p, treatment_train_p)
    # print(results_df)
    # results_df.to_csv('results/benchmark_data_results/grid_search_results/test_grid_search_results_hypothesis_p.csv', index=False) 

    # # Apply grid search for the main model for hypothesis 2 (ate_k_1__)
    # results_df = apply_grid_search(param_grid, X_train_k, y_train_k, treatment_train_k)
    # results_df.to_csv('results/benchmark_data_results/grid_search_results/test_grid_search_results_hypothesis_k.csv', index=False)
    
