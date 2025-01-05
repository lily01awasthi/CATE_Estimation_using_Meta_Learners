
import pandas as pd
from scripts.benchmark_data_scripts.data_preprocessing import load_data, preprocess_data, split_data_for_test
from sklearn.metrics import mean_squared_error
import numpy as np
from models.Meta_learners_benchmark_data.s_learner import s_learner
from models.Meta_learners_benchmark_data.t_learner import t_learner
from models.Meta_learners_benchmark_data.x_learner import x_learner
from models.Meta_learners_benchmark_data.r_learner import r_learner


# Evaluation Metric Functions
def compute_rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared= False )

def compute_bias(y_true, y_pred):
    return np.mean(y_pred - y_true)

def compute_variance(y_pred):
    return np.var(y_pred)

# Example usage
if __name__ == '__main__':

    # Load  data
    training_data, ground_truth = load_data(path="data/benchmark_data")
    meta_learner_data = preprocess_data(training_data, ground_truth)
    (X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p,X_train_k, 
     X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k) = split_data_for_test(meta_learner_data)


    # Get CATE estimates for all meta-learners for hypothesis p
    learners_p = {
        "S-Learner": s_learner(X_train_p, X_test_p, y_train_p),
        "T-Learner": t_learner(X_train_p, X_test_p, y_train_p),
        "X-Learner": x_learner(X_train_p, X_test_p, y_train_p),
        "R-Learner": r_learner(X_train_p, X_test_p, y_train_p, "p")
    }

    # Get CATE estimates for all meta-learners for hypothesis k
    learners_k = {
        "S-Learner": s_learner(X_train_k, X_test_k, y_train_k),
        "T-Learner": t_learner(X_train_k, X_test_k, y_train_k),
        "X-Learner": x_learner(X_train_k, X_test_k, y_train_k),
        "R-Learner": r_learner(X_train_k, X_test_k, y_train_k, "k")
    }


    # Extract true outcomes for RMSE calculation
    true_outcomes = y_test_p

    # Initialize results storage
    result_df = []

    # Evaluate metrics for hypothesis p
    for learner_name, cate_predictions in learners_p.items():
        rmse = compute_rmse(y_test_p, cate_predictions)
        bias = compute_bias(y_test_p, cate_predictions)
        variance = compute_variance(cate_predictions)
        result_df.append({
            "Learner": learner_name,
            "Hypothesis": "p",
            "RMSE": rmse,
            "Bias": bias,
            "Variance": variance
        })

    # Evaluate metrics for hypothesis k
    for learner_name, cate_predictions in learners_k.items():
        rmse = compute_rmse(y_test_k, cate_predictions)
        bias = compute_bias(y_test_k, cate_predictions)
        variance = compute_variance(cate_predictions)
        result_df.append({
            "Learner": learner_name,
            "Hypothesis": "k",
            "RMSE": rmse,
            "Bias": bias,
            "Variance": variance
        })

    # Convert results to DataFrame for better visualization
    result_df = pd.DataFrame(result_df)

    # Print the results
    print(result_df)

  
    
