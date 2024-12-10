
import sys
import os
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

# Example usage
if __name__ == '__main__':

    # Load  data
    training_data, ground_truth = load_data(path="data/benchmark_data")
    meta_learner_data = preprocess_data(training_data, ground_truth)
    (X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p,X_train_k, 
     X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k) = split_data_for_test(meta_learner_data)

    # get cate estimates for all meta-Learners for hypothesis p
    s_learner_cate_p = s_learner(X_train_p, X_test_p, y_train_p)
    t_learner_cate_p = t_learner(X_train_p, X_test_p, y_train_p)
    x_learner_cate_p = x_learner(X_train_p, X_test_p, y_train_p)
    r_learner_cate_p = r_learner(X_train_p, X_test_p, y_train_p,"p")

    # get cate estimates for all meta-Learners for hypothesis k
    s_learner_cate_k = s_learner(X_train_k, X_test_k, y_train_k)
    t_learner_cate_k = t_learner(X_train_k, X_test_k, y_train_k)
    x_learner_cate_k = x_learner(X_train_k, X_test_k, y_train_k)
    r_learner_cate_k = r_learner(X_train_k, X_test_k, y_train_k,"k")


    # Extract true outcomes for RMSE calculation
    true_outcomes = y_test_p

    redult_df = {}

    # Calculate RMSE for S-Learner
    s_metrics_p = compute_rmse(y_test_p, s_learner_cate_p)
    s_metrics_k = compute_rmse(y_test_k, s_learner_cate_k)
    print("S-Learner evaluation metrics for hypothesis p:", s_metrics_p)
    print("S-Learner evaluation metrics for hypothesis k:", s_metrics_k)
    redult_df["S-Learner"] = [s_metrics_p,s_metrics_k]

    # Calculate RMSE for T-Learner
    t_metrics_p = compute_rmse(y_test_p, t_learner_cate_p)
    t_metrics_k = compute_rmse(y_test_k, t_learner_cate_k)
    print("T-Learner evaluation metrics for hypothesis p:", t_metrics_p)
    print("T-Learner evaluation metrics for hypothesis k:", t_metrics_k)
    redult_df["T-Learner"] = [t_metrics_p,t_metrics_k]

    # Calculate RMSE for X-Learner
    x_metrics_p = compute_rmse(y_test_p, x_learner_cate_p)
    x_metrics_k = compute_rmse(y_test_k, x_learner_cate_k)
    print("X-Learner evaluation metrics for hypothesis p:", x_metrics_p)
    print("X-Learner evaluation metrics for hypothesis k:", x_metrics_k)
    redult_df["X-Learner"] = [x_metrics_p,x_metrics_k]

    # Calculate RMSE for R-Learner
    r_metrics_p = compute_rmse(y_test_p, r_learner_cate_p)
    r_metrics_k = compute_rmse(y_test_k, r_learner_cate_k)
    print("R-Learner evaluation metrics for hypothesis p:", r_metrics_p)
    print("R-Learner evaluation metrics for hypothesis k:", r_metrics_k)
    redult_df["R-Learner"] = [r_metrics_p,r_metrics_k]
    
