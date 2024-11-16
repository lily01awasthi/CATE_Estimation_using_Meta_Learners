
import sys
import os
    
import pandas as pd
# Import the necessary functions from the scripts
from models.Meta_Learners.s_Learner import s_fit, predict_outcomes_s, estimate_CATE_s
from models.Meta_Learners.x_learner import x_fit, predict_outcomes_x, estimate_CATE_x
from models.Meta_Learners.r_learner import r_fit, predict_outcomes_r, estimate_CATE_r
from models.Meta_Learners.t_learner import t_fit, predict_outcomes_t, estimate_CATE_t
from scripts.data_preprocessor import load_data, preprocessor
from sklearn.metrics import mean_squared_error
import numpy as np


class TrainAndPredict:
    def __init__(self, X_train, y_train, treatment_train, X_test):
        """
        Initialize the TrainAndPredict class with preprocessed data.

        Parameters:
        X_train: pd.DataFrame, the covariates of the training set
        y_train: np.array, the outcome variable of the training set
        treatment_train: np.array, the treatment assignment of the training set
        X_test: pd.DataFrame, the covariates of the test set
        """
        self.X_train = X_train
        self.y_train = y_train
        self.treatment_train = treatment_train
        self.X_test = X_test

    def extract_features(self):
        """
        Extract column names for treatment, outcome, and covariates.

        Returns:
        treatment_col: str, treatment assignment column name
        outcome_col: str, outcome variable column name
        covariate_cols: list of str, covariate variable column names
        """
        treatment_col = 'GrowthMindsetIntervention'
        outcome_col = 'SchoolAchievementLevel'
        covariate_cols = list(self.X_train.columns)
        return treatment_col, outcome_col, covariate_cols

    def train_and_predict(self):
        """
        Train each meta-learner and predict the CATE for the test set.

        Returns:
        DataFrames with CATE estimates for each meta-learner.
        """
        treatment_col, outcome_col, covariate_cols = self.extract_features()

        # S-Learner
        s_model = s_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                        treatment_col, outcome_col, covariate_cols)
        s_predictions = predict_outcomes_s(self.X_test, s_model, treatment_col)
        s_cate = estimate_CATE_s(s_predictions)
        data_with_s_cate = self.X_test.copy()
        data_with_s_cate['s_CATE'] = s_cate

        # T-Learner
        t_model_treated, t_model_control = t_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        t_predictions = predict_outcomes_t(self.X_test, t_model_treated, t_model_control)
        t_cate = estimate_CATE_t(t_predictions)
        data_with_t_cate = self.X_test.copy()
        data_with_t_cate['t_CATE'] = t_cate

        # X-Learner
        x_model_treated, x_model_control = x_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        x_predictions = predict_outcomes_x(self.X_test, x_model_treated, x_model_control)
        x_cate = estimate_CATE_x(x_predictions)
        data_with_x_cate = self.X_test.copy()
        data_with_x_cate['x_CATE'] = x_cate

        # R-Learner
        r_tau_model = r_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                            treatment_col, outcome_col, covariate_cols)
        r_predictions = predict_outcomes_r(self.X_test, r_tau_model)
        r_cate = estimate_CATE_r(r_predictions)
        data_with_r_cate = self.X_test.copy()
        data_with_r_cate['r_CATE'] = r_cate

        # Print CATE estimates for each meta-learner
        print("S-Learner CATE:", data_with_s_cate.head())
        print("T-Learner CATE:", data_with_t_cate.head())
        print("X-Learner CATE:", data_with_x_cate.head())
        print("R-Learner CATE:", data_with_r_cate.head())

        return data_with_s_cate, data_with_t_cate, data_with_x_cate, data_with_r_cate

def calculate_rmse(true_values, predicted_values):
    return np.sqrt(mean_squared_error(true_values, predicted_values))


# Example usage
if __name__ == '__main__':

    # Ensure the project root is in sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    df = load_data()
    print("Data preprocessing...")
    X_train, y_train, treatment_train, X_test, y_test, treatment_test = preprocessor(df)
    print("Preprocessing done!")

    # Instantiate TrainAndPredict class with preprocessed data
    train_predictor = TrainAndPredict(X_train, y_train, treatment_train, X_test)
    s_estimates, t_estimates, x_estimates, r_estimates = train_predictor.train_and_predict()

    # Save the CATE estimates to a CSV file
    s_output_path = 'results/s_predictions.csv'
    # s_output_path_extratrees = 'results/s_predictions_withExtratrees.csv'
    t_output_path = 'results/t_predictions.csv'
    x_output_path = 'results/x_predictions.csv'
    r_output_path = 'results/r_predictions.csv'

    # s_estimates.to_csv(s_output_path_extratrees, index=False)
    s_estimates.to_csv(s_output_path, index=False)
    t_estimates.to_csv(t_output_path, index=False)
    x_estimates.to_csv(x_output_path, index=False)
    r_estimates.to_csv(r_output_path, index=False)

    # Extract true outcomes for RMSE calculation
    true_outcomes = y_test

    # Calculate RMSE for S-Learner
    s_rmse = calculate_rmse(true_outcomes, s_estimates['s_CATE'])
    print("S-Learner RMSE:", s_rmse)

    # Calculate RMSE for T-Learner
    t_rmse = calculate_rmse(true_outcomes, t_estimates['t_CATE'])
    print("T-Learner RMSE:", t_rmse)

    # Calculate RMSE for X-Learner
    x_rmse = calculate_rmse(true_outcomes, x_estimates['x_CATE'])
    print("X-Learner RMSE:", x_rmse)

    # Calculate RMSE for R-Learner
    r_rmse = calculate_rmse(true_outcomes, r_estimates['r_CATE'])
    print("R-Learner RMSE:", r_rmse)
