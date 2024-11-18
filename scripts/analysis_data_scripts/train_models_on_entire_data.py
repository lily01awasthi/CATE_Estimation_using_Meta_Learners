import sys
import os

import pandas as pd
from models.Meta_Learners.s_Learner import s_fit, predict_outcomes_s, estimate_CATE_s
from models.Meta_Learners.x_learner import x_fit, predict_outcomes_x, estimate_CATE_x
from models.Meta_Learners.r_learner import r_fit, predict_outcomes_r, estimate_CATE_r
from models.Meta_Learners.t_learner import t_fit, predict_outcomes_t, estimate_CATE_t
from scripts.analysis_data_scripts.data_preprocessor import load_data, preprocessor


class TrainAndPredict:
    def __init__(self, X, y, treatment):
        """
        Initialize the TrainAndPredict class with preprocessed data.

        Parameters:
        X: pd.DataFrame, the covariates of the dataset
        y: np.array, the outcome variable of the dataset
        treatment: np.array, the treatment assignment of the dataset
        """
        self.X = X
        self.y = y
        self.treatment = treatment

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
        covariate_cols = list(self.X.columns)
        return treatment_col, outcome_col, covariate_cols

    def train_and_predict(self):
        """
        Train each meta-learner and predict the CATE for the entire dataset.

        Returns:
        DataFrames with CATE estimates for each meta-learner.
        """
        treatment_col, outcome_col, covariate_cols = self.extract_features()

        # S-Learner
        s_model = s_fit(pd.concat([self.X, pd.Series(self.treatment, name=treatment_col)], axis=1),
                        treatment_col, outcome_col, covariate_cols)
        s_predictions = predict_outcomes_s(self.X, s_model, treatment_col)
        s_cate = estimate_CATE_s(s_predictions)
        data_with_s_cate = self.X.copy()
        data_with_s_cate['s_CATE'] = s_cate

        # T-Learner
        t_model_treated, t_model_control = t_fit(pd.concat([self.X, pd.Series(self.treatment, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        t_predictions = predict_outcomes_t(self.X, t_model_treated, t_model_control)
        t_cate = estimate_CATE_t(t_predictions)
        data_with_t_cate = self.X.copy()
        data_with_t_cate['t_CATE'] = t_cate

        # X-Learner
        x_model_treated, x_model_control = x_fit(pd.concat([self.X, pd.Series(self.treatment, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        x_predictions = predict_outcomes_x(self.X, x_model_treated, x_model_control)
        x_cate = estimate_CATE_x(x_predictions)
        data_with_x_cate = self.X.copy()
        data_with_x_cate['x_CATE'] = x_cate

        # R-Learner
        r_tau_model = r_fit(pd.concat([self.X, pd.Series(self.treatment, name=treatment_col)], axis=1),
                            treatment_col, outcome_col, covariate_cols)
        r_predictions = predict_outcomes_r(self.X, r_tau_model)
        r_cate = estimate_CATE_r(r_predictions)
        data_with_r_cate = self.X.copy()
        data_with_r_cate['r_CATE'] = r_cate

        # Print CATE estimates for each meta-learner
        print("S-Learner CATE:", data_with_s_cate.head())
        print("T-Learner CATE:", data_with_t_cate.head())
        print("X-Learner CATE:", data_with_x_cate.head())
        print("R-Learner CATE:", data_with_r_cate.head())

        return data_with_s_cate, data_with_t_cate, data_with_x_cate, data_with_r_cate


# Example usage
if __name__ == '__main__':
    # Ensure the project root is in sys.path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in sys.path:
        sys.path.append(project_root)

    df = load_data()
    print("Data preprocessing...")
    X, y, treatment = preprocessor(df, split_data=False)
    print("Preprocessing done!")

    # Instantiate TrainAndPredict class with preprocessed data
    train_predictor = TrainAndPredict(X, y, treatment)
    s_estimates, t_estimates, x_estimates, r_estimates = train_predictor.train_and_predict()

    # Save the CATE estimates to a CSV file
    s_output_path = 'results/analysis_data_results/entire_data/s_predictions.csv'
    t_output_path = 'results/analysis_data_results/entire_data/t_predictions.csv'
    x_output_path = 'results/analysis_data_results/entire_data/x_predictions.csv'
    r_output_path = 'results/analysis_data_results/entire_data/r_predictions.csv'

    s_estimates.to_csv(s_output_path, index=False)
    t_estimates.to_csv(t_output_path, index=False)
    x_estimates.to_csv(x_output_path, index=False)
    r_estimates.to_csv(r_output_path, index=False)
