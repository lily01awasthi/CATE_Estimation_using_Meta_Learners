# Import the necessary functions from the scripts
from models.Meta_Learners.s_Learner import s_fit, predict_outcomes, estimate_CATE
from models.Meta_Learners.x_learner import x_fit, predict_outcomes_x, estimate_CATE_x
from models.Meta_Learners.r_learner import r_fit, predict_outcomes_r, estimate_CATE_r
from models.Meta_Learners.t_learner import t_fit, predict_outcomes_t, estimate_CATE_t
from data import data_preprocessing
import pandas as pd


class TrainAndPredict:
    def __init__(self, data):
        """
        Initialize the TrainAndPredict class with the dataset.

        Parameters:
        data: pd.DataFrame, the preprocessed dataset
        """
        self.data = data

    def extract_features(self):
        """
        Extract covariates, treatment, and outcome from preprocessed data.

        Returns:
        treatment_col: str, treatment assignment column name
        outcome_col: str, outcome variable column name
        covariate_cols: list of str, covariate variable column names
        """
        # Define columns
        treatment_col = 'GrowthMindsetIntervention'
        outcome_col = 'StudentAchievementScore'
        covariate_cols = [
            'FutureSuccessExpectations', 'StudentRaceEthnicity', 'StudentGender',
            'FirstGenCollegeStatus', 'SchoolUrbanicity', 'PreInterventionFixedMindset',
            'SchoolAchievementLevel', 'SchoolMinorityComposition', 'PovertyConcentration',
            'TotalStudentPopulation'
        ]
        return treatment_col, outcome_col, covariate_cols

    def train_and_predict(self):
        """
        Train the S-Learner model and predict the CATE.

        Returns:
        CATE: pd.Series, the Conditional Average Treatment Effect estimates
        """
        treatment_col, outcome_col, covariate_cols = self.extract_features()

        # S-Learner
        s_model = s_fit(self.data, treatment_col, outcome_col, covariate_cols)
        s_predictions = predict_outcomes(self.data[covariate_cols], s_model, treatment_col)
        s_cate = estimate_CATE(s_predictions)
        data_with_s_cate = self.data.copy()
        data_with_s_cate['CATE'] = s_cate

        # T-Learner
        t_model_treated, t_model_control = t_fit(self.data, treatment_col, outcome_col, covariate_cols)
        t_predictions = predict_outcomes_t(self.data[covariate_cols], t_model_treated, t_model_control)
        t_cate = estimate_CATE_t(t_predictions)
        data_with_t_cate = self.data.copy()
        data_with_t_cate['CATE'] = t_cate

        # X-Learner
        x_model_treated, x_model_control = x_fit(self.data, treatment_col, outcome_col, covariate_cols)
        x_predictions = predict_outcomes_x(self.data[covariate_cols], x_model_treated, x_model_control)
        x_cate = estimate_CATE_x(x_predictions)
        data_with_x_cate = self.data.copy()
        data_with_x_cate['CATE'] = x_cate

        # R-Learner
        r_tau_model, r_y_model, r_t_model, y_residual, t_residual = r_fit(self.data, treatment_col, outcome_col,
                                                                          covariate_cols)
        print(y_residual.describe())
        print(t_residual.describe())
        r_predictions = predict_outcomes_r(self.data[covariate_cols], r_tau_model, r_y_model, r_t_model)
        r_cate = estimate_CATE_r(r_predictions)
        data_with_r_cate = self.data.copy()
        data_with_r_cate['CATE'] = r_cate
        # print(f"y residual{max(y_residual),min(y_residual)}, t residual {max(t_residual),min(t_residual)}")
        y_arry = pd.DataFrame(y_residual)
        t_arry = pd.DataFrame(t_residual)
        print(t_arry.describe())
        print(y_arry.describe())

        return data_with_s_cate, data_with_t_cate, data_with_x_cate, data_with_r_cate


# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    path = '../data/dataset.csv'  # Replace with the actual path to your dataset
    processed_data = data_preprocessing.preprocess_data(path)  # Get the preprocessed data
    tp = TrainAndPredict(processed_data)
    s_estimates, t_estimates, x_estimates, r_estimates = tp.train_and_predict()
    # print(s_estimates, t_estimates, x_estimates, r_estimates)

    # Save the CATE estimates to a CSV file
    s_output_path = '../results/s_predictions.csv'
    t_output_path = '../results/t_predictions.csv'
    x_output_path = '../results/x_predictions.csv'
    r_output_path = '../results/r_predictions.csv'

    s_estimates.to_csv(s_output_path, index=False)
    t_estimates.to_csv(t_output_path, index=False)
    x_estimates.to_csv(x_output_path, index=False)
    r_estimates.to_csv(r_output_path, index=False)
