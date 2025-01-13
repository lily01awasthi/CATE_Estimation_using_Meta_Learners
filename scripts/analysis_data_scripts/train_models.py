
import sys
import os
import pandas as pd
# Import the necessary functions from the scripts
from models.Meta_Learners.s_Learner import s_fit, predict_outcomes_s, estimate_CATE_s
from models.Meta_Learners.x_learner import x_fit, predict_outcomes_x, estimate_CATE_x
from models.Meta_Learners.r_learner import r_fit, predict_outcomes_r, estimate_CATE_r
from models.Meta_Learners.t_learner import t_fit, predict_outcomes_t, estimate_CATE_t
from scripts.analysis_data_scripts.data_preprocessor import load_data, preprocessor
from scripts.analysis_data_scripts.evaluate_models import evaluate_meta_learner,bootstrap_emse_no_groundtruth
from models.Meta_Learners.meta_learner_models import S_learner_model, T_learner_model, X_learner_model, R_learner_model

class TrainAndPredict:
    def __init__(self, X_train, y_train, treatment_train, X_test,treatment_test):
        """
        Initialize the TrainAndPredict class with training and test data.

        Parameters:
        X_train: DataFrame, training covariates
        y_train: training outcomes
        treatment_train: training treatment assignments
        X_test: test covariates
        treatment_test: test treatment assignments
        """
        self.X_train = X_train
        self.y_train = y_train
        self.treatment_train = treatment_train
        self.X_test = X_test
        self.treatment_test = treatment_test

    def extract_features(self):
        """
        Extract the treatment, outcome, and covariate columns from the training data.

        Returns:
        treatment_col: str, the name of the treatment column
        outcome_col: str, the name of the outcome column
        covariate_cols: list of str, the names of the covariate columns
        """
        treatment_col = 'GrowthMindsetIntervention'
        outcome_col = 'SchoolAchievementLevel'
        covariate_cols = list(self.X_train.columns)
        return treatment_col, outcome_col, covariate_cols
    
    def get_cate_estimates(self,s_predictions, t_predictions, x_predictions, r_predictions):
        """
        Get the CATE estimates for each meta-learner.
        
        Parameters:
        s_predictions: array, predicted outcomes form the S-Learner
        t_predictions: array, predicted outcomes from the T-Learner
        x_predictions: array, predicted outcomes from the X-Learner
        r_predictions: array, predicted cate from the R-Learner
        
        Returns:
        data_with_s_cate: DataFrame with CATE estimates for the S-Learner
        data_with_t_cate: DataFrame with CATE estimates for the T-Learner
        data_with_x_cate: DataFrame with CATE estimates for the X-Learner
        data_with_r_cate: DataFrame with CATE estimates for the R-Learner
        
        """
        
        s_cate = estimate_CATE_s(s_predictions)
        data_with_s_cate = self.X_test.copy()
        data_with_s_cate['treatment'] = self.treatment_test  
        data_with_s_cate['s_CATE'] = s_cate

        t_cate = estimate_CATE_t(t_predictions)
        data_with_t_cate = self.X_test.copy()
        data_with_t_cate['treatment'] = self.treatment_test
        data_with_t_cate['t_CATE'] = t_cate

        x_cate = estimate_CATE_x(x_predictions)
        data_with_x_cate = self.X_test.copy()
        data_with_x_cate['treatment'] = self.treatment_test
        data_with_x_cate['x_CATE'] = x_cate

        r_cate = estimate_CATE_r(r_predictions)
        data_with_r_cate = self.X_test.copy()
        data_with_r_cate['treatment'] = self.treatment_test
        data_with_r_cate['r_CATE'] = r_cate

        return data_with_s_cate, data_with_t_cate, data_with_x_cate, data_with_r_cate

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
        print(f"s_predictions:{s_predictions}")

        # T-Learner
        t_model_treated, t_model_control = t_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        t_predictions = predict_outcomes_t(self.X_test, t_model_treated, t_model_control)
        print(f"t_predictions:{t_predictions}")

        # X-Learner
        x_model_treated, x_model_control, x_model_cate = x_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                                                 treatment_col, outcome_col, covariate_cols)
        x_predictions = predict_outcomes_x(self.X_test, x_model_treated, x_model_control,x_model_cate)
        print(f"x_predictions:{x_predictions}")

        # R-Learner
        r_tau_model = r_fit(pd.concat([self.X_train, pd.Series(self.treatment_train, name=treatment_col)], axis=1),
                            treatment_col, outcome_col, covariate_cols)
        r_predictions = predict_outcomes_r(self.X_test, r_tau_model)
        print(f"r_predictions:{r_predictions}")


        return s_predictions, t_predictions, x_predictions, r_predictions


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
    train_predictor = TrainAndPredict(X_train, y_train, treatment_train, X_test,treatment_test)
    s_predictions, t_predictions, x_predictions, r_predictions = train_predictor.train_and_predict()
    s_estimates, t_estimates, x_estimates, r_estimates = train_predictor.get_cate_estimates(s_predictions, t_predictions, x_predictions, r_predictions)

    # Save the outcome predictions to a CSV file
    s_output_path = 'results/analysis_data_results/test_data/s_predictions.csv'
    t_output_path = 'results/analysis_data_results/test_data/t_predictions.csv'
    x_output_path = 'results/analysis_data_results/test_data/x_predictions.csv'
    r_output_path = 'results/analysis_data_results/test_data/r_predictions.csv'

    # s_estimates.to_csv(s_output_path_extratrees, index=False)
    s_estimates.to_csv(s_output_path, index=False)
    t_estimates.to_csv(t_output_path, index=False)
    x_estimates.to_csv(x_output_path, index=False)
    r_estimates.to_csv(r_output_path, index=False)

    # Extract true outcomes for RMSE calculation
    true_outcomes = y_test

    results = {}

    # Calculate RMSE for S-Learner
    s_metrics = evaluate_meta_learner(s_predictions, y_test, treatment_test, 'S-Learner')
    print("S-Learner evaluation metrics:", s_metrics)

    # Calculate RMSE for T-Learner
    t_metrics = evaluate_meta_learner(t_predictions, y_test, treatment_test, 'T-Learner')
    print("T-Learner evaluation metrics:", t_metrics)

    # Calculate RMSE for X-Learner
    x_metrics = evaluate_meta_learner(x_predictions, y_test, treatment_test, 'X-Learner')
    print("X-Learner evaluation metrics:", x_metrics)

   # calculate varianle of eastimated cate given by each meta learner
    s_emse = bootstrap_emse_no_groundtruth(X_train, treatment_train, S_learner_model, 'S-Learner')
    print(f"s_emse:{s_emse}")

    t_emse = bootstrap_emse_no_groundtruth(X_train, treatment_train, T_learner_model, 'T-Learner')
    print(f"t_emse:{t_emse}")

    x_emse = bootstrap_emse_no_groundtruth(X_train, treatment_train, X_learner_model, 'X-Learner')
    print(f"x_emse:{x_emse}")

    r_emse = bootstrap_emse_no_groundtruth(X_train, treatment_train, R_learner_model, 'R-Learner')
    print(f"r_emse:{r_emse}")

    

