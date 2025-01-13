import pandas as pd
from sklearn.model_selection import train_test_split
from models.Meta_learners_benchmark_data.s_learner import s_learner
from models.Meta_learners_benchmark_data.t_learner import t_learner
from models.Meta_learners_benchmark_data.x_learner import x_learner
from models.Meta_learners_benchmark_data.r_learner import r_learner

def load_data(path):
    # Load datasets
    training_data = pd.read_csv(f"{path}/checkins_lessons_checkouts_training.csv")
    ground_truth = pd.read_csv(f"{path}/construct_experiments_ates_test.csv")
    return training_data, ground_truth

def preprocess_data(training_data, ground_truth):
    # Filter relevant rows based on the "Type" column
    filtered_training_data = training_data[training_data["Type"].isin(["Checkin", "Checkout"])]

    # Handle missing values in the "IsCorrect" column by removing rows 
    filtered_training_data = filtered_training_data.dropna(subset=["IsCorrect"])

    # Convert the "Timestamp" column to datetime if needed
    filtered_training_data["Timestamp"] = pd.to_datetime(filtered_training_data["Timestamp"])

    # Merge training data with ground truth based on ConstructId and QuestionConstructId
    merged_data = pd.merge(
        filtered_training_data,
        ground_truth,
        how="inner",
        left_on="ConstructId",
        right_on="QuestionConstructId"
    )

    # Convert ControlLessonConstructIds to integer for comparison
    merged_data["ControlLessonConstructIds"] = merged_data["ControlLessonConstructIds"].str.strip("{}").astype(int)    

    # Check overlaps between QuestionConstructId and treatment/control IDs
    question_ids = merged_data["QuestionConstructId"].unique()
    treatment_ids = merged_data["TreatmentLessonConstructId"].unique()
    control_ids = merged_data["ControlLessonConstructIds"].unique()

    # Identify overlapping IDs
    common_treatment_ids = set(question_ids) & set(treatment_ids)
    common_control_ids = set(question_ids) & set(control_ids)

    # Filter treatment and control groups
    treatment_group = merged_data[merged_data["QuestionConstructId"].isin(common_treatment_ids)]
    control_group = merged_data[merged_data["QuestionConstructId"].isin(common_control_ids)]

    # Add a Treatment column
    treatment_group["Treatment"] = 1
    control_group["Treatment"] = 0

    # Combine treatment and control groups
    combined_data = pd.concat([treatment_group, control_group])

    # Select features for meta-learners
    meta_learner_data = combined_data[[
        "Treatment", "IsCorrect", "AnswerValue", "CorrectAnswer",
        "QuestionSequence", "ConstructId", "ate_p_1__", "ate_k_1__"
    ]]

    # Rename ground truth CATE columns
    meta_learner_data = meta_learner_data.rename(columns={
        "ate_p_1__": "TrueCATE_p",
        "ate_k_1__": "TrueCATE_k"
    })

    return meta_learner_data

def split_per_hypothesis_for_test(x, hypothesis):
    """
    Split the data into training and test sets for both hypotheses.
    This function retains the Treatment column separately.
    """
    X_train, X_test, y_train, y_test = train_test_split(x, hypothesis, test_size=0.3, random_state=42)
    treatment_train = X_train["Treatment"].values  # Extract Treatment column for training
    treatment_test = X_test["Treatment"].values  # Extract Treatment column for testing

    return X_train, X_test, y_train, y_test, treatment_train, treatment_test


def split_data_for_test(meta_learner_data):
    """
    Split the meta-learner data into training and test sets for both hypotheses.
    Returns features, targets, and treatment assignments separately.
    """
    # Define features and target
    X = meta_learner_data.drop(columns=["TrueCATE_p", "TrueCATE_k"])

    # Create separate targets for both hypotheses
    y_p_hypothesis_1 = meta_learner_data["TrueCATE_p"]  # Ground truth for Hypothesis 1 (ate_p_1__)
    y_k_hypothesis_2 = meta_learner_data["TrueCATE_k"]  # Ground truth for Hypothesis 2 (ate_k_1__)

    # Split the data into training and test sets for both hypotheses
    X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p = split_per_hypothesis_for_test(X, y_p_hypothesis_1)
    X_train_k, X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k = split_per_hypothesis_for_test(X, y_k_hypothesis_2)

    return (
        X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p,
        X_train_k, X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k
    )

def split_per_hypothesis(x, hypothesis):
    # Split the data into training and test sets for both hypotheses
    X_train, X_test, y_train, y_test = train_test_split(x, hypothesis, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def split_data(meta_learner_data):

    # Create separate targets for both hypotheses
    y_p_hypothesis_1 = meta_learner_data["TrueCATE_p"]  # Ground truth for Hypothesis 1 (ate_p_1__)
    y_k_hypothesis_2 = meta_learner_data["TrueCATE_k"]  # Ground truth for Hypothesis 2 (ate_k_1__)

    # Split the data into training and test sets for both hypotheses
    X_train_p, X_test_p, y_train_p, y_test_p = split_per_hypothesis(meta_learner_data, y_p_hypothesis_1)
    X_train_k, X_test_k, y_train_k, y_test_k = split_per_hypothesis(meta_learner_data, y_k_hypothesis_2)

    return X_train_p, X_test_p, y_train_p, y_test_p, X_train_k, X_test_k, y_train_k, y_test_k    

# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    # Load  data
    training_data, ground_truth = load_data(path="data/benchmark_data")
    meta_learner_data = preprocess_data(training_data, ground_truth)
    print(meta_learner_data.head())
    (X_train_p, X_test_p, y_train_p, y_test_p, treatment_train_p, treatment_test_p,X_train_k, 
     X_test_k, y_train_k, y_test_k, treatment_train_k, treatment_test_k) = split_data_for_test(meta_learner_data)

    # Evaluate S-Learner for Hypothesis 1 (ate_p_1__)
    s_learner_cate_p, s_learner_mse_p, s_learner_bias_p, s_learner_variance_p = s_learner(X_train_p, X_test_p, y_train_p, y_test_p)

    # Evaluate S-Learner for Hypothesis 2 (ate_k_1__)
    s_learner_cate_k, s_learner_mse_k, s_learner_bias_k, s_learner_variance_k = s_learner(X_train_k, X_test_k, y_train_k, y_test_k)

    # Output results
    print(f"S-Learner MSE (Hypothesis 1): {s_learner_mse_p}")
    print(f"S-Learner Bias (Hypothesis 1): {s_learner_bias_p}")
    print(f"S-Learner Variance (Hypothesis 1): {s_learner_variance_p}")
    print(f"S-Learner MSE (Hypothesis 2): {s_learner_mse_k}")
    print(f"S-Learner Bias (Hypothesis 2): {s_learner_bias_k}")
    print(f"S-Learner Variance (Hypothesis 2): {s_learner_variance_k}")

    # Evaluate T-Learner for Hypothesis 1 (ate_p_1__)
    t_learner_cate_p, t_learner_mse_p, t_learner_bias_p, t_learner_variance_p = t_learner(X_train_p, X_test_p, y_train_p, y_test_p)

    # Evaluate T-Learner for Hypothesis 2 (ate_k_1__)
    t_learner_cate_k, t_learner_mse_k, t_learner_bias_k, t_learner_variance_k = t_learner(X_train_k, X_test_k, y_train_k, y_test_k)
    print("........................................................................")
    # Output results
    print(f"T-Learner MSE (Hypothesis 1): {t_learner_mse_p}")
    print(f"T-Learner Bias (Hypothesis 1): {t_learner_bias_p}")
    print(f"T-Learner Variance (Hypothesis 1): {t_learner_variance_p}")
    print(f"T-Learner MSE (Hypothesis 2): {t_learner_mse_k}")
    print(f"T-Learner Bias (Hypothesis 2): {t_learner_bias_k}")
    print(f"T-Learner Variance (Hypothesis 2): {t_learner_variance_k}")

    # Evaluate X-Learner for Hypothesis 1 (ate_p_1__)
    x_learner_cate_p, x_learner_mse_p, x_learner_bias_p, x_learner_variance_p = x_learner(X_train_p, X_test_p, y_train_p, y_test_p)

    # Evaluate X-Learner for Hypothesis 2 (ate_k_1__)
    x_learner_cate_k, x_learner_mse_k, x_learner_bias_k, x_learner_variance_k = x_learner(X_train_k, X_test_k, y_train_k, y_test_k)
    print("........................................................................")
    # Output results
    print(f"X-Learner MSE (Hypothesis 1): {x_learner_mse_p}")
    print(f"X-Learner Bias (Hypothesis 1): {x_learner_bias_p}")
    print(f"X-Learner Variance (Hypothesis 1): {x_learner_variance_p}")
    print(f"X-Learner MSE (Hypothesis 2): {x_learner_mse_k}")
    print(f"X-Learner Bias (Hypothesis 2): {x_learner_bias_k}")
    print(f"X-Learner Variance (Hypothesis 2): {x_learner_variance_k}")

    # Evaluate R-Learner for Hypothesis 1 (ate_p_1__)
    r_learner_cate_p, r_learner_mse_p, r_learner_bias_p, r_learner_variance_p = r_learner(X_train_p, X_test_p, y_train_p, y_test_p)

    # Evaluate R-Learner for Hypothesis 2 (ate_k_1__)
    r_learner_cate_k, r_learner_mse_k, r_learner_bias_k, r_learner_variance_k = r_learner(X_train_k, X_test_k, y_train_k, y_test_k)
    print("........................................................................")
    # Output results
    print(f"R-Learner MSE (Hypothesis 1): {r_learner_mse_p}")
    print(f"R-Learner Bias (Hypothesis 1): {r_learner_bias_p}")
    print(f"R-Learner Variance (Hypothesis 1): {r_learner_variance_p}")
    print(f"R-Learner MSE (Hypothesis 2): {r_learner_mse_k}")
    print(f"R-Learner Bias (Hypothesis 2): {r_learner_bias_k}")
    print(f"R-Learner Variance (Hypothesis 2): {r_learner_variance_k}")


