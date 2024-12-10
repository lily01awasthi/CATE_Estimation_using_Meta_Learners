import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def calculate_psm(data, features, outcome, treatment, true_cate, hypothesis_name):
    """
    Calculate CATE using Propensity Score Matching (PSM) and evaluate RMSE, bias, and variance.

    Parameters:
    - data: DataFrame containing the dataset.
    - features: List of feature column names.
    - outcome: Outcome variable column name.
    - treatment: Treatment variable column name (1 for treated, 0 for control).
    - true_cate: Column containing true CATE values (if available).
    - hypothesis_name: Name of the hypothesis (for labeling results).

    Returns:
    - Results dictionary containing RMSE, bias, variance, and CATE estimates.
    """
    # Step 1: Estimate propensity scores
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(data[features], data[treatment])
    data['propensity_score'] = logistic_model.predict_proba(data[features])[:, 1]

    # Step 2: Perform Nearest Neighbor Matching
    treated = data[data[treatment] == 1]
    control = data[data[treatment] == 0]
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])

    # Define a matching threshold (optional)
    threshold = 0.1
    matched_indices = distances.flatten() <= threshold

    # Match treated individuals to control individuals
    matched_treated = treated[matched_indices]
    matched_control = control.iloc[indices.flatten()[matched_indices]]

    # Step 3: Estimate CATE
    cate_estimates = matched_treated[outcome].values - matched_control[outcome].values

    # Evaluate RMSE, bias, and variance if ground truth CATE is available
    if true_cate in data.columns:
        true_cate_values = matched_treated[true_cate].values
        rmse = np.sqrt(mean_squared_error(true_cate_values, cate_estimates))
        bias = np.mean(cate_estimates - true_cate_values)
        variance = np.var(cate_estimates)
    else:
        rmse, bias, variance = None, None, None

    # Return results
    results = {
        "Hypothesis": hypothesis_name,
        "Matched Count": len(matched_treated),
        "Unmatched Count": len(treated) - len(matched_treated),
        "Estimated CATE": cate_estimates,
        "True CATE": true_cate_values if true_cate in data.columns else None,
        "RMSE": rmse,
        "Bias": bias,
        "Variance": variance,
    }
    return results


if __name__ == '__main__':
    # Load and preprocess benchmark data
    from scripts.benchmark_data_scripts.data_preprocessing import load_data, preprocess_data

    training_data, ground_truth = load_data(path="data/benchmark_data")
    meta_learner_data = preprocess_data(training_data, ground_truth)

    # Define feature columns, outcome, treatment, and true CATE for each hypothesis
    features = ["IsCorrect", "AnswerValue", "CorrectAnswer", "QuestionSequence", "ConstructId"]
    outcome = "IsCorrect"
    treatment = "Treatment"

    # Hypothesis 1: CATE estimation
    hypothesis_1_results = calculate_psm(
        data=meta_learner_data,
        features=features,
        outcome=outcome,
        treatment=treatment,
        true_cate="TrueCATE_p",
        hypothesis_name="Hypothesis 1 (P)"
    )

    # Save results to CSV
    pd.DataFrame(hypothesis_1_results['Estimated CATE'], columns=["Estimated CATE"]).to_csv(
        'results/benchmark_data_results/psm_results/hypothesis_1_results.csv', index=False
    )

    # Print results
    print("Hypothesis 1 Results:")
    print(f"Matched Count: {hypothesis_1_results['Matched Count']}")
    print(f"Unmatched Count: {hypothesis_1_results['Unmatched Count']}")
    print(f"RMSE: {hypothesis_1_results['RMSE']}")
    print(f"Bias: {hypothesis_1_results['Bias']}")
    print(f"Variance: {hypothesis_1_results['Variance']}")

    # Hypothesis 2: CATE estimation
    hypothesis_2_results = calculate_psm(
        data=meta_learner_data,
        features=features,
        outcome=outcome,
        treatment=treatment,
        true_cate="TrueCATE_k",
        hypothesis_name="Hypothesis 2 (K)"
    )

    # Save results to CSV
    pd.DataFrame(hypothesis_2_results['Estimated CATE'], columns=["Estimated CATE"]).to_csv(
        'results/benchmark_data_results/psm_results/hypothesis_2_results.csv', index=False
    )

    # Print results
    print("Hypothesis 2 Results:")
    print(f"Matched Count: {hypothesis_2_results['Matched Count']}")
    print(f"Unmatched Count: {hypothesis_2_results['Unmatched Count']}")
    print(f"RMSE: {hypothesis_2_results['RMSE']}")
    print(f"Bias: {hypothesis_2_results['Bias']}")
    print(f"Variance: {hypothesis_2_results['Variance']}")
