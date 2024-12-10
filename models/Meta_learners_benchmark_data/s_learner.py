from models.Meta_learners_benchmark_data.meta_learner_models import S_learner_model

def s_learner(X_train, X_test, y_train, treatment_col="Treatment"):
    # Ensure 'Treatment' is a column in X_train and X_test
    if treatment_col not in X_train.columns:
        raise ValueError(f"The column '{treatment_col}' must be present in X_train and X_test.")

    model = S_learner_model
    model.fit(X_train, y_train)

    # Step 3: Make predictions for treated and control groups
    # Create copies of X_test to modify Treatment column
    X_test_treated = X_test.copy()
    X_test_treated[treatment_col] = 1  # Set Treatment = 1 for treated group

    X_test_control = X_test.copy()
    X_test_control[treatment_col] = 0  # Set Treatment = 0 for control group

    # Ensure consistent feature order
    X_test_treated = X_test_treated[X_train.columns]
    X_test_control = X_test_control[X_train.columns]

    # Predictions for treated and control groups
    treated_outcomes = model.predict(X_test_treated)
    control_outcomes = model.predict(X_test_control)

    # Estimate CATE
    s_learner_cate = treated_outcomes - control_outcomes

    return s_learner_cate
