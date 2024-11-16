from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def s_learner(X_train, X_test, y_train, y_test, treatment_col="Treatment"):
    
    # Initialize and train S-Learner model
    s_learner = GradientBoostingRegressor(random_state=42)
    s_learner.fit(X_train, y_train)

    # Step 3: Make predictions for treated and control groups
    # Predictions for treated group
    X_test_treated = X_test.copy()
    X_test_treated[treatment_col] = 1
    treated_outcomes = s_learner.predict(X_test_treated)

    # Predictions for control group
    X_test_control = X_test.copy()
    X_test_control[treatment_col] = 0
    control_outcomes = s_learner.predict(X_test_control)

    # Estimate CATE
    s_learner_cate = treated_outcomes - control_outcomes

    # Step 4: Evaluate using Mean Squared Error (MSE), Bias, and Variance
    mse = mean_squared_error(y_test, s_learner_cate)
    bias = np.mean(s_learner_cate - y_test)
    variance = np.var(s_learner_cate)

    return s_learner_cate, mse, bias, variance