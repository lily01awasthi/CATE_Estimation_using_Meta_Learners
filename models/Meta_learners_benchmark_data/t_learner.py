import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

def t_learner(X_train, X_test, y_train, y_test):

    # Initialize T-Learner models 
    treated_model = GradientBoostingRegressor(random_state=42)
    control_model = GradientBoostingRegressor(random_state=42)

    # Train T-Learner models for treated and control groups 
    treated_model.fit(X_train[X_train["Treatment"] == 1].drop(columns=["Treatment"]), 
                        y_train[X_train["Treatment"] == 1])
    control_model.fit(X_train[X_train["Treatment"] == 0].drop(columns=["Treatment"]), 
                        y_train[X_train["Treatment"] == 0])

    # Predict outcomes for treated and control groups 
    treated_outcomes = treated_model.predict(X_test.drop(columns=["Treatment"]))
    control_outcomes = control_model.predict(X_test.drop(columns=["Treatment"]))

    # Estimate CATE 
    t_learner_cate = treated_outcomes - control_outcomes

    # Step 4: Evaluate using Mean Squared Error (MSE), Bias, and Variance 
    t_learner_mse = mean_squared_error(y_test, t_learner_cate)
    t_learner_bias = np.mean(t_learner_cate - y_test)
    t_learner_variance = np.var(t_learner_cate)

    return t_learner_cate, t_learner_mse, t_learner_bias, t_learner_variance

