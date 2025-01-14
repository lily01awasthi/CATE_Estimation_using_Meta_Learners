import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict
from models.Meta_learners_benchmark_data.meta_learner_models import R_learner_model_k, R_learner_model_p

def r_learner(X_train, X_test, y_train, hypothesis):
    # Standardizing the features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=["Treatment"]))
    X_test_scaled = scaler.transform(X_test.drop(columns=["Treatment"]))

    # Training logistic regression model to estimate propensity scores 
    propensity_model = LogisticRegression(max_iter=1000, random_state=42)
    propensity_model.fit(X_train_scaled, X_train["Treatment"])

    # Predicting treatment probabilities (propensity scores) 
    e_hat = cross_val_predict(propensity_model, X_train_scaled, X_train["Treatment"], cv=5, method="predict_proba")[:, 1]

    # Training outcome model and calculating residuals 
    if hypothesis == "k":
        outcome_model = R_learner_model_k
    else:
        outcome_model = R_learner_model_p
    outcome_model.fit(X_train_scaled, y_train)

    y_hat = cross_val_predict(outcome_model, X_train_scaled, y_train, cv=5)

    y_residual = y_train - y_hat

    # Calculating treatment residuals (difference between treatment indicator and propensity score)
    t_residual = X_train["Treatment"] - e_hat

    # Handling potential division by zero
    t_residual = np.where(t_residual == 0, np.finfo(float).eps, t_residual)

    # Clip and scale residuals to handle extreme values 
    epsilon = 1e-3  # Threshold for clipping
    t_residual_clipped = np.clip(t_residual, epsilon, None)

    # Scale residuals using MinMaxScaler
    scaler_y = MinMaxScaler()
    scaler_t = MinMaxScaler()

    # Convert Series to NumPy array and reshape
    y_residual_scaled = scaler_y.fit_transform(y_residual.to_numpy().reshape(-1, 1)).flatten()
    t_residual_scaled = scaler_t.fit_transform(t_residual_clipped.reshape(-1, 1)).flatten()

    # Calculate residual target safely
    residual_target = y_residual_scaled / t_residual_scaled

    # Handle potential NaN, positive and negative infinities
    residual_target = np.nan_to_num(residual_target, nan=0.0, posinf=0.0, neginf=0.0)

    # Checking for extreme values
    residual_target = np.clip(residual_target, -1e6, 1e6)  # Clip excessively large values

    # Training R-Learner model 
    r_learner_model = Ridge(alpha=10.0)
    r_learner_model.fit(X_train_scaled, residual_target)

    # Predicting treatment effects on test set 
    r_learner_cate = r_learner_model.predict(X_test_scaled)

    return r_learner_cate

