from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.Meta_learners_benchmark_data.meta_learner_models import X_learner_model

def x_learner(X_train, X_test, y_train):
    
    # Step 3: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.drop(columns=["Treatment"]))
    X_test_scaled = scaler.transform(X_test.drop(columns=["Treatment"]))

    control_model, treated_model = X_learner_model

    treated_model.fit(X_train_scaled[X_train["Treatment"] == 1], y_train[X_train["Treatment"] == 1])
    control_model.fit(X_train_scaled[X_train["Treatment"] == 0], y_train[X_train["Treatment"] == 0])

    # Calculate residuals 
    treated_outcomes_control = control_model.predict(X_train_scaled[X_train["Treatment"] == 1])
    control_outcomes_treated = treated_model.predict(X_train_scaled[X_train["Treatment"] == 0])

    tau_treated = y_train[X_train["Treatment"] == 1] - treated_outcomes_control
    tau_control = control_outcomes_treated - y_train[X_train["Treatment"] == 0]

    # Train refinement models 
    refinement_model_treated = Ridge(alpha=1.0)
    refinement_model_control = Ridge(alpha=1.0)

    refinement_model_treated.fit(X_train_scaled[X_train["Treatment"] == 1], tau_treated)
    refinement_model_control.fit(X_train_scaled[X_train["Treatment"] == 0], tau_control)

    # Predict refined treatment effects 
    tau_treated_refined = refinement_model_treated.predict(X_test_scaled)
    tau_control_refined = refinement_model_control.predict(X_test_scaled)

    # Estimate propensity scores 
    propensity_model = LogisticRegression()
    propensity_model.fit(X_train.drop(columns=["Treatment"]), X_train["Treatment"])
    propensity_scores = propensity_model.predict_proba(X_test.drop(columns=["Treatment"]))[:, 1]

    # Combine refined effects 
    x_learner_cate = propensity_scores * tau_treated_refined + (1 - propensity_scores) * tau_control_refined

    return x_learner_cate

