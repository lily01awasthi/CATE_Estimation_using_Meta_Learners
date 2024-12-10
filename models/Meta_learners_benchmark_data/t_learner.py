from models.Meta_learners_benchmark_data.meta_learner_models import T_learner_model

def t_learner(X_train, X_test, y_train):

    control_model, treated_model = T_learner_model

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

    return t_learner_cate

