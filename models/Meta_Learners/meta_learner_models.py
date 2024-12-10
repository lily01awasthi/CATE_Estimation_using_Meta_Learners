from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge

# Define the model
#S-Learner: Random Forest
S_learner_model = RandomForestRegressor(
    max_depth=8,
    max_features='log2',
    min_samples_split=20,
    n_estimators=300,
    random_state=42
)

# T-Learner: AdaBoost 
T_learner_model_control = AdaBoostRegressor(
    learning_rate=0.5,
    n_estimators=200,
    random_state=42
)
T_learner_model_treated = AdaBoostRegressor(
    learning_rate=0.5,
    n_estimators=200,
    random_state=42
)
T_learner_model = (T_learner_model_control, T_learner_model_treated)

# X-Learner: AdaBoost
X_learner_model_treated = AdaBoostRegressor(
    learning_rate=0.5,
    n_estimators=100,
    random_state=42
)
X_learner_model_control = AdaBoostRegressor(
    learning_rate=0.5,
    n_estimators=100,
    random_state=42
)
X_learner_model = ( X_learner_model_control,X_learner_model_treated)

# R-Learner: Ridge Regression
R_learner_model = Ridge(alpha=1.0, random_state=42)
