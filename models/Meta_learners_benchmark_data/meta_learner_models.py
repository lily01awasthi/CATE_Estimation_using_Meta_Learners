
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor


# Define the model
#S-Learner: Random Forest
S_learner_model = RandomForestRegressor(
    max_depth=8,
    max_features=None,
    min_samples_split=10,
    n_estimators=100,
    random_state=42
)

# T-Learner: Random Forest
T_learner_model_control = RandomForestRegressor(
    max_depth=None,
    max_features=None,
    min_samples_split=10,
    n_estimators=100,
    random_state=42
)

T_learner_model_treated = RandomForestRegressor(
    max_depth=None,
    max_features=None,
    min_samples_split=20,
    n_estimators=200,
    random_state=42
)
T_learner_model = (T_learner_model_control, T_learner_model_treated)

# X-Learner: Neural Network
# {'control_params': {'activation': 'tanh', 'hidden_layer_sizes': (100, 100), 'learning_rate_init': 0.01, 'solver': 'lbfgs'}, 'treated_params': {'activation': 'relu', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001, 'solver': 'lbfgs'}, 'cate_params': {'activation': 'tanh', 'hidden_layer_sizes': (100,), 'learning_rate_init': 0.001, 'solver': 'lbfgs'}}
X_learner_model_treated = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(100,),
    learning_rate_init=0.001,
    solver='lbfgs',
    random_state=42
)

X_learner_model_control = MLPRegressor(
    activation='tanh',
    hidden_layer_sizes=(100, 100),
    learning_rate_init=0.01,
    solver='lbfgs',
    random_state=42
)
X_learner_model = (X_learner_model_control, X_learner_model_treated)

# R-Learner: Hypothesis k AdaBoost
R_learner_model_k = AdaBoostRegressor(
    learning_rate=0.01,
    n_estimators=50,
    random_state=42
)
# R-Learner: Hypothesis p ExtraTrees
R_learner_model_p = ExtraTreesRegressor(
    max_depth=None,
    min_samples_split=2,
    n_estimators=100,
    random_state=42
)


