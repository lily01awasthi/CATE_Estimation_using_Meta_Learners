from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression


def random_forest_model():
    return RandomForestRegressor()


def gradient_boosting_model():
    return GradientBoostingRegressor()


def logistic_regression_model():
    return LogisticRegression()
