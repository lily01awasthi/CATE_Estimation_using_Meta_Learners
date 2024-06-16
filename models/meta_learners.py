# In meta_learners.py
from models.models import random_forest_model
import numpy as np
from sklearn.base import clone


class TLearner:
    """TLearner uses separate models for treated and control groups to estimate treatment effects."""

    def __init__(self, base_estimator=None):
        """
                Initializes the T-Learner with specified base estimator models for treated and control groups.
                :param base_estimator: A sklearn-style estimator class  used for predictions.
                """
        # If no estimator provided, use RandomForestModel as default
        if base_estimator is None:
            base_estimator = random_forest_model
        # Create clones of the base estimator for treated and control groups
        self.model_treated = clone(base_estimator())  # Model for the treated group
        self.model_control = clone(base_estimator())  # Model for the control group

    def fit(self, X_train, treatment, y_train):
        """
                Fits separate models for treated and control data.
                :param X_train: Features dataframe
                :param treatment: Binary series indicating whether each observation was treated or not
                :param y_train: Outcome variable
                """

        # Fit the model for the treated group
        self.model_treated.fit(X_train[treatment == 1], y_train[treatment == 1])
        # Fit the model for the control group
        self.model_control.fit(X_train[treatment == 0], y_train[treatment == 0])

    def predict(self, X):
        """
               Predicts the outcome using both the treated and control models.
               :param X: Features dataframe on which to perform predictions
               :return: A tuple of predictions from the treated model and the control model
               """

        pred_treated = self.model_treated.predict(X)
        pred_control = self.model_control.predict(X)
        print(pred_treated, pred_control)
        return pred_treated, pred_control
