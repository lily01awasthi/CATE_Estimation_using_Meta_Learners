from data.data_preprocessing import load_data, preprocess_data
from models.meta_learners import TLearner
import pandas as pd
from sklearn.model_selection import train_test_split


def train_and_predict(data_path):
    """
        Loads data, preprocesses it, splits it, trains a T-Learner, and saves predictions.
        :param data_path: Path to the CSV file containing the data.
        """
    # Load and preprocess the data
    data = load_data(data_path)
    data = preprocess_data(data)

    # Prepare data for modeling
    X = data.drop(['Y'], axis=1)  # Features
    y = data['Y']  # Outcome
    treatment = data['Z']  # Treatment indicator

    # treatment is the binary indicator used for stratification to ensure the training and testing sets contain a
    # similar mix of treated and control observations.
    # Split the data into training and testing sets, stratifying by treatment to maintain balance
    X_train, X_test, y_train, y_test, treatment_train, treatment_test = train_test_split(
        X, y, treatment, test_size=0.2, random_state=42, stratify=treatment)

    # Initialize and train the T-Learner
    t_learner = TLearner()
    t_learner.fit(X_train, treatment_train, y_train)

    # Predict the effects on the test set
    predicted_treated, predicted_control = t_learner.predict(X_test)

    # Save the predictions to a CSV file for further analysis
    predictions_df = pd.DataFrame({
        'Predicted Treated': predicted_treated,
        'Predicted Control': predicted_control
    })
    predictions_df.to_csv('../results/predictions.csv', index=False)


if __name__ == '__main__':
    train_and_predict('../data/dataset.csv')
