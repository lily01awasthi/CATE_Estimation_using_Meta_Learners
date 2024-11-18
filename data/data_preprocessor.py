from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split


def rename_columns(df):
    # Rename columns for clarity
    dataset = df.rename(columns={
        'Y': 'StudentAchievementScore',
        'Z': 'GrowthMindsetIntervention',
        'S3': 'FutureSuccessExpectations',
        'C1': 'StudentRaceEthnicity',
        'C2': 'StudentGender',
        'C3': 'FirstGenCollegeStatus',
        'XC': 'SchoolUrbanicity',
        'X1': 'PreInterventionFixedMindset',
        'X2': 'SchoolAchievementLevel',
        'X3': 'SchoolMinorityComposition',
        'X4': 'PovertyConcentration',
        'X5': 'TotalStudentPopulation'
    })
    return dataset


def treatment_outcome_and_control():
    # Define covariates, treatment, and outcome
    covariate_cols = ['FutureSuccessExpectations', 'StudentRaceEthnicity', 'StudentGender', 'FirstGenCollegeStatus',
                      'SchoolUrbanicity', 'PreInterventionFixedMindset', 'SchoolAchievementLevel',
                      'SchoolMinorityComposition', 'PovertyConcentration', 'TotalStudentPopulation']
    treatment_col = 'GrowthMindsetIntervention'
    outcome_col = 'StudentAchievementScore'
    return covariate_cols, treatment_col, outcome_col


def split_train_test_data(dataset, treatment_col):
    # Split data into training and testing sets (with stratification on treatment variable)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset[treatment_col])
    return train_data, test_data


def standardize_continuous_features(train_data, test_data, continuous_cols):
    # Standardize continuous variables
    scaler = StandardScaler()
    train_data[continuous_cols] = scaler.fit_transform(train_data[continuous_cols])
    test_data[continuous_cols] = scaler.transform(test_data[continuous_cols])
    return train_data, test_data


def preprocessor(df):
    dataset = rename_columns(df)
    covariate_cols, treatment_col, outcome_col = treatment_outcome_and_control()

    # Split data into training and testing sets (with stratification on treatment variable)
    train_data, test_data = split_train_test_data(dataset, treatment_col)

    # Define column names for the transformed DataFrame
    continuous_cols = ['PreInterventionFixedMindset', 'SchoolAchievementLevel', 'SchoolMinorityComposition',
                       'PovertyConcentration', 'TotalStudentPopulation']

    # Standardize continuous features
    train_data, test_data = standardize_continuous_features(train_data, test_data, continuous_cols)

    # Extract covariates, treatment, and outcome
    X_train = train_data[covariate_cols]
    y_train = train_data[outcome_col]
    treatment_train = train_data[treatment_col]

    X_test = test_data[covariate_cols]
    y_test = test_data[outcome_col]
    treatment_test = test_data[treatment_col]

    return X_train, y_train, treatment_train, X_test, y_test, treatment_test


def load_data(path='data/analysis_data/dataset.csv'):
    # Load data
    df = pd.read_csv(path)
    return df


# Example usage
if __name__ == '__main__':
    # Load data
    df = load_data()
    print("Data preprocessing ...")
    X_train, y_train, treatment_train, X_test, y_test, treatment_test = preprocessor(df)
    print("Preprocessing done!")
    print("Training features preview:\n", X_train.head())
    print("Testing features preview:\n", X_test.head())
