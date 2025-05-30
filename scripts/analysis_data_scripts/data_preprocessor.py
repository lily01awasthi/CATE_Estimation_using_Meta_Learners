
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import  train_test_split

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

def standardize_categorical_and_numerical_features(continuous_cols):
    # Preprocessing: Standardize continuous variables and one-hot encode categorical variables

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols)
        ],
        remainder='passthrough'  # passthrough the remaining columns (if any)
    )   
    return preprocessor

from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def preprocessor(df, split_data=True):
    dataset = rename_columns(df)
    covariate_cols, treatment_col, outcome_col = treatment_outcome_and_control()

    # Define column names for the transformed DataFrame
    continuous_cols = ['PreInterventionFixedMindset', 'SchoolAchievementLevel', 'SchoolMinorityComposition', 
                       'PovertyConcentration', 'TotalStudentPopulation']
    categorical_cols = ['schoolid','StudentRaceEthnicity', 'StudentGender', 'FirstGenCollegeStatus', 'SchoolUrbanicity']

    preprocessor = standardize_categorical_and_numerical_features(continuous_cols)

    if split_data: 
        # Split data into training and testing sets (with stratification on treatment variable)
        train_data, test_data = split_train_test_data(dataset, treatment_col)

        # Fit and transform the training data
        X_train = preprocessor.fit_transform(train_data[covariate_cols])
        X_test = preprocessor.transform(test_data[covariate_cols])

        # Combine continuous and categorical feature names
        feature_names = list(continuous_cols) + list(categorical_cols)

        # Convert transformed arrays to DataFrames for better inspection and debugging
        X_train = pd.DataFrame(X_train, columns=feature_names)
        X_test = pd.DataFrame(X_test, columns=feature_names)

        # Convert transformed arrays to DataFrames for better inspection and debugging
        y_train = train_data[outcome_col].values
        treatment_train = train_data[treatment_col].values
        y_test = test_data[outcome_col].values
        treatment_test = test_data[treatment_col].values

        return X_train, y_train, treatment_train, X_test, y_test, treatment_test
    else:
        # Process the entire dataset
        X = preprocessor.fit_transform(dataset[covariate_cols])
        feature_names = list(continuous_cols) + list(categorical_cols)
        X = pd.DataFrame(X, columns=feature_names)

        y = dataset[outcome_col].values
        treatment = dataset[treatment_col].values

        return X, y, treatment

def load_data(path='data/analysis_data/dataset.csv'):
    # Load data
    df = pd.read_csv(path)
    return df

# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    # Load  data
    df = load_data()
    print("data preprocessing .........")
    X_train, y_train, treatment_train, X_test, y_test, treatment_test = preprocessor(df)
    print("preprocessing done!")
    print(X_train)
    print("Training features preview:\n", X_train.head())
    print("Testing features preview:\n", X_test.head())