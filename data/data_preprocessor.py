
# Key preprocessing steps 
"""
Key Preprocessing Steps:
Column Renaming:

Renamed columns for clarity, making the dataset more interpretable, particularly by giving descriptive names to variables related to student achievement, mindset intervention, and demographics.
Defining Covariates, Treatment, and Outcome:

Covariates: Includes student-level and school-level features, such as future expectations, race, gender, first-gen status, school characteristics (urbanicity, minority composition), and more.
Treatment: Whether the student received the growth mindset intervention (binary).
Outcome: The student achievement score post-intervention.
Train-Test Split:

Split the data into training and testing sets using stratification on the treatment variable, which is a good practice to handle imbalanced treatment groups.
Standardization and One-Hot Encoding:

Standardized continuous covariates (like school achievement level, minority composition, etc.) and one-hot encoded categorical variables (such as race, gender, and school urbanicity). This ensures that the machine learning models are fed properly scaled and encoded data.
"""
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

def standardize_categorical_and_numerical_features(continuous_cols, categorical_cols):
    # Preprocessing: Standardize continuous variables and one-hot encode categorical variables

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)  # drop='first' to avoid collinearity
        ]
    )    
    return preprocessor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def preprocessor(df):
    dataset = rename_columns(df)
    covariate_cols, treatment_col, outcome_col = treatment_outcome_and_control()

    # Split data into training and testing sets (with stratification on treatment variable)
    train_data, test_data = split_train_test_data(dataset, treatment_col)

    # Define column names for the transformed DataFrame
    continuous_cols = ['PreInterventionFixedMindset', 'SchoolAchievementLevel', 'SchoolMinorityComposition', 
                       'PovertyConcentration', 'TotalStudentPopulation']
    categorical_cols = ['StudentRaceEthnicity', 'StudentGender', 'FirstGenCollegeStatus', 'SchoolUrbanicity']

    preprocessor = standardize_categorical_and_numerical_features(continuous_cols, categorical_cols)

    # Fit and transform the training data
    X_train = preprocessor.fit_transform(train_data[covariate_cols])
    X_test = preprocessor.transform(test_data[covariate_cols])

    # Get categorical feature names from the OneHotEncoder
    ohe = preprocessor.named_transformers_['cat']
    
    # Check if get_feature_names_out is available; if not, fall back to get_feature_names
    try:
        categorical_feature_names = ohe.get_feature_names_out(categorical_cols)
    except AttributeError:
        # Fall back for older versions
        categorical_feature_names = ohe.get_feature_names(categorical_cols)

    # Combine continuous and categorical feature names
    feature_names = list(continuous_cols) + list(categorical_feature_names)

    # Convert transformed arrays to DataFrames for better inspection and debugging
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # Print column names and a sample of transformed data to inspect
    print("Encoded Column Names:", feature_names)

    # Convert transformed arrays to DataFrames for better inspection and debugging
    y_train = train_data[outcome_col].values
    treatment_train = train_data[treatment_col].values
    y_test = test_data[outcome_col].values
    treatment_test = test_data[treatment_col].values

    return X_train, y_train, treatment_train, X_test, y_test, treatment_test

def load_data(path='data/dataset.csv'):
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