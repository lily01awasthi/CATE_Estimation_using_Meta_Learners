
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

def staderize_categorical_and_numerical_features():
    # Preprocessing: Standardize continuous variables and one-hot encode categorical variables
    continuous_cols = ['PreInterventionFixedMindset', 'SchoolAchievementLevel', 'SchoolMinorityComposition', 
                    'PovertyConcentration', 'TotalStudentPopulation']

    categorical_cols = ['StudentRaceEthnicity', 'StudentGender', 'FirstGenCollegeStatus', 'SchoolUrbanicity']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )    
    return preprocessor

def preprocessor(df):


    dataset = rename_columns(df)

    covariate_cols, treatment_col, outcome_col = treatment_outcome_and_control()

    # Split data into training and testing sets (with stratification on treatment variable)
    # since the distribution of data across treated and control population is imbalanced
    train_data, test_data = split_train_test_data(dataset,treatment_col)

    preprocessor = staderize_categorical_and_numerical_features()

    # 6. Fit and transform the training data (covariates only)
    X_train_processed = preprocessor.fit_transform(train_data[covariate_cols])
    y_train = train_data[outcome_col].values
    treatment_train = train_data[treatment_col].values

    return X_train_processed, y_train, treatment_train, test_data


def load_data():
    # Load data
    df = pd.read_csv('data/dataset.csv')
    return df

# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    # Load  data
    df = load_data()
    print("data preprocessing .........")
    preprocessor(df)
    print("preprocessing done!")