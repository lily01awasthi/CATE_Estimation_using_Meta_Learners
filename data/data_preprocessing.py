import pandas as pd


def load_data(path):
    """Load dataset from a CSV file located in the specified directory."""
    return pd.read_csv(path)


def preprocess_data(path):
    """Preprocess data: handle missing values, encode categorical variables, etc."""
    dataset = load_data(path)
    data = dataset.rename(columns={
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

    return data


# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    path = 'dataset.csv'
    processed_data = preprocess_data(path)
    print(processed_data.columns)
