import pandas as pd
import os

def load_data(filename='dataset.csv'):
    """Load dataset from a CSV file located in the specified directory."""
    # Construct the file path dynamically
    return pd.read_csv(filename)

def preprocess_data(df):
    """Preprocess data: handle missing values, encode categorical variables, etc."""
    # Forward fill to handle missing values
    df.fillna(method='ffill', inplace=True)
    # Encode 'treatment' as a category if it's one of the columns
    if 'treatment' in df.columns:
        df['treatment'] = df['treatment'].astype('category')
    return df

# Example usage within the same script, if run as a standalone for testing:
if __name__ == '__main__':
    # Example file name, change according to your dataset
    data = load_data('dataset.csv')
    processed_data = preprocess_data(data)
    print(processed_data.head())