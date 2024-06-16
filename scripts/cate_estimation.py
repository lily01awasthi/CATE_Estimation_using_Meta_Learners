import pandas as pd


def load_data(filename='../results/predictions.csv'):
    """Load dataset from a CSV file located in the specified directory."""
    # Construct the file path dynamically
    return pd.read_csv(filename)

