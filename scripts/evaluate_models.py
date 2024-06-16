import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, r2_score,  mean_absolute_error

# Load data and model
# Load predictions
predictions = pd.read_csv('../results/predictions.csv')
true_outcomes = pd.read_csv('../results/true_outcomes.csv')

# Evaluate the model
#mse = mean_squared_error(true_outcomes, predictions)
accuracy =  mean_absolute_error(true_outcomes['true_outcomes'],predictions['predicted_outcome'])
with open('../results/model_performance.txt', 'w') as file:
    file.write(f'MSE: {accuracy}\n')
