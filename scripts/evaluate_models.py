import pandas as pd
from models.meta_learners import TLearner
from sklearn.metrics import mean_squared_error

# Load data and model
data = pd.read_csv('../data/dataset.csv')
model = TLearner()  # assuming the model is loaded or initialized

# Evaluate the model
predictions = model.predict(data.drop(['outcome'], axis=1), data['treatment'])
mse = mean_squared_error(data['outcome'], predictions)
with open('../results/model_performance.txt', 'w') as file:
    file.write(f'MSE: {mse}\n')
