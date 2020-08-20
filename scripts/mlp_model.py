import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import sti.sti_core
import pickle

# Data from optimization
filename_data ='data/merged/data.csv'

# Store the final model here for use later
filename_model = 'models/mlp.sav'

df = pd.read_csv(filename_data)

# Using standardized problem, so start state is 0
X = df.iloc[:, 5:11].values

# Intermediate points we'd like to predict
y = df.iloc[:,11:].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)


# network = (64, 96, 128, 128, 96, 64, 32)

# Network
depth = 5
width = depth * 10
condensation = int(width / 2)

network = [width] * depth
network.append(condensation)

print("MLP hidden arch:", network)

# Training epohcs
epochs = 1800

# Pipeline definition inc. scaling
model = Pipeline([
             ('scaler', StandardScaler()),
             ('mlp', MLPRegressor(hidden_layer_sizes=network, max_iter=epochs))
             ])

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = model.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(model.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Store the model
with open(filename_model, 'wb') as file:
    pickle.dump(model, file)

