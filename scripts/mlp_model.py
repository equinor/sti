import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.neural_network import MLPRegressor
import sti.sti_core
import pickle

# Data from optimization
filename_data ='data/merged/data_boost.csv'

# Store the final model here for use later
filename_model = 'models/mlp-boost'

df = pd.read_csv(filename_data)

# Using standardized problem, so start state is 0
X = df.iloc[:, 5:11].values

# Intermediate points we'd like to predict
y = df.iloc[:,11:].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)


# Network
# Depth-width relationship based on https://arxiv.org/abs/2001.07523

# So far best model trained with 1000 epohcs
# MLP depth:  6
# MLP hidden arch: [48, 48, 48, 48, 48, 48, 24]
# R^2: 0.574738439348979
# Root Mean Squared Error: 452.36801092920155
for i in range(6, 9):
    depth = i
    width = depth * 8
    condensation0 = int(width / 2)

    network = [width] * depth
    network.append(condensation0)

    print("MLP depth: ", depth)
    print("MLP hidden arch:", network)

    # Training epohcs
    epochs = 1000

    # Pipeline definition inc. scaling
    model = Pipeline([
                 # ('scaler', StandardScaler()),
                 ('scaler', MinMaxScaler((-1,1))),
                 # ('poly', PolynomialFeatures(degree=2)),
                 ('mlp', MLPRegressor(hidden_layer_sizes=network, max_iter=epochs, verbose=True))
                 ])

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the test data: y_pred
    y_pred = model.predict(X_test)

    # Compute and print R^2 and RMSE
    print("R^2: {}".format(model.score(X_test, y_test)))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print("Root Mean Squared Error: {}".format(rmse))

    filename_this_model = filename_model + str(depth) + ".sav"

    # Store the model
    with open(filename_this_model, 'wb') as file:
        pickle.dump(model, file)

