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

#Y: dependent variable vector
X = df.iloc[:, 0:11].values
y = df.iloc[:,11:].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)


# MLP Pipeline
network = (128, 128, 96, 64, 32)
epochs = 900 #Hack
model = Pipeline([
             ('scaler', StandardScaler()),
             ('mlp', MLPRegressor(hidden_layer_sizes=network, max_iter=epochs))
             ])

# Fit the regressor to the training data
model.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = model.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(model.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# # Compare some predictions
# tests = range(1, 2500,5)

# for i in tests:
#     from_state = np.array(df.iloc[i, 0:5])
#     to_state = np.array(df.iloc[i, 5:10])
#     sti = np.array(df.iloc[i, 11:])
#     dls = np.array(df.iloc[i,11])

#     sti_pred = model.predict(X[i,:].reshape(1, -1)).flatten()

#     print("Actual")
#     faststi.print_sti(from_state, to_state, sti, dls)

#     print("\nFrom regression")
#     faststi.print_sti(from_state, to_state, sti_pred, dls)

# Store the model
with open(filename_model, 'wb') as file:
    pickle.dump(model, file)
