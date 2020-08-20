import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import sti.sti_core
import pickle

# Data from optimization
filename_data ='data/merged/data.csv'

# Store the final model here for use later
filename_model = 'models/linear.sav'

df = pd.read_csv(filename_data)

#Y: dependent variable vector
X = df.iloc[:, 5:11].values
y = df.iloc[:,11:].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

# Create the regressor: reg_all
# model = LinearRegression()
model = Pipeline([
             ('scaler', StandardScaler()),
             ('poly', PolynomialFeatures(degree=5)),
             ('ridge', Ridge(alpha=1.0))
             ])

# Fit the regressor to the training data
model.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = model.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(model.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Store the model
pickle.dump(model, open(filename_model, 'wb'))

