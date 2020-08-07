import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sti import faststi

# Data from optimization
df = pd.read_csv('20200807-111925.csv')

#Y: dependent variable vector
X = df.iloc[:, 0:11].values
y = df.iloc[:,11:].values

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# Compare some predictions
tests = (12, 24, 53, 69)

for i in tests:
    from_state = np.array(df.iloc[i, 0:5])
    to_state = np.array(df.iloc[i, 5:10])
    sti = np.array(df.iloc[i, 11:])
    dls = np.array(df.iloc[i,11])

    sti_pred = reg_all.predict(X[i,:].reshape(1, -1)).flatten()

    print("Actual")
    faststi.print_sti(from_state, to_state, sti, dls)

    print("\nFrom regression")
    faststi.print_sti(from_state, to_state, sti_pred, dls)

