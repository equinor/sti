import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import sti.sti_core as sti_core
import pickle

# DNN testing
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import EarlyStopping
import tensorflow as tf

# Scaling
from sklearn.preprocessing import StandardScaler

# Data from optimization
filename_data ='data.csv'

# Store the final model here for use later
filename_model = 'linear-mod.sav'

df = pd.read_csv(filename_data)

#Y: dependent variable vector
X = df.iloc[:, 0:11].values
y = df.iloc[:,11:].values

# Scale inputs and outputs
scaler_in = StandardScaler()
scaler_out = StandardScaler()
X = scaler_in.fit_transform(X)
y =scaler_out.fit_transform(y)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=42)

# DNN Code dump HACK HACK
predictors = X_train
target = y_train

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
# Note that input_shape=(n_cols,) signals that we do not specify number of items in training data.
model.add(Dense(50, activation='relu',input_shape=(n_cols,)))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(9))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Standardize predictors
# early_stopping_monitor = EarlyStopping(patience=2)

# Fit
model.fit(predictors, target, validation_split=0.3, epochs=2500)#, callbacks=[early_stopping_monitor])
# model.save('50DenseReluX50DenseReluX32DenseRelu.h5')


# Predict on the test data: y_pred

# loss, acc = model.evaluate(val_dataset)  # returns loss and metrics
loss = model.evaluate(x=X_test, y=y_test)
print("loss: %.2f" % loss)


# # Compare some predictions
# tests = range(1, 20000, 1000)

# for i in tests:
#     # from_state = np.array(df.iloc[i, 0:5])
#     # to_state = np.array(df.iloc[i, 5:10])

#     truth = df.iloc[i, 11:].values
#     sti = truth[0:]

#     states = np.array(df.iloc[i,0:11].values)
#     from_state = states[0:5]
#     to_state = states[5:10]
#     dls = states[10]

    
#     # This is already scaled
#     sti_pred = model.predict(X[i,:].reshape(1, -1)).flatten()

#     # Inverse scaling of output
#     sti_pred = scaler_out.inverse_transform(sti_pred)

#     print("Actual")
#     sti_core.print_sti(from_state, to_state, sti, dls)
#     print("\nFrom regression")
#     sti_core.print_sti(from_state, to_state, sti_pred, dls)


