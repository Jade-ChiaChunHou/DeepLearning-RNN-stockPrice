###########################################
# Part 1 - Download data from Yahoo Finance
###########################################

import numpy as np
import pandas as pd

import yfinance as yf

import matplotlib.pyplot as plt

# training set
stock_train = yf.download("MSFT", "2014-01-02", "2019-08-30")
del stock_train['Adj Close']
stock_train.head()
stock_train.to_csv("microsoft_train_2019.csv")


# testset
stock_test = yf.download("MSFT", "2019-09-01", "2019-10-07")
del stock_test['Adj Close']
stock_test.head()
stock_test.to_csv("microsoft_test_2019.csv")

#############################
# Part 2 - Data Preprocessing
#############################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('microsoft_train_2019.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 988):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


##########################
# Part 3- Building the RNN
##########################


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#####################
# model 1: epoch = 50

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 50, batch_size = 32)





######################
# model 2: epoch = 100

# Initialising the RNN
regressor2 = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor2.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50, return_sequences = True))
regressor2.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor2.add(LSTM(units = 50))
regressor2.add(Dropout(0.2))

# Adding the output layer
regressor2.add(Dense(units = 1))

# Compiling the RNN
regressor2.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor2.fit(X_train, y_train, epochs = 100, batch_size = 32)



######################
# model 3: epoch = 150

# Initialising the RNN
regressor3 = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor3.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor3.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor3.add(LSTM(units = 50, return_sequences = True))
regressor3.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor3.add(LSTM(units = 50, return_sequences = True))
regressor3.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor3.add(LSTM(units = 50))
regressor3.add(Dropout(0.2))

# Adding the output layer
regressor3.add(Dense(units = 1))

# Compiling the RNN
regressor3.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor3.fit(X_train, y_train, epochs = 150, batch_size = 32)


######################
# model 4: epoch = 200

# Initialising the RNN
regressor4 = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor4.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor4.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor4.add(LSTM(units = 50, return_sequences = True))
regressor4.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor4.add(LSTM(units = 50, return_sequences = True))
regressor4.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor4.add(LSTM(units = 50))
regressor4.add(Dropout(0.2))

# Adding the output layer
regressor4.add(Dense(units = 1))

# Compiling the RNN
regressor4.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor4.fit(X_train, y_train, epochs = 200, batch_size = 32)




#############################################################
# Part 4 - Making the predictions and visualising the results
#############################################################


# Getting the real stock price of 2017
dataset_test = pd.read_csv('microsoft_test_2019.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 85):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#########################
# prediction from model 1

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


#########################
# prediction from model 2

predicted_stock_price2 = regressor2.predict(X_test)
predicted_stock_price2 = sc.inverse_transform(predicted_stock_price2)


#########################
# prediction from model 3

predicted_stock_price3 = regressor3.predict(X_test)
predicted_stock_price3 = sc.inverse_transform(predicted_stock_price3)


#########################
# prediction from model 4

predicted_stock_price4 = regressor4.predict(X_test)
predicted_stock_price4 = sc.inverse_transform(predicted_stock_price4)



# Visualising the results
fig = plt.figure(figsize=(50, 75))

plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = '#AED6F1', label = 'Predicted Stock Price50')
plt.plot(predicted_stock_price2, color = '#5DADE2', label = 'Predicted Stock Price100')
plt.plot(predicted_stock_price3, color = '#2874A6', label = 'Predicted Stock Price150')
plt.plot(predicted_stock_price4, color = '#1B4F72', label = 'Predicted Stock Price200')
plt.title('Microsoft Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

fig.savefig('msft_201909_50_100_150_200.png', dpi=fig.dpi)