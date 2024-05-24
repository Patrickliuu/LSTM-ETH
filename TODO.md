# TODO:
Please look at activation function (not that we use sigmoid for log return)
Please look at mse regularizations
T-Statistic auf LPD 
USE Log for LSTM instead of min_max
Explore tf.reduce_mean(gradients, axis=1) in compute_lpd_lstm_dataframe instead of reduce_sum
In the buy and hold I am not sure if I should transform the log back to original in the case of 
univariate and min max to original in case of LSTM.
In random Networks change the number of netz and the number of epochs




# Q&A
1-LSTM: Does it make sense to flatten for the regression: Take mean in linear regression plus take the sum in LPD
2- The following is our model settings in two different cases, do they make sense:


## Univariate (close spalte):
Apply logarithmic transformation to differences and use MinMaxScaler with a feature range of (-1, 1) and linear output for small neural networks, with RELU for hidden layers. The ReLU function maps (-inf,+inf) to (0,+inf).

Das Code:
def create_model(input_shape):
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


## Multivariate (Open, Volume, Close ,... Spalten):
Standardize data using (x-mean)/sigma and use linear output at the end of the LSTM. The LSTM layer in Keras uses tanh, which maps (-inf,+inf) to (-1,+1).

Das Code:
def build_lstm_model(input_data, neurons,
                     dropout=0.2, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    model.add(Activation('linear'))

    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(245)
window_len = 5
test_size = 0.2
zero_base = True
lstm_neurons = 50
epochs = 20
batch_size = 32
loss = 'mse'
dropout = 0.24
optimizer = 'adam'




3- Ist der Umfang genug? Xtreme Learning, regularization?
4- Implement some sort of early stoppage?