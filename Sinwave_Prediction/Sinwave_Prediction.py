'''
Try sequence to sequence model next
'''

from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Activation, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time

np.random.seed(1)
global_start_time = time.time()

# Initialize hyper-parameters
epochs  = 5
seq_len = 100
tgt_len = 10
batch_size = 1

def read_data(filename):
    # Read from file
    print ('\n=== Reading data from file ===')
    df = pd.read_csv(filename, delimiter=',')
    res = df.values
    
    plt.plot(res, label='Data from file')
    plt.legend()
    plt.show()
    print ('File data shape:', res.shape)
    
    return df.values
 
def prepare_data(inputs, seq_len, batch_size):
   # Prepare input for LSTM format
   print ('\n=== Preparing data ===')
   data_gen = TimeseriesGenerator(inputs, inputs, length=seq_len+tgt_len, batch_size=batch_size)
   data_gen, _ = data_gen[0]
   
   print ('Data shape:', data_gen.shape)
   
   return data_gen

def scale_inputs(inputs, minval, maxval):
    # Scale data
    print ('\n=== Scaling inputs ===')
    data_scaled = []
    scaler = MinMaxScaler(feature_range=(minval, maxval))
    for window in inputs[:,:]:
        data_scaled.append(scaler.fit_transform(window))
    res = np.array(data_scaled)
    
    return res

def split_train_test(data, split):
    # Split data into train and test sets
    print ('\n=== Splitting data (', split*100, '% training ) ===')
    s_index = int(split * data.shape[0])
    X_train = data[:s_index, :-tgt_len]
    y_train = data[:s_index, -tgt_len:]
    X_test = data[s_index:, :-tgt_len]
    y_test = data[s_index:, -tgt_len:]
    
    return X_train, y_train, X_test, y_test

def reshape_inputs(inputs):
    return np.reshape(inputs, (inputs.shape[0], inputs.shape[1], inputs.shape[2]))

def reshape_outputs(outputs):
    return np.reshape(outputs, (outputs.shape[0], outputs.shape[1]))

def build_model():
    model = Sequential()
    model.add(LSTM(50, input_shape=(seq_len, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('linear'))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    #Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()
    
def predict_pred_y(model, data, seq_len, tgt_len):
    curr_frame = data[0]
    predicted = []
    for i in range(0, len(data), tgt_len):
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,:])
        curr_frame = curr_frame[tgt_len:]
        curr_frame = np.insert(curr_frame, [seq_len-tgt_len]*tgt_len, np.asarray(predicted[-1]).reshape(-1,1), axis=0)
    return predicted

def predict_true_y(model, data, window_size):
    predicted = []
    for i in range(len(data)):
        curr_frame = data[i]
        predicted.append(model.predict(curr_frame[np.newaxis,:,:])[0,:])
    return predicted

def print_model_history(history):
    print(history.history.keys())

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
input_data = read_data("C:/Users/Tan/Dropbox/Shared with Josiah/Code/Exercise Sin Wave/sinwave.csv")
data_gen = prepare_data(input_data, seq_len, input_data.shape[0])
#data_gen = scale_inputs(data_gen, -1, 1)
#np.random.shuffle(data_scaled)

X_train, y_train, X_test, y_test = split_train_test(data_gen, 0.95)

X_train = reshape_inputs(X_train)
y_train = reshape_outputs(y_train)
X_test = reshape_inputs(X_test)
y_test = reshape_outputs(y_test)

model = build_model()

history = model.fit(X_train, y_train, epochs=epochs)

print_model_history(history)

predicted = predict_pred_y(model, X_test, seq_len, tgt_len)
#predicted = predict_pred_y(model, X_test, seq_len)

predicted = np.array(predicted).reshape(-1,1)
#y_test = np.array(y_test).reshape(-1,1)
y_test_plot = y_test[:,1]
plot_results(predicted, y_test_plot)

model.save('LSTM_stateless')