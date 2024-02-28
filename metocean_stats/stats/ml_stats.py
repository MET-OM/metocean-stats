
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def predict_ts(ts_origin,var_origin, ts_train,var_train, model='LinearRegression'):
    """
    Input:
    ts_origin: pandas DataFrame
    var_origin: variable name (str)
    ts_train: pandas DataFrame
    var_train: variable name (str)
    Output:
    ts_pred: pandas DataFrame
    """
    Y_pred = ts_origin.data[var_origin]*0

    # Merge or join the dataframes based on time
    merged_data = pd.merge(ts_origin.data[var_origin], ts_train.data[var_train], how='inner', left_on='time', right_on='time')

    # Handling missing values if any
    merged_data = merged_data.dropna()
    # Extracting features and target variables
    X = merged_data[[var_origin+'_x']]
    Y = merged_data[var_train+'_y']

    # Splitting the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    # Creating and fitting the linear regression model
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import GradientBoostingRegressor
    if model == 'LinearRegression':
        model = LinearRegression()
    elif model == 'SVR_RBF':    
        model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    elif model == 'SVR_LINEAR':            
        model = SVR(kernel="linear", C=100, gamma="auto")
    elif model == 'SVR_POLY':    
        model = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    elif model == 'GBR':
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
    #elif model == 'LSTM':
        #from sklearn.preprocessing import MinMaxScaler
        #from keras.models import Sequential
        #from keras.layers import LSTM, Dense
        #scaler_X = MinMaxScaler()
        #scaler_Y = MinMaxScaler()
        #X_scaled = scaler_X.fit_transform(X)
        #Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))
        #X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)
        #model = Sequential()
        #model.add(LSTM(units=50, return_sequences=False))
        #model.add(Dense(units=1))
        #model.compile(optimizer='adam', loss='mean_squared_error')
        #history = model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

    model.fit(X_train, Y_train)

    # Predicted time series
    Y_pred[:] = model.predict(ts_origin.data[var_origin].values.reshape(-1, 1))

    return Y_pred