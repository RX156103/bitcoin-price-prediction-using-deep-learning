#!/usr/bin/env python
# coding: utf-8

# ## Load libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# ## Load the dataset

# In[19]:


data = pd.read_csv("bitcoin_ticker.csv")


# In[20]:


data.head()


# In[21]:


data['rpt_key'].value_counts()


# ## Subset USD

# In[22]:


df = data.loc[(data['rpt_key'] == 'btc_usd')]


# In[23]:


df.head()


# ### Convert datetime_id to data type and filter dates greater than  2017-06-28 00:00:00

# In[24]:


df = df.reset_index(drop=True)
df['datetime'] = pd.to_datetime(df['datetime_id'])
df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]


# In[25]:


df = df[['datetime', 'last', 'diff_24h', 'diff_per_24h', 'bid', 'ask', 'low', 'high', 'volume']]


# In[26]:


df.head()


# ### we require only the last value, so we subset that and convert it to numpy array

# In[27]:


df = df[['last']]


# In[28]:


dataset = df.values
dataset = dataset.astype('float32')


# In[29]:


dataset


# Neural networks are sensitive to input data, especiallly when we are using activation functions like sigmoid or tanh activation functions are used. ISo we rescale our data to the range of 0-to-1, using MinMaxScaler

# In[30]:


scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)


# In[31]:


dataset


# In[32]:


train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))


# Now let us define the function called create_dataset, which take two inputs, 
# 
# 1. Dataset - numpy array that we want to convert into a dataset
# 2. look_back - number of previous time steps to use as input variables to predict the next time period
# 

# In[33]:


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
  dataX, dataY = [], []
  for i in range(len(dataset)-look_back-1):
    a = dataset[i:(i+look_back), 0]
    dataX.append(a)
    dataY.append(dataset[i + look_back, 0])
  return np.array(dataX), np.array(dataY)


# In[34]:


look_back = 10
trainX, trainY = create_dataset(train, look_back=look_back)
testX, testY = create_dataset(test, look_back=look_back)


# In[35]:


trainX


# In[36]:


trainY


# In[37]:


# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))


# ## Build our Model

# In[38]:


model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=256, verbose=2)


# In[39]:


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# We have to invert the predictions before calculating error to so that reports will be in same units as our original data

# In[40]:


trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# In[41]:



trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))


# In[42]:


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
 


# In[43]:


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict


# In[44]:


plt.plot(df['last'], label='Actual')
plt.plot(pd.DataFrame(trainPredictPlot, columns=["close"], index=df.index).close, label='Training')
plt.plot(pd.DataFrame(testPredictPlot, columns=["close"], index=df.index).close, label='Testing')
plt.legend(loc='best')
plt.show()


# In[ ]:





