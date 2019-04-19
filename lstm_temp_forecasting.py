# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:31:37 2019

@author: crazyconda
"""


fname = 'jena_climate_2009_2016.csv'


f = open(fname)
data = f.read()
f.close()

len(data)

lines = data.split('\n')
len(lines)
print lines[0]

header = lines[0].split(',')
print header

lines = lines[1:]

print lines[0]

print type(lines[0])

#Let's convert all of these 420,551 lines of data into a Numpy array:
import numpy as np

float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

float_data.shape

print (float_data[0])

float_data.shape

#plot the tmeprature over time
from matplotlib import pyplot as plt

temp = float_data[:, 1]  # temperature (in degrees Celsius)
plt.plot(range(len(temp)), temp)
plt.show()

#Here is a more narrow plot of the first ten days of temperature data (since the data is recorded every ten minutes, we get 144 data points per day):

plt.plot(range(1440), temp[:1440])
plt.show()

"""## Preparing the data
The exact formulation of our problem will be the following: given data going as far back as lookback timesteps (a timestep is 10 minutes) and sampled every steps timesteps, can we predict the temperature in delay timesteps?

We will use the following parameter values:

lookback = 720, i.e. our observations will go back 5 days.
steps = 6, i.e. our observations will be sampled at one data point per hour.
delay = 144, i.e. our targets will be 24 hours in the future.
To get started, we need to do two things:


*   Preprocess the data to a format a neural network can ingest. This is easy: the data is already numerical, so we don't need to do any vectorization. However each timeseries in the data is on a different scale (e.g. temperature is typically between -20 and +30, but pressure, measured in mbar, is around 1000). So we will normalize each timeseries independently so that they all take small values on a similar scale.
*   Write a Python generator that takes our current array of float data and yields batches of data from the recent past, alongside with a target temperature in the future. Since the samples in our dataset are highly redundant (e.g. sample N and sample N + 1 will have most of their timesteps in common), it would be very wasteful to explicitly allocate every sample. Instead, we will generate the samples on the fly using the original data.


We preprocess the data by subtracting the mean of each timeseries and dividing by the standard deviation. We plan on using the first 200,000 timesteps as training data, so we compute the mean and standard deviation only on this fraction of the data:
"""

float_data[0,:] #print the first row/sample the data

mean = float_data[:200000].mean(axis=0)
std = float_data[:200000].std(axis=0)
print mean
print std

#normalize the data, perform standard scaling on first 200,000 records only, not the complete data



float_data -= mean

float_data /= std



mean = float_data[:200000].mean(axis=0)
std = float_data[:200000].std(axis=0)
print mean
print std

std

float_data[0]  #print the first record after feature scaling

"""## Building a Data Generator
Now here is the data generator that we will use. It yields a tuple (samples, targets) where samples is one batch of input data and  targets is the corresponding array of target temperatures. It takes the following arguments:



*   data: The original array of floating point data, which we just normalized in the code snippet above.
*   lookback: How many timesteps back should our input data go.
*   delay: How many timesteps in the future should our target be.
*   min_index and max_index: Indices in the data array that delimit which timesteps to draw from. This is useful for keeping a segment of the data for validation and another one for testing.
*   shuffle: Whether to shuffle our samples or draw them in chronological order.
*   batch_size: The number of samples per batch.
*   step: The period, in timesteps, at which we sample data. We will set it 6 in order to draw one data point every hour.
"""

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    count = 0
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            #print ("number of rows: " +  str(len(rows)))

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            #print ("number of indices: " +  str(len(indices)))

            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
            #print "samples", len(samples[j]), samples[j]
            #print "targets", targets[j]
            #print 'value of j is:', j
   
     
        count = count + 1
        #print "COUNT:",count
        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)

print next(train_gen)

#print train_gen.next() #same as above

"""Now let's use our abstract generator function to instantiate three generators, one for training, one for validation and one for testing. Each will look at different temporal segments of the original data: the training generator looks at the first 200,000 timesteps, the validation generator looks at the following 100,000, and the test generator looks at the remainder."""

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step, 
                      batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                     lookback=lookback,
                     delay=delay,
                     min_index=300001,
                     max_index=None,
                     step=step,
                     batch_size=batch_size)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps = (300000 - 200001 - lookback) // batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
test_steps = (len(float_data) - 300001 - lookback) // batch_size

print len(train_gen.next()[0]) #getting the batch size
print len(train_gen.next()[0][0]) #number of records each batch has

train_gen.next()

count = 0
for line in train_gen.next()[0][0]:
  print line
  count = count + 1

print count

"""## First Approach: ANN Model"""

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae', metrics = ['accuracy'])
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

float_data.shape

"""## First RNN - GRU Based"""

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

"""###Using recurrent dropout to fight overfitting"""

#the most succesful one

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.2,
                     recurrent_dropout=0.2,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)

"""### Stacking recurrent layers"""

#o stack recurrent layers on top of each other in Keras, 
#all intermediate layers should return their full sequence of outputs (a 3D tensor) rather than their output at the last timestep. 
#This is done by specifying return_sequences=True:

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32,
                     dropout=0.1,
                     recurrent_dropout=0.5,
                     return_sequences=True,
                     input_shape=(None, float_data.shape[-1])))
model.add(layers.GRU(64, activation='relu',
                     dropout=0.1, 
                     recurrent_dropout=0.5))
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=40,
                              validation_data=val_gen,
                              validation_steps=val_steps)


