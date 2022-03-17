# Deposit_Next_NN

# Import Packages
```
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import CSVLogger
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import random as python_random

pd.options.display.max_rows = 200 #sets the max rows head() shows.

SEED = 42
import os
import random as rn
import numpy as np
import tensorflow

def reset_seeds():
  os.environ['PYTHONHASHSEED']=str(SEED)
  np.random.seed(SEED)
  tensorflow.random.set_seed(SEED)
  rn.seed(SEED)
```

# Import Data
```
data = pd.read_csv('data.csv')
```

# Data Exploration
```
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[8, 4])
data.tenure.plot.box(ax=axes[0])
data.withdrawal.plot.box(ax=axes[1])
data.turnover.plot.box(ax=axes[2])
data.deposit.plot.box(ax=axes[3])

data.describe()

```
There are lots of outliers for each input feature. After creating a benchmarking model, I will train a model for the datapoints where the outliers of tenure are removed. (outlier=3*std)

![ ](/figures/figure1.png)

# Benchmark Model
First model will be trained by using all the datapoints available in the dataset.

* 4-layer NN was build by using fully connected layers.
* 3 dense hidden layers with 128-64-32 neurons
* 20,000 datapoints
* 33% of the datapoints is kept as validation set.
* Input layer consists of 4 neurons: ‘tenure’, ‘deposit’, ‘turnover’, ‘withdrawal’
* Activation functions were used to the layers: ‘relu’
* Adam algorithm was used to update the NN weights.
* Mean Absolute Error (MAE) was chosen to be the loss function. Thus, during the training MAE is aimed to minimize.

During the entire training, data will pass through the network many times. Each time it passes (1 epoch) the respective weight is updated. After each epoch we expect the loss function to decrease since the model will improve as the weights will be updated during the back-propagation.
In regression problems, there are several evaluation metrics. For the current problem and the dataset, I chose to use mean absolute error because I didn’t want large errors to create large biases on the evaluation. When calculating MAE, individual differences between the predicted and expected values have equal weight while in RMSE larger differences are penalized. Looking at the box. plots in the previous cell, it is seen that there are lots of outliers in our dataset. Thus, it will be more intuitive to use a metric that is robust to outliers, like MAE.

A very long epochs used in order to see the point after which the validation loss fluctuates with minor differences and become almost stable. The reason for this is that we aim to minimize the validation loss and achieve this in an optimal way by lowering the computational resources.

```
reset_seeds()

target = data.deposit_next
input_features = data[['tenure', 'deposit', 'turnover', 'withdrawal']]

scaler = StandardScaler()
scaled = scaler.fit_transform(input_features)

X_train, X_test, y_train, y_test = train_test_split(scaled, target, test_size=0.2, random_state=42)

model_b = Sequential()
model_b.add(Dense(128, input_shape=(4,), activation='relu'))
model_b.add(Dropout(0.2, input_shape=(128,))) 
model_b.add(Dense(64, activation='relu'))
model_b.add(Dense(32, activation='relu'))
model_b.add(Dense(1))
model_b.compile(optimizer='adam', loss='mae', metrics='mae')

csv_logger = CSVLogger('log.csv', append=True, separator=';')
history=model_b.fit(X_train,y_train,epochs=1000,batch_size=64,validation_data=[X_test,y_test],callbacks=[csv_logger])

```

***Visualizing epoch vs loss during the training:***
```
import seaborn as sns
loss = pd.read_csv('log.csv',  sep=';')
print(loss.head())

fig, ax = plt.subplots(figsize=(5,4))
ax = sns.lineplot(data=loss, x='epoch', y='mae', label='training')
ax = sns.lineplot(data=loss, x='epoch', y='val_mae', label='validation')
ax.grid()
ax.set_title('Benchmark Model')
ax.legend()

```
![ ](/figures/figure2.png)

Around epochs=230 we can stop the training since the validation loss reaches the minimum. If we train the model more than necessary there is the risk of overfitting as the model won’t be able to generalize.

```
model_b.fit(X_train,y_train,epochs=230,batch_size=64,validation_data=[X_test,y_test],callbacks=[csv_logger])
print('Final MAE of validation: %f' %(model_b.evaluate(X_test, y_test)[1]))
```

![ ](/figures/figure3.png)

# Feature Engineering
By using the features that we have, one can create new features. By comparing the deposit and deposit_next of the users, we can find out which users decided to put more money in their account than before or which users simply didn’t invest more in the next 30 days. 

Another information we can get would be if a user changed their mind to play online gambling. These people withdraw all the money they put in their first activity (tenure=0) without spending any (turnover=0) and not investing in the next 30 days (deposit_next=0). These people might not be essential for the training, however, I chose to keep them for now.

```
def get_label(deposit_next, deposit):
    if deposit_next==0:
        return 'no invest'
    elif deposit_next>deposit: 
        return 'go bigger' 
    elif deposit_next<deposit:
        return 'go easy'
    
def get_level(tenure):
    if tenure<=90:
        return 'beginner'
    elif tenure>90 and tenure<=180: 
        return 'preinter' 
    elif tenure>180 and tenure<=365:
        return 'inter'
    elif tenure>365:
        return 'advanced'

def changed_decision(deposit, turnover, withdrawal, deposit_next):
    if deposit == withdrawal and turnover==0 and deposit_next==0:
        return 'yes'
    else:
        return 'no'

def did_win(deposit, turnover):
    if deposit < turnover:
        return 'yes'
    else:
        return 'no' 
    
data['next_decision'] = data.apply(lambda x: get_label(x.deposit_next, x.deposit), axis=1)
data['changed_decision'] = data.apply(lambda x: changed_decision(x.deposit, x.turnover, x.withdrawal, x.deposit_next), axis=1)
data['won_min_once'] = data.apply(lambda x: did_win(x.deposit, x.turnover), axis=1)
data['level'] = data.apply(lambda x: get_level(x.tenure), axis=1)
```

I can use only 'won_min_once' and ‘level’ as a new input feature since the other two new columns were created by using the target.
```
data.head()
```

![ ](/figures/figure4.png)

## Label Encoding:
Since the new feature is a categorical feature, I will use label encoding to include its effect on the output.

I create a copy of the dataframe ‘data’ since I would like to keep it for legacy.

```
data_enc = data.copy()
data_enc['won_min_once'] = LabelEncoder().fit_transform(data_enc['won_min_once']) 
data_enc['level'] = LabelEncoder().fit_transform(data_enc['level']) 
data_enc.head()
```

Now I will train a new model by including the new feature that was encoded. Note that input layer will have *6 neurons* this time.

```
reset_seeds()

target = data_enc.deposit_next
input_features = data_enc[['tenure', 'deposit', 'turnover', 'withdrawal', 'won_min_once']]

scaled = scaler.fit_transform(input_features)
X_train, X_test, y_train, y_test = train_test_split(scaled, target, test_size=0.2, random_state=42)

model1 = Sequential()
model1.add(Dense(128, input_shape=(5,), activation='relu'))
model1.add(Dropout(0.5)) 
model1.add(Dense(64, activation='relu'))
model1.add(Dense(32, activation='relu'))
model1.add(Dense(1))
model1.compile(optimizer='adam', loss='mae', metrics='mae')

csv_logger = CSVLogger('log_enc.csv', append=True, separator=';')
history=model1.fit(X_train,y_train,epochs=400,batch_size=64,validation_data=[X_test,y_test],callbacks=[csv_logger])

```

```
import seaborn as sns
loss = pd.read_csv('log_enc.csv',  sep=';')
print(loss.head())

fig, ax = plt.subplots(figsize=(5,4))
ax = sns.lineplot(data=loss, x='epoch', y='mae', label='training')
ax = sns.lineplot(data=loss, x='epoch', y='val_mae', label='validation')
ax.set_title('after feature engineering')
ax.grid()
ax.legend()
```

![ ](/figures/figure5.png)

```
model1.fit(X_train,y_train,epochs=40,batch_size=64,validation_data=[X_test,y_test])
print('Final MAE of validation: %f' %(model1.evaluate(X_test, y_test)[1]))
```

![ ](/figures/figure6.png)
Performance of the model didn’t improve.

# Does removing the outliers in the tenure column improve the performance?
```
print(data.shape)
data_noout = data[data.tenure<data.std().tenure*3]
print(data_noout.shape)
```

![ ](/figures/figure7.png)

1094 datapoints were removed.
```
reset_seeds()

target = data_noout.deposit_next
input_features = data_noout[['tenure', 'deposit', 'turnover', 'withdrawal']]

scaled = scaler.fit_transform(input_features)

X_train, X_test, y_train, y_test = train_test_split(scaled, target, test_size=0.2, random_state=42)

model2 = Sequential()
model2.add(Dense(128, input_shape=(4,), activation='relu'))
model2.add(Dropout(0.5)) 
model2.add(Dense(64, activation='relu'))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mae', metrics='mae')

csv_logger = CSVLogger('log_noout.csv', append=True, separator=';')
history=model.fit(X_train,y_train,epochs=400,batch_size=64,validation_data=[X_test,y_test],callbacks=[csv_logger])
```

```
import seaborn as sns
loss = pd.read_csv('log_noout.csv',  sep=';')
print(loss.head())

fig, ax = plt.subplots(figsize=(5,4))
ax = sns.lineplot(data=loss, x='epoch', y='mae', label='training')
ax = sns.lineplot(data=loss, x='epoch', y='val_mae', label='validation')
ax.set_title('After Removing the Outliers in Tenure Column')
ax.grid()
ax.legend()

```
![ ] (/figures/figure8.png)

```
model2.fit(X_train,y_train,epochs=230,batch_size=64,validation_data=[X_test,y_test])
print('Final MAE of validation: %f' %(model2.evaluate(X_test, y_test)[1]))
```

Final MAE of validation: 454.693756
![ ](/figures/figure9.png)

Model improved.

# Comparing the 3 models:
```
loss3 = pd.read_csv('log_noout.csv',  sep=';')
loss2 = pd.read_csv('log_enc.csv',  sep=';')
loss1 = pd.read_csv('log.csv',  sep=';')

fig, ax = plt.subplots(figsize=(9,6))
ax = sns.lineplot(data=loss3, x='epoch', y='mae', label='training_3', color='b',linestyle="dashed")
ax = sns.lineplot(data=loss3, x='epoch', y='val_mae', label='validation', color='b')
ax = sns.lineplot(data=loss2, x='epoch', y='mae', label='training_2',color='r', linestyle="dashed")
ax = sns.lineplot(data=loss2, x='epoch', y='val_mae', label='validation', color='r')
ax = sns.lineplot(data=loss1, x='epoch', y='mae', label='training_1', color='k', linestyle="dashed")
ax = sns.lineplot(data=loss1, x='epoch', y='val_mae', label='validation', color='k')
ax.set_title('Comparison of three models')
ax.grid()
ax.set_xlim(0,250)
ax.legend()

```

![ ] (/figures/figure10.png)

END


