# Create your first MLP in Keras
import pandas as pd
import numpy
#m = horzcat(xlsread(filename, 'E28:E819'),xlsread(filename, 'L28:N819'),xlsread(filename, 'P28:R819'),xlsread(filename, 'T28:T819'),xlsread(filename, 'V28:X819'),xlsread(filename, 'Z28:Z819')).';
df = pd.read_excel('bos.xlsx', index_col=None, na_values=['NA'], names=['Year','Month', 'Temp Diff', 'NAO', 'EA', 'WP', 'PNA', 'EA/WR', 'SCA', 'POL', 'Expl', 'ENSO', 'AO', 'PDO'], parse_cols = "A,B, E,L:N, P:R, T, V:X, Z", skiprows=26,skip_footer=4)#:R819, T28:T819, V28:X819,Z28:Z819")
start = 24
num_p = 4
interval = 6
from keras.utils.np_utils import to_categorical
import numpy as np
#Next steps: Cross-validate, why is time so terrible, month one-hots, over-fitting?, categorical-years
y= []
n = []
def one_hot(i):
    a = np.zeros(12)
    a[i] = 1
    return a
d = {'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7, 'September':8, 'October':9, 'November':10, 'December':11}
for i, j in df.loc[start+interval:,df.columns != 'Temp Diff'].iterrows():
    y.append([])
    for q in range(i-start-interval, i-interval):
        y[-1] += [df.loc[q]['NAO'], df.loc[q]['PDO'], df.loc[q]['ENSO'], df.loc[q]['AO']]
    n.append(one_hot(d[df.loc[q]['Month']]))
print(n)
T = np.array(n)
yy = np.array(y)
print(T.shape, yy.shape)
#X = np.concatenate((yy, T), axis = 1)
X = yy
#print(np.size(X))
Y = df.loc[start+interval:, df.columns == 'Temp Diff'].as_matrix()
from keras.models import Sequential
from keras.layers import Dense
import numpy
numpy.random.seed(7)

model = Sequential()
model.add(Dense(num_p*start, input_dim=start*num_p, activation='sigmoid'))
model.add(Dense(num_p*start, activation='relu'))
#model.add(Dense(2*num_p*start, activation='sigmoid'))
#model.add(Dense(2*num_p*start, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


