# Create your first MLP in Keras
import pandas as pd
import numpy
target = "Snow Diff"
#m = horzcat(xlsread(filename, 'E28:E819'),xlsread(filename, 'L28:N819'),xlsread(filename, 'P28:R819'),xlsread(filename, 'T28:T819'),xlsread(filename, 'V28:X819'),xlsread(filename, 'Z28:Z819')).';
df = pd.read_excel('bosNew.xlsx', index_col=None, na_values=['NA'], names=['Year','Month', target, 'NAO', 'EA', 'WP', 'PNA', 'EA/WR', 'SCA', 'POL','ENSO', 'AO', 'PDO'], parse_cols = "A,B, H,L:N, P:V", skiprows=1,skip_footer=7)#:R819, T28:T819, V28:X819,Z28:Z819")
start = 48
num_p = 6
interval = 6
from keras.utils.np_utils import to_categorical
import numpy as np
#Next steps: Cross-validate, why is time so terrible, month one-hots, over-fitting?, categorical-years, try including less important indicators
y= []
y2 =[]
n = []
n2 = []
n3 = []
n4 = []
def one_hot(i):
    a = np.zeros(12)
    a[i] = 1
    return a
d = {'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7, 'September':8, 'October':9, 'November':10, 'December':11}
for i, j in df.loc[start+interval:,df.columns != target].iterrows():
    if int(j['Year']) <= 2010:
        y.append([])
        for q in range(i-start-interval, i-interval):
            y[-1]+=[df.loc[q][target], df.loc[q]['NAO'], df.loc[q]['PDO'], df.loc[q]['ENSO'], df.loc[q]['AO'], df.loc[q]['PNA']]
        n.append(one_hot(d[df.loc[i]['Month']]))
        n3.append([df.loc[i]['Year']])
    else:
        y2.append([])
        for q in range(i-start-interval, i-interval):
            y2[-1]+=[df.loc[q][target], df.loc[q]['NAO'], df.loc[q]['PDO'], df.loc[q]['ENSO'], df.loc[q]['AO'], df.loc[q]['PNA']]
        n2.append(one_hot(d[df.loc[i]['Month']]))
        n4.append([df.loc[i]['Year']])

print(n)
T = np.array(n)
yy = np.array(y)
print(T.shape, yy.shape, np.array(n3).shape)
#X = np.concatenate((yy,  np.array(n3)), axis = 1)
#X = np.transpose(yy, (0, 2, 1))
X = yy
#xT = np.concatenate(( np.array(n4),np.array(y2)), axis = 1)
#xT = np.transpose(np.array(y2), (0, 2, 1))
xT = np.array(y2)
#print(np.size(X))
Y = df.loc[start+interval:, df.columns == target].as_matrix()
Y, yT = Y[:len(y)], Y[len(y):]
print(X.shape, xT.shape)
print(Y.shape, yT.shape)
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
numpy.random.seed(7)
import keras as k
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, Flatten, Add
model = Sequential()


class Sinx(Activation):
    def __init__(self, activation, **kwargs):
        super(Sinx, self).__init__(activation, **kwargs)
        self.__name__ = 'sin'


def sin(x):
    return k.backend.sin(x)


get_custom_objects().update({'sin': Sinx(sin)})
class Cosx(Activation):
    def __init__(self, activation, **kwargs):
        super(Cosx, self).__init__(activation, **kwargs)
        self.__name__ = 'cos'


def cos(x):
    return k.backend.cos(x)


get_custom_objects().update({'cos': Cosx(cos)})
#model.add(Dense(40, input_dim=start*num_p, activation='sigmoid'))
s = Dense(units=start, activation='sin', input_dim=num_p*start, kernel_initializer="lecun_normal")
c =Dense(units=80, input_dim=start*num_p, activation='cos', kernel_initializer="lecun_normal")
first = Sequential()
first.add(s)
second = Sequential()
second.add(c)
#model.add(Add([first, second]))
model.add(s)
#model.add(Flatten())
#model.add(Dense(2*num_p*start, activation='sigmoid'))
#model.add(Dense(2*num_p*start, activation='relu'))
model.add(Dense(output_dim = 1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10)
scores = model.evaluate(X, Y)
print(scores)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.save('Boston-'+target.replace(' ','-')+'.h5')
predictions = model.predict(xT, verbose=1)
print(predictions)
print(yT)
from math import sqrt
m = ((predictions - yT) ** 2).mean(axis=None)
s = ((predictions - yT) ** 2).std(axis=None)
print([m-(2.6*s)/sqrt(len(y2)), m, m+(2.6*s)/sqrt(len(y2))])



