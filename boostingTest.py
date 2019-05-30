import xgboost as xgb
counties = ['Albany', 'Amherst', 'Birch Hill', 'Blue Hill', 'Boston', 'Concord', 'Edgartown', 'Providence', 'Tully Lake', 'Worcester']
#indicators = {i: {'Temp Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA'], 'Snow Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA'], 'Rain Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA']} for i in counties}
indicators = {i: {'Temp Diff': ['NAO', 'EA', 'WP', 'PNA', 'EA/WR', 'SCA',
                              'POL', 'Expl', 'ENSO', 'AO', 'PDO']} for i in counties}
#print(indicators)
from math import sqrt
T = indicators.copy()
x = []
import pandas as pd
import numpy
from pandas.core.window import rolling
year = 2008
u = {}
uu = []
for interval in range(12, 25, 6):
    print(str(interval) + ' years Lookahead time:')
    for year in range(2008, 1972, -7):
        print(str(year) + ': ' + str(year + 7))
        for county in counties:
            df = pd.read_excel('Counties/' + county + '.xlsx', index_col=None, na_values=['NA'],
                               names=['Date', 'Temp Diff', 'Snow Diff', 'Rain Diff', 'NAO', 'EA', 'WP', 'PNA', 'EA/WR', 'SCA',
                                      'POL', 'Expl', 'ENSO', 'AO', 'PDO'], parse_cols="A,D, G,J, K:U", skiprows=1,
                               skip_footer=0)  #:R819, T28:T819, V28:X819,Z28:Z819")

            for predictor in indicators[county].keys():

                #m = horzcat(xlsread(filename, 'E28:E819'),xlsread(filename, 'L28:N819'),xlsread(filename, 'P28:R819'),xlsread(filename, 'T28:T819'),xlsread(filename, 'V28:X819'),xlsread(filename, 'Z28:Z819')).';
                start = 96
                num_p = len(indicators[county][predictor])
                #interval = 6
                import numpy as np
                #Next steps: Cross-validate, why is time so terrible, month one-hots, over-fitting?, categorical-years
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

                for j in indicators[county][predictor]:
                    df[j+'MA'] = df[j].rolling(window=48,center=False).mean()
                    df[j + 'STD'] = df[j].rolling(window=start, center=False).std()
                d = {'January': 0, 'February': 1, 'March': 2, 'April': 3, 'May': 4, 'June': 5, 'July': 6, 'August': 7,
                     'September': 8, 'October': 9, 'November': 10, 'December': 11}
                predictions = []
                vals = []
                for i, j in df.loc[start + interval:].iterrows():
                    if int(j['Date'].year) <= year:
                        pred = []
                        y.append([])
                        for j in indicators[county][predictor]:
                            y[-1] += [df.loc[i - interval][j+'STD'], df.loc[i - interval][j+'MA']]
                        #print(y[-1])
                        #y[-1] += [df.loc[i-interval][q + 'STD'] for q in indicators[county][indicator]]
                    else:
                        pred = []
                        y2.append([])
                        for j in indicators[county][predictor]:
                            y2[-1] += [df.loc[i - interval][j+'STD'], df.loc[i - interval][j+'MA']]

                T = np.array(n)
                yy = np.array(y)
                # print(T.shape, yy.shape, np.array(n3).shape)
                # X = np.concatenate((yy,  np.array(n3)), axis = 1)
                # X = np.transpose(yy, (0, 2, 1))
                TTest = np.array(n2)
                X = yy
                # xT = np.concatenate(( np.array(n4),np.array(y2)), axis = 1)
                # xT = np.transpose(np.array(y2), (0, 2, 1))
                xT = np.array(y2, dtype=float)
                # print(np.size(X))
                Y = df.loc[start + interval:, df.columns == predictor].as_matrix()
                Y, yT = Y[:len(y)], Y[len(y):]
                # specify parameters via map
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression

                X =
                param = {'max_depth': 4, 'eta': .1, 'verbosity': 1, 'objective': 'reg:logistic'}
                num_round = 3
                #print(X.T.shape)
                #print(xT.T.shape)
                #print(len(indicators[county][predictor]))
                l = np.array(np.random.random(2*len(indicators[county][predictor])))

                #print(xT)
                #print(l)
                def f(y):
                    if y == 0 or y == None:
                        y= .00000001
                        #print(y)
                f = np.vectorize(f)
                #xT = f(xT)
                X = X.T
                xT = xT
                #X.sort_index(axis=1, inplace=True)
                #xT.sort_index(axis=1, inplace=True)
                dtrain = xgb.DMatrix(np.asmatrix(X), label=l)
                dtest = xgb.DMatrix(np.asmatrix(xT), label=l)
                bst = xgb.train(param, dtrain, num_round)

                # make prediction
                bst.save_model(county+'dump.raw.txt')
                bst = xgb.Booster(param)
                bst.load_model(county+'dump.raw.txt')
                predictions = bst.predict(dtest)
                import matplotlib.pyplot as plt
                #print(predictions)
                #plt = xgb.plot_importance(bst)
                #plt.figure.savefig(county+'Importance.png')
                #bst.dump_model(county+'dump.raw.txt', county+'featmap.txt')
                #xgb.plot_tree(bst, num_trees=2)
                u[county] = predictions
                uu.append(yT)
                #print(predictions.shape)
                #print(yT.shape)
                m = ((predictions - yT) ** 2).mean(axis=None)
                s = ((predictions - yT) ** 2).std(axis=None)
                print(county + ': ' + str([m - (2.6 * s) / sqrt(len(y2)), m, m + (2.6 * s) / sqrt(len(y2))]))
        '''
        ppp = 0
        for j in u.keys():
            predictions = u[j]
            yT = uu[ppp]
            m = ((predictions - yT) ** 2).mean(axis=None)
            s = ((predictions - yT) ** 2).std(axis=None)
            #print(j + ': ' + str([m - (2.6 * s) / sqrt(len(y2)), m, m + (2.6 * s) / sqrt(len(y2))]))
            ppp+=1
        import numpy as np
        from model import genJh
        from BP import MP
        from post_BP import produce
        L = 10
        J, h_ = genJh('Temp Diff')
        def full2(x, field, lam):
            r = h_ + lam*np.eye(L)@x
            z = np.matmul(np.linalg.inv(np.linalg.inv(J)+lam*np.eye(L)),r)
            return z
        com = []
        for j in counties:
            com.append(np.asmatrix(u[county]))
            #print(u[county].shape)
        com = np.array(com).T
        #print(com.shape)
        new = []
        for x in com:
            #print(x.shape)
            xSub = full2(x.T, 'Temp Diff', .01)
            new.append(xSub.tolist())
        new = np.array(new)
        #print(new.shape)
        new = new.reshape(new.shape[0], new.shape[1])
        new = new.T
        new = new.reshape(new.shape[0], new.shape[1], 1)
        p  = 0
        print(str(year) + ': '+ str(year+7))
        for predictions in new:
            m = ((predictions - uu[p]) ** 2).mean(axis=None)
            s = ((predictions - uu[p]) ** 2).std(axis=None)
            print(counties[p] + ': ' + str([m - (2.6 * s) / sqrt(len(y2)), m, m + (2.6 * s) / sqrt(len(y2))]))
            p+=1
        '''