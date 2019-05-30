counties = ['Albany', 'Amherst', 'Birch Hill', 'Blue Hill', 'Boston', 'Concord', 'Edgartown', 'Providence', 'Tully Lake', 'Worcester']
#indicators = {i: {'Temp Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA'], 'Snow Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA'], 'Rain Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA']} for i in counties}
indicators = {i: {'Temp Diff': ['NAO', 'PDO', 'ENSO', 'AO', 'PNA']} for i in counties}
import pandas as pd
import numpy as np
for county in counties:
    df = pd.read_excel('Counties/' + county + '.xlsx', index_col=None, na_values=['NA'],
                       names=['Date', 'Temp Diff', 'Snow Diff', 'Rain Diff', 'NAO', 'EA', 'WP', 'PNA', 'EA/WR', 'SCA',
                              'POL', 'Expl', 'ENSO', 'AO', 'PDO'], parse_cols="A,D, G,J, K:U", skiprows=1,
                       skip_footer=0)  #:R819, T28:T819, V28:X819,Z28:Z819")
    interval = 6
    start = 36
    errors = []
    predictions = []
    vals = []
    for i, j in df.loc[start + interval:].iterrows():
        pred = []
        for q in range(i - start - interval, i - interval):
            if df.loc[q]['Date'].month == j['Date'].month:
                pred.append(df.loc[q]['Temp Diff'])
        predictions.append(sum(pred)/len(pred))
        vals.append(j['Temp Diff'])
    predictions = np.array(predictions)
    vals = np.array(vals)
    m = ((predictions - vals) ** 2).mean(axis=None)
    s = ((predictions - vals) ** 2).std(axis=None)
    from math import sqrt
    print(county + ': '+ str([m - (2.6 * s)), m, m + (2.6 * s) )]))