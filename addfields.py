from impyute import em
import numpy as np
import xlrd
book = xlrd.open_workbook('Matlab Codes/Gaussian Graphical Models/Counties/allLocs.xlsx')
sheet = book.sheet_by_name('sheet1')
y = [[]]
for r in range(1,sheet.nrows):
	try:
		y[0].append(float(sheet.cell_value(r, 1)))
	except:
		y[0].append(np.nan)
import math
data = np.array(y).T
print(data)
n = em(data, loops=50)
p = n.T.tolist()[0]
import pandas as pd
sh = pd.read_excel('Matlab Codes/Gaussian Graphical Models/Counties/allLocs.xlsx', sheet_name='sheet1', index=False, na_values=[np.nan])
t = {}
p = list(sh.index.unique())
for j in p:
	t[j] = df.loc[j]
y = np.array([sh['TEMP'].as_matrix()])
print(y)
u ={}
for j in p:
	u[j] = np.array([t[j]['TEMP'].as_matrix()])
for j in p:
	y = u[j]
	if np.isnan(y.T):

	n = em(y.T, loops=50)
p = n.T.tolist()[0]
sh['TEMP'] = p
sh.to_excel('Matlab Codes/Gaussian Graphical Models/Counties/allLocs.xlsx', sheet_name='sheet1', index=False)
