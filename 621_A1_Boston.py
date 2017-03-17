import csv
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor


def read_csv(file_name):                           # from DGB's github
	data = []
	targets = []
	with open(file_name, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		# Skip the first row. There may be a better way of doing this.
		header = True
		for row in lines:
			if header:
				header = False
			else:
				data.append(map(float, row[:-1]))
				targets.append(float(row[-1]))
	return data, targets


def linear(X0,X1,Y0):
    algo = linear_model.LinearRegression()
    algo.fit(X0,Y0)
    hypotheses = algo.predict(X1)
    return hypotheses

def KNN(X0,X1,Y0,Y1):
    k = 1
    algo = KNeighborsRegressor(n_neighbors=k)
    algo.fit(X0, Y0)
    hypotheses = algo.predict(X1)
    MSE = calcMSE(hypotheses, Y1)
    bestk = k
    while k < len(Y0)/len(Y1):
        k += 1
        newalgo = KNeighborsRegressor(n_neighbors=k)
        newalgo.fit(X0,Y0)
        newhyp = newalgo.predict(X1)
        if calcMSE(newhyp, Y1) < MSE:
            hypotheses = newhyp
            MSE = calcMSE(newhyp, Y1)
            bestk = k
    return hypotheses, bestk

def calcMSE(hyp, obs):               #there's a fxn for this in scikit
    n = len(hyp)
    sumdiff = 0
    for i in range(n):
        diff = hyp[i] - obs[i]
        sumdiff += diff**2
    meandiff = sumdiff/n
    return meandiff


bosData, bosTarg = read_csv('boston.csv')

train_x = bosData[:-50]
test_x = bosData[-50:len(bosData)]

train_y = bosTarg[:-50]
test_y = bosTarg[-50:len(bosTarg)]

lin_y = linear(train_x,test_x,train_y)

knn_y, kfinal = KNN(train_x,test_x,train_y,test_y)

print 'MSE for linear regression is ' + str(calcMSE(lin_y,test_y))
print 'MSE for K Nearest Neighbors with k = ' + str(kfinal) + ' is ' + str(calcMSE(knn_y,test_y))