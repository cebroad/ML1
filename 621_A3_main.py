import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import normalize
import sys
import re

def linimp(years, vals, num):
    '''
    Use linear regression to predict the value of one missing point
    NOT USED because subsequent observations could be predicted using previous predictions
    :param years: list of all years theoretically having observations
    :param vals: list of all observations -- missing observations are ''
    :param num: index of the missing observation to predict
    :return: predicted value
    '''
    valyr = []
    valsp = []
    for i in range(len(years)):
        if vals[i] != '':
            valyr.append([years[i]])
            valsp.append([float(vals[i])])
    valyr = np.array(valyr)
    valsp = np.array(valsp)
    valyr = valyr.astype(np.int)
    valsp = valsp.astype(np.float)
    linreg = LinearRegression()
    linreg.fit(valyr, valsp)
    pred = linreg.predict(int((years[num])))
    return pred

def linimp2(years, data):
    '''
    Use one linear regression to fill in all missing values at once
    :param years: list of all years theoretically having observations
    :param data: list of all observations -- missing observations are ''
    :return: list of all data (observations and predictions) in order of year
    '''
    valyr = []
    valsp = []
    misyr = []
    for i in range(len(years)):
        if data[i] != '':
            valyr.append([years[i]])
            valsp.append([float(data[i])])
        elif data[i] == '':
            misyr.append([years[i]])
    valyr = np.array(valyr)
    valsp = np.array(valsp)
    valyr = valyr.astype(np.int)
    valsp = valsp.astype(np.float)
    linreg = LinearRegression()
    try:
        linreg.fit(valyr, valsp)
        for i in range(len(data)):
            if data[i] == '':
                data[i] = float(linreg.predict(int(years[i])))
    except:
        pass
    return data

def pulldata(filename, addyears = False, countries = None):
    '''
    :param filename: csv file with header -- columns are years, rows are countries
    :return: dictionary of tuples (country, value) indexed by year, list of countries, list of years; country list
    '''
    vals = []
    labels = []
    valdict = {}
    with open(filename, 'r') as exp:
        data = csv.reader(exp)
        i = 0
        for line in data:
            i += 1
            if i == 1:
                pass
            else:
                if addyears == True:
                    for j in range(11):
                        line.insert(1,'')
                    for k in range(6):
                        line.append('')
                label = line[0]
                if re.search('Korea\, Dem', label) != None:
                    label = 'North Korea'
                labels.append(label)
                country = line[1:]
                country = linimp2(years, country)
                #for i in range(len(country)):
                #    if country[i] == '':
                #        country[i] = linimp(years, country, i)
                #    else:
                #        country[i] = float(country[i])
                vals = vals + country
                for j in range(len(years)):
                    valdict.setdefault(years[j], []).append([label, country[j]])
    if addyears == False:
        for country in countries:
            if country not in labels:
                vals = vals + ['']*len(years)
                for i in range(len(years)):
                    valdict.setdefault(years[i], []).append([country, ''])
    return valdict, labels

def features(spdict, country_year):
    '''
    feature extraction -- each row corresponds to one year in one country (in the order of country_year)
    :param spdict: span dictionary -- dictionary indexed by year of tuples (country, life expectancy)
    :param country_year: list of tuples (country, year) with all possible combinations
    :return: feature matrix, list of life expectancies corresponding to the rows of X by country/year
    '''
    X = []
    Y = []
    for cy in country_year:
        country = cy[0]
        year = cy[1]
        i = 0
        try:
            for item in spdict[year]:
                if item[0] == country:
                    if item[1] != '':
                        Y.append(item[1])
                    else:
                        Y.append(0)
                    i += 1
        except:
            Y.append(0)
        if i == 0:
            Y.append(0)
        feat = []
        i = 0
        '''
        for item in births[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(-5000)
                i += 1
        if i == 0:
            feat.append(float(-5000))
        else:
            i = 0
        '''
        for item in deaths[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(-5000)
                i += 1
        if i == 0:
            feat.append(float(-5000))
        else:
            i = 0
        '''
        for item in HIV[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(0)
                i += 1
        if i == 0:
            feat.append(0)
        else:
            i = 0
        for item in income[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(-5000)
                i += 1
        if i == 0:
            feat.append(float(-5000))
        else:
            i = 0
        for item in internet[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(0)
                i += 1
        if i == 0:
            feat.append(0)
        else:
            i = 0
        '''
        for item in fertility[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(-5000)
                i += 1
        if i == 0:
            feat.append(-5000)
        else:
            i = 0
        for item in inf_mort[year]:
            if item[0] == country:
                if item[1] != '':
                    feat.append(item[1])
                else:
                    feat.append(-5000)
                i += 1
        if i == 0:
            feat.append(-5000)
        else:
            i = 0
        for item in countries:
            if country == item:
                feat.append(1)
            else:
                feat.append(0)
        X.append(feat)
    X = np.array(X)
    X = X.astype(np.float)
    imp = Imputer(missing_values = -5000)
    X = imp.fit_transform(X)
    normalize(X, copy = False, axis = 0)
    Y = map(float, Y)
    print X
    return X, Y

def training():
    '''
    Trying different models to find the best MSE
    :return: nothing
    '''
    X, Y = features(spans, cy_yr)
    X_train1 = X[:-2020]
    X_test1 = X[-2020:]
    Y_train1 = Y[:-2020]
    Y_test1 = Y[-2020:]
    X_train2 = X[2020:4040]
    X_test2 = X[4040:]
    Y_train2 = Y[2020:4040]
    Y_test2 = Y[4040:]
    linreg1 = LinearRegression()
    linreg1.fit(X_train1, Y_train1)
    pred = linreg1.predict(X_test1)
    print 'Linear Regression MSE (future): ' + str(mean_squared_error(Y_test1, pred))
    linreg2 = LinearRegression()
    linreg2.fit(X_train2, Y_train2)
    pred = linreg2.predict(X_test2)
    print 'Linear Regression MSE (past): ' + str(mean_squared_error(Y_test2, pred))
    for depth in range(1,25):
        tree1 = DecisionTreeRegressor(max_depth = depth)
        tree1.fit(X_train1,Y_train1)
        pred = tree1.predict(X_test1)
        print str(depth) + ' Tree MSE (future): ' + str(mean_squared_error(Y_test1, pred))
        tree2 = DecisionTreeRegressor(max_depth = depth)
        tree2.fit(X_train2, Y_train2)
        pred = tree2.predict(X_test2)
        print str(depth) + ' Tree MSE (past): ' + str(mean_squared_error(Y_test2, pred))
    for estimators in range(1,25):
        forest1 = RandomForestRegressor(n_estimators = estimators)
        forest1.fit(X_train1, Y_train1)
        pred = forest1.predict(X_test1)
        print str(estimators) + ' Forest MSE (future): ' + str(mean_squared_error(Y_test1, pred))
        forest2 = RandomForestRegressor(n_estimators = estimators)
        forest2.fit(X_train2, Y_train2)
        pred = forest2.predict(X_test2)
        print str(estimators) + ' Forest MSE (past): ' + str(mean_squared_error(Y_test2, pred))

def test():
    '''
    Read in csv of test dates/countries/GDPs
    Run the winning model
    Write predictions to the output file
    :return: nothing
    '''
    mydata = []
    X_train, Y_train = features(spans, cy_yr)
    with open(sys.argv[1], 'r') as testdata:
        data = csv.reader(testdata)
        for line in data:
            mydata.append([line[0], line[1]])
    X_test, Y_test = features(spans, mydata)
    '''tree = DecisionTreeRegressor(max_depth = 15)
    tree.fit(X_train, Y_train)
    pred = tree.predict(X_test)'''
    forest = RandomForestRegressor(n_estimators = 4)
    forest.fit(X_train, Y_train)
    pred = forest.predict(X_test)
    output = open(sys.argv[2], 'w')
    for item in pred:
        output.write(str(item) + '\n')

def debug():
    test_countries = []
    with open(sys.argv[1], 'r') as testdata:
        data = csv.reader(testdata)
        for line in data:
            test_countries.append(line[0])
    print test_countries
    for item in test_countries:
        if item not in countries:
            print item

if __name__ == '__main__':
    years = range(1950, 2017)
    years = [str(year) for year in years]
    spans, countries = pulldata('life expectancy by country and year.csv', True)
    cy_yr = []
    for year in years:
        for country in countries:
            cy_yr.append([country, year])
    births = pulldata('birth_rate.csv', countries=countries)[0]                                       # birth rate
    deaths = pulldata('death_rate.csv', countries=countries)[0]                                       # death rate
    HIV = pulldata('HIV rates by country and year.csv', countries=countries)[0]                       # HIV rate
    income = pulldata('Income per capita by country and year.csv', countries=countries)[0]            # Income per capita
    internet = pulldata('Internet user rates by country and year.csv', countries=countries)[0]        # Internet user rate
    fertility = pulldata('fertility.csv', countries=countries)[0]
    inf_mort = pulldata('infant_mortality.csv', countries=countries)[0]
    test()