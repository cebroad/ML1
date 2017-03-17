import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer

def linimp(years, lifespans, num):
    valyr = []
    valsp = []
    for i in range(len(years)):
        if lifespans[i] != '':
            valyr.append([years[i]])
            valsp.append([float(lifespans[i])])
    valyr = np.array(valyr)
    valsp = np.array(valsp)
    valyr = valyr.astype(np.int)
    valsp = valsp.astype(np.float)
    linreg = LinearRegression()
    linreg.fit(valyr, valsp)
    pred = linreg.predict(int((years[num])))
    return pred

def linimp2(years, lifespans):
    valyr = []
    valsp = []
    misyr = []
    for i in range(len(years)):
        if lifespans[i] != '':
            valyr.append([years[i]])
            valsp.append([float(lifespans[i])])
        elif lifespans[i] == '':
            misyr.append([years[i]])
    valyr = np.array(valyr)
    valsp = np.array(valsp)
    valyr = valyr.astype(np.int)
    valsp = valsp.astype(np.float)
    linreg = LinearRegression()
    linreg.fit(valyr, valsp)
    for i in range(len(lifespans)):
        if lifespans[i] == '':
            lifespans[i] = float(linreg.predict(int(years[i])))
    print lifespans
    return lifespans

spans = []

with open('life expectancy by country and year.csv', 'r') as exp:
    data = csv.reader(exp)
    i = 0
    for line in data:
        i += 1
        if i == 1:
            years = line[1:]
        else:
            country = line[1:]
            country = linimp2(years, country)
            #for i in range(len(country)):
            #    if country[i] == '':
            #        country[i] = linimp(years, country, i)
            #    else:
            #        country[i] = float(country[i])
            spans = spans + country

spans = np.array(spans)

spans = spans.astype(np.float)

print spans

gdps = []

with open('GDP by country and year.csv', 'r') as gdp:
    data = csv.reader(gdp)
    i = 0
    for line in data:
        i += 1
        if i == 1:
            years = line[1:]
        else:
            country = line[1:]
            for i in range(len(country)):
                if country[i] == '':
                    country[i] = 0
            gdps.append(country)

yrs_gdps = []

for i in range(len(gdps)):
    country = gdps[i]
    for j in range(len(country)):
        yrs_gdps.append([years[j],country[j]])

X_train = np.array(yrs_gdps)

X_train = X_train.astype(np.float)

model = LinearRegression()
model.fit(X_train, spans)
print model.coef_