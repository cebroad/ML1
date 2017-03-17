### Taxi trip data clustering analysis
### MSAN 621 -- Machine Learning I
### Claire Broad
### December 11, 2016
### Argument 1: GoogleMaps API Key
### Argument 2: Taxi data file (use 'part0-green_tripdata_2016-03.csv' to replicate my results)
### Argument 3: GoogleMaps Javascript API Key

import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import KMeans, DBSCAN
import googlemaps
import sys
from collections import Counter
import time
from math import pi, cos, sin
from operator import itemgetter
import gmplot

def mapit(coordinates, type=None, cluster_num=None, entry_num=None):
    '''
    :param coordinates: list -- [latitude, longitude] for a single point
    :param type: type of result (for filename)
    :param cluster_num: cluster index
    :param entry_num: corresponds to order in analysis writeup
    :return: nothing (writes to html file)
    '''
    formatted_coords = ' {lat: %f, lng: %f}' % (coordinates[0], coordinates[1])
    mapstart = open('mapstart.txt', 'r').read()
    mapend = open('mapend.txt', 'r').read()
    mapend2 = open('mapend2.txt', 'r').read()
    APIKey = sys.argv[3]
    html = mapstart + formatted_coords + mapend + APIKey + mapend2
    filename = type + '_Cluster_' + str(cluster_num) + '_Entry_' + str(entry_num) + 'map.html'
    with open(filename, 'w') as file:
        file.write(html)


def locations(coordinates, type=None):
    '''
    Uses DBScan to cluster a set of points, prints details of the 5 most populous clusters to the console
    :param coordinates: list of coordinates - [latitude, longitude]
    :param type: optional input to determine what to print - 'Pickup' or 'Dropoff'
    :return: nothing (prints output to console)
    '''
    if type == 'Pickup':
        print '\n\t**Top pick-up locations**'
    elif type == 'Dropoff':
        print '\n\t**Top drop-off locations**'
    start = time.time()
    db = DBSCAN(eps=0.001,min_samples=10)
    db.fit(coordinates)
    labels = db.labels_
    cluster_num = max(labels)
    end = time.time()
    # print end - start
    clusters = {}
    for i in range(len(labels)):
        clusters.setdefault(labels[i],[]).append(coordinates[i])
    # for item in clusters:
    #     print item, len(clusters[item])
    del clusters[-1]                                                         # ignore outliers
    cluster_sizes = []
    for i in range(cluster_num):
        cluster_sizes.append(len(clusters[i]))
    cluster_sizes = sorted(cluster_sizes)
    max_floor = cluster_sizes[-6]
    cluster_cover = 0
    i = 0
    for item in clusters:
        if len(clusters[item]) > max_floor:
            lats = [point[0] for point in clusters[item]]
            longs = [point[1] for point in clusters[item]]
            center = [np.mean(lats),np.mean(longs)]
            print '\n------------------------------------\n'
            print 'Latitude range: ' + str(max(lats) - min(lats))
            print '\t',min(lats),max(lats)
            print 'Longitude range: ' + str(max(longs) - min(longs))
            print '\t',min(longs),max(longs)
            g = gmaps.reverse_geocode(center)
            print g[0]['address_components'][2]['long_name'], center, len(clusters[item])
            cluster_cover += len(clusters[item])
            mapit(center, type='Volume_' + type, cluster_num = item, entry_num = i)
            i += 1
    print '\n------------------------------------\n'
    percent = float(cluster_cover)/len(coordinates)
    print 'Percentage of trips: ' + str(percent)

def time_period():
    '''
    Uses KMeans clustering to find the top 3 most popular times of day
    :return: nothing (prints results to console)
    '''
    print '\n\t**Time of day clustering**\n'
    # pickup_circle = [[cos((pi * n)/12),sin((pi * n)/12)] for n in pickup_times]
    cnum = 12
    kmeans = KMeans(n_clusters=cnum)
    kmeans.fit(pickup_times)
    labels = kmeans.labels_
    clusters = {}
    for i in range(len(labels)):
        clusters.setdefault(labels[i],[]).append(pickup_times[i])
    # for item in clusters:
    #     print item, len(clusters[item]), min(clusters[item]), max(clusters[item]), (max(clusters[item]) - min(clusters[item])), len(clusters[item])/(max(clusters[item]) - min(clusters[item]))
    cluster_sizes = []
    for i in range(cnum):
        cluster_sizes.append(len(clusters[i])/(max(clusters[i]) - min(clusters[i])))
    cluster_sizes = sorted(cluster_sizes)
    max_floor = cluster_sizes[-4]
    # print max_floor
    for item in clusters:
        if len(clusters[item])/(max(clusters[item]) - min(clusters[item])) > max_floor:
            print len(clusters[item]), min(clusters[item]), max(clusters[item])          # number of observations, start of period, end of period

def time_money(data, type=None):
    '''
    Uses DBScan to find clusters, then calculates top 5 most lucrative areas and prints details to console
    :param data: list of tuples of location + fare data -- each entry is of the form ([latitude, longitude], fare/minutes)
    :param type: type of data -- 'Pickup' or 'Dropoff'
    :return: nothing (prints results to console)
    '''
    if type == 'Pickup':
        print '\n\t**Money-for-time by pick-up location**'
    elif type == 'Dropoff':
        print '\n\t**Money-for-time by drop-off location**'
    locations = [item[0] for item in data]
    db = DBSCAN(eps=0.001,min_samples=100)
    db.fit(locations)
    labels = db.labels_
    cluster_num = max(labels)
    cluster_money = {}
    cluster_location = {}
    for i in range(len(labels)):
        cluster_money.setdefault(labels[i],[]).append(data[i][1])
        cluster_location.setdefault(labels[i], []).append(data[i][0])
    # for item in clusters:
    #     print item, len(clusters[item])
    del cluster_money[-1]
    money = [(key, np.mean(cluster_money[key])) for key in cluster_money.keys()]
    sorted_money = sorted(money, key = itemgetter(1))
    top_indices = sorted_money[-5:]
    i = 0
    for index in top_indices:
        # print index
        lats = [point[0] for point in cluster_location[index[0]]]
        longs = [point[1] for point in cluster_location[index[0]]]
        center = [np.mean(lats), np.mean(longs)]
        print '\n------------------------------------\n'
        print 'Latitude range: ' + str(max(lats) - min(lats))
        print '\t', min(lats), max(lats)
        print 'Longitude range: ' + str(max(longs) - min(longs))
        print '\t', min(longs), max(longs)
        g = gmaps.reverse_geocode(center)
        print g[0]['address_components'][2]['long_name'], center, len(cluster_location[index[0]])
        print str(index[1]) + ' dollars per minute'
        mapit(center, type='Money_' + type, cluster_num=index[0], entry_num=i)
        i+=1

if __name__ == '__main__':
    googlemaps_key = sys.argv[1]
    gmaps = googlemaps.Client(key=googlemaps_key)
    pickup_coordinates = []
    pickup_long = []
    pickup_lat = []
    dropoff_coordinates = []
    dropoff_long = []
    dropoff_lat = []
    pickup_times = []
    money_pickup = []
    money_dropoff = []
    with open(sys.argv[2], 'r') as taxi:
        taxireader = csv.reader(taxi)
        i = 0
        for line in taxireader:
            if i > 0:
                try:
                    if float(line[5]) and float(line[6]) != 0:
                        pickup_coordinates.append([float(line[6]),float(line[5])])
                        pickup_long.append(float(line[5]))
                        pickup_lat.append(float(line[6]))
                except:
                    pass
                try:
                    if float(line[7]) and float(line[8]) != 0:
                        dropoff_coordinates.append([float(line[8]),float(line[7])])
                        dropoff_long.append(float(line[7]))
                        dropoff_lat.append(float(line[8]))
                except:
                    pass
                try:
                    dt = datetime.strptime(line[1], '%Y-%m-%d %H:%M:%S')
                    # print dt.time()
                    time_num = float(dt.hour) + (float(dt.minute)/60) + (float(dt.second)/3600)
                    # print time_num
                    pickup_times.append(time_num)
                except:
                    pass
                try:
                    pickup_time = datetime.strptime(line[1], '%Y-%m-%d %H:%M:%S')
                    dropoff_time = datetime.strptime(line[2], '%Y-%m-%d %H:%M:%S')
                    elapsed = dropoff_time - pickup_time
                    minutes = (elapsed.total_seconds()/60)
                    money_per_min = float(line[11])/minutes
                    if float(line[5]) and float(line[6]) != 0:
                        money_pickup.append(([float(line[6]),float(line[5])], money_per_min))
                    if float(line[7]) and float(line[8]) != 0:
                        money_dropoff.append(([float(line[8]),float(line[7])], money_per_min))
                except:
                    pass
            i += 1
    pickup_times = np.array(pickup_times).reshape(-1,1)
    print 'Observations: ' + str(i)
    locations(pickup_coordinates, type = 'Pickup')              # Uncomment for Problem 2
    locations(dropoff_coordinates, type = 'Dropoff')            # Uncomment for Problem 3
    time_period()                                               # Uncomment for Problem 4
    time_money(money_pickup, type = 'Pickup')                   # Uncomment for Problem 5
    time_money(money_dropoff, type = 'Dropoff')                 # Uncomment for Problem 6
