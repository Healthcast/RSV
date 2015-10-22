#!/usr/bin/pyhton


import csv
import numpy as np
import matplotlib.pyplot as plt





def load_hospital_data(paras, data, address1):
    dates = data["date1"]
    hos = data["hospital"]
    ys = data["ylabels"]
    X = data["X"]
    y = data["y"]
    city = data["city"]
    ss = data["season_start"]

    
    l=[]
    with open(address1) as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)

    #remove data in August and September
    newl=[]
    for i in l:
        d = i[''].split('-')
        if int(d[1]) != 7 and int(d[1]) != 8 and int(d[1]) != 9:
            newl.append(i)

    #insert date data
    for i in newl:
        dates.append(i[''])
        del i['']

    #load city names
    city = newl[0].keys()


    #init ylabel and date 
    hos = np.zeros(shape=(len(dates), len(city)), dtype=np.int)
    ys = np.zeros(shape=(len(dates), len(city)), dtype=np.int)


    #insert hospital data
    for i in range(len(newl)):
        hos[i] = [newl[i][x] for x in city]

    #determine the start week of each season: Octorber 01
    for p in range(2009, 2016):
        start=0
        for j in range(len(dates)):
            d = dates[j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                ss.append(j)
                start = 1
    ss = [0] + ss
    ss = ss + [len(dates)-1]

    #label and insert the "start week" to ylabel
    for j in range (len(city)):
        for m in range(len(ss)-1):
            had_one=0
            for i in range(ss[m], ss[m+1]-1):
                if hos[i,j]>0 and hos[i+1,j] > hos[i,j] and hos[i+2,j] > hos[i,j] \
                   and had_one==0:
                    ys[i,j] = 1
                    had_one=1
                else:    ys[i,j] = 0
                
    X = np.zeros(shape=(len(dates), 2), dtype=np.int)
    city_id = city.index(paras['city'])
    X[:,0] = hos.T[city_id]
    X[:,1] =[x*x for x in hos.T[city_id]] 

    y = np.zeros(shape=(len(dates), 1), dtype=np.int)
    y = ys.T[city_id]

    data["hospital"] = hos
    data["X"] = X
    data["y"] = y
    data["ylabels"] = ys




#@data: data structure holding all data
#@address1, file address of hospital data
#@address2, file address of weather data

def load_data(paras, data, address1, address2):
    load_hospital_data(paras, data, address1)

