#!/usr/bin/pyhton


import csv
import numpy as np
import matplotlib.pyplot as plt



#@data: data structure holding all data
#@address1, file address of hospital data
#@address2, file address of weather data

def loading_data(paras, data, address1, address2):
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
        data["date1"].append(i[''])
        del i['']

    #load city names
    data["city"] = newl[0].keys()
    
    #init ylabel and date 
    data["hospital"] = np.zeros(shape=(len(data['date1']), \
                                       len(data["city"])), dtype=np.int)
    data["ylabels"] = np.zeros(shape=(len(data['date1']), \
                                       len(data["city"])), dtype=np.int)


    #insert hospital data
    for i in range(len(newl)):
        data["hospital"][i] = [newl[i][x] for x in data["city"]]

    #determine the start week of each season: Octorber 01
    for p in range(2009, 2016):
        start=0
        for j in range(len(data["date1"])):
            d = data["date1"][j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                data["season_start"].append(j)
                start = 1
    data["season_start"] = [0] + data["season_start"]
    data["season_start"] = data["season_start"] + [len(data["date1"])-1]

    #label and insert the "start week" to ylabel
    for j in range (len(data["city"])):
        for m in range(len(data["season_start"])-1):
            had_one=0
            for i in range(data["season_start"][m], data["season_start"][m+1]-1):
                if data["hospital"][i,j]>0 \
                       and data["hospital"][i+1,j] > data["hospital"][i,j] \
                       and data["hospital"][i+2,j] > data["hospital"][i,j] \
                       and had_one==0:
                    data["ylabels"][i,j] = 1
                    had_one=1
                else:    data["ylabels"][i,j] = 0
                
    data["X"] = np.zeros(shape=(len(data["date1"]), 2), dtype=np.int)
    city_id = data["city"].index(paras['city'])
    data["X"][:,0] = data["hospital"].T[city_id]
    data["X"][:,1] =[x*x for x in data["hospital"].T[city_id]] 

    data["y"] = np.zeros(shape=(len(data["date1"]), 1), dtype=np.int)
    data["y"] = data["ylabels"].T[city_id]
