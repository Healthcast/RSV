#!/usr/bin/pyhton


import csv
import numpy as np
import matplotlib.pyplot as plt





def load_hospital_data(data, address):
    dates = []
    hos = np.zeros(shape=(10,10), dtype=np.int)
    ys = np.zeros(shape=(10,10), dtype=np.int)
    city = []
    ss = []

    
    l=[]
    with open(address+"rsv.csv") as csvfile:
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

    data["hospital"] = hos
    data["ylabels"] = ys
    data["city"] = city
    data["date1"] = dates
    data["season_start"] = ss












    
#filter weather data by date and city
def filter_raw_data(dates, city, address):

    l=[]
    with open(address) as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)
    newl = [x for x in l if x['date'] in dates]

    if len(newl) != len(dates):
        print "rsv data can not match to the weather data by date"

    #filter data by city
#    print (set(newl[0].keys()) & set(city))

    return newl












        
def load_weather_data(data, address):
    weather={}
    city2=[]
    dates = data["date1"]


    AH = filter_raw_data(data["date1"], data["city"], address+"AbsHumidity.csv")
    mAH = filter_raw_data(data["date1"], data["city"], address+"meanAbsHumidity.csv")
    mT = filter_raw_data(data["date1"], data["city"], address+"meanTemperature.csv")
    T = filter_raw_data(data["date1"], data["city"], address+"temperature.csv")
    

    # calculate the min city set that all data-set share
    city2 = set(AH[0].keys()) & set(mAH[0].keys()) & set(mT[0].keys()) \
            & set(T[0].keys())
    city2.remove("date")
    city2 = list(city2)
#    for i in range(len(dates)):
#        city2 = city2 & set(AH[i].keys()) & set(mAH[i].keys()) \
#                & set(mT[i].keys()) & set(T[i].keys())
#            
#    print "city2 after"



    #remove "NA" from data
    for i in range(len(dates)):
        for j in city2:
            if AH[i][j] == "NA" or AH[i][j] == "NaN":
                AH[i][j] = 0
            if mAH[i][j] == "NA" or mAH[i][j] == "NaN":
                mAH[i][j] = 0
            if mT[i][j] == "NA" or mT[i][j] == "NaN":
                mT[i][j] = 0
            if T[i][j] == "NA" or T[i][j] == "NaN":
                T[i][j] = 0



    #AH --> AbsHumidity
    #mah --> meanAbsHumidity
    #mt --> meanTemperature
    #t -- > temperature
    ah = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
    mah = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
    mt = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
    t = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)


    #insert hospital data
    for i in range(len(dates)):
        ah[i] = [float(AH[i][x]) for x in city2]
        mah[i] = [float(mAH[i][x]) for x in city2]
        mt[i] = [float(mT[i][x]) for x in city2]
        t[i] = [float(T[i][x]) for x in city2]


    data["city2"] = city2
    data["weather"]["ah"] = ah
    data["weather"]["mah"] = mah
    data["weather"]["mt"] = mt
    data["weather"]["t"] = t








#This function will calculate X and y
#@paras: system parameters
#@data: data structure holding all data
#@address1, file address of hospital data
#@address2, file address of weather data

def load_data(paras, data, address):
    #load all datasets
    load_hospital_data(data, address)
    load_weather_data(data, address)

    #retrieve X and y
    X = np.zeros(shape=(len(data["date1"]), 4), dtype=np.float)
    y = np.zeros(shape=(len(data["date1"]), 1), dtype=np.int)

    print set(data["city2"]) & set(data["city"])
    weather_id = data["city2"].index(paras['city'])
    rsv_id = data["city"].index(paras['city'])

    X[:,0] = data["weather"]["ah"].T[rsv_id]
    X[:,1] = data["weather"]["mah"].T[rsv_id]
    X[:,2] = data["weather"]["mt"].T[rsv_id]
    X[:,3] = data["weather"]["t"].T[rsv_id]

    y = data["ylabels"].T[rsv_id].copy()


    #-----+----- => -----++++++
    c = data["city"]
    dates = data["date1"]
    ss = data["season_start"]
    for m in range(len(ss)-1):
        had_one=0
        for i in range(ss[m], ss[m+1]-1):
            if y[i] == 1:
                had_one=1
            if had_one == 1:
                y[i] = 1

    
    y = y*2-1    # 111000 => 111-1-1-1

    #per week one feature
    m =0
    while m < len(dates):
        a = np.zeros(shape=(len(dates), 1), dtype=np.int)
        a[m]=1
        m+=1
        X = np.concatenate((X, a), axis=1)            
        

    data["X"] = X
    data["y"] = y

