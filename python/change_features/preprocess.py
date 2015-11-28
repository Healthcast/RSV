#!/usr/bin/pyhton


import csv
import numpy as np
import matplotlib.pyplot as plt





def load_hospital_data(data, address):
    dates = []
    hos = np.zeros(shape=(10,10), dtype=np.int)
    ys = np.zeros(shape=(10,10), dtype=np.int)
    city1 = []
    ss = []

    
    l=[]
    with open(address+"rsv.csv") as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)

    #remove data in August and September
#    newl=[]
#    for i in l:
#        d = i[''].split('-')
#        if int(d[1]) != 7 and int(d[1]) != 8 and int(d[1]) != 9:
#            newl.append(i)

    newl = l
    #insert date data
    for i in newl:
        dates.append(i[''])
        del i['']

    #load city names
    city1 = newl[0].keys()


    #init ylabel and date 
    hos = np.zeros(shape=(len(dates), len(city1)), dtype=np.int)
    ys = np.zeros(shape=(len(dates), len(city1)), dtype=np.int)


    #insert hospital data
    for i in range(len(newl)):
        hos[i] = [newl[i][x] for x in city1]

    #determine the start week of each season: Octorber 01
    for p in range(2009, 2016):
        start=0
        for j in range(len(dates)):
            d = dates[j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                ss.append(j)
                start = 1
#    ss = [0] + ss
    ss = ss + [len(dates)-1]

    #label and insert the "start week" to ylabel
    for j in range (len(city1)):
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
    data["city1"] = city1
    data["dates"] = dates
    data["season_start"] = ss











    
#filter all daye based data according to the "date" of hospital data
#......w1......w2......w3
#1234567 1234567 1234567
#calculate weekly weather data by accounting the prior 7 days weather data 
def filter_raw_data(dates,  address):

    l=[]
    with open(address) as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)

    newl=[]
    for x in range(len(l)):
        if l[x]['date'] in dates:
            for i in range(91):
               newl.append(l[x-20+i]) 

    return newl







#correct the non_available data (NaN and NA)
def correct_data(d):
    (r, c) = d.shape

    for i in d.T:
        #correct non_available data "-999" at the prefix of vector
        if int(i[0]) == -999:
            for j in range(1,r):
                if int(i[j]) != -999:
                    i[0:j] = np.zeros(shape=(j), dtype=np.float) + i[j]
                    break

        #correct non_available data "-999" at the sufffix of vector
        if int(i[r-1]) == -999:
            for j in range(r-2, -1, -1):
                if int(i[j]) != -999:
                    i[j+1:r] = np.zeros(shape=(r-j-1), dtype=np.float) + i[j]
                    break

        #correct non_available data "-999" in the middle of vector
        start=0
        end=0
        step_size=0.0
        step=0
        for j in range(0,r):
            if int(i[j]) == -999 and start ==0:
                    start = j-1
            if int(i[j]) != -999 and start != 0:
                end = j
            if start !=0 and end != 0:
                step = end - start - 1
                step_size = float(i[end] - i[start])/step
                for m in range(start+1, end):
                    i[m] = i[m-1]+step_size
                start = 0
                end = 0
                step = 0
                step_size=0
    return d



        
def load_weather_data(data, address):
    weather={}
    city2=[]
    dates = data["dates"]
    lnd = data["LND"]

    #refine weather data by "dates" in hospital data
    #AH --> AbsHumidity
    #T -- > temperature
    AH = filter_raw_data(data["dates"], address+"AbsHumidity.csv")
    T = filter_raw_data(data["dates"], address+"temperature.csv")
    

    # calculate the min city set that all data-set share
    city2 = set(AH[0].keys()) & set(T[0].keys())
    city2.remove("date")
    city2 = list(city2)

#    for i in range(len(dates)):
#        city2 = city2 & set(AH[i].keys()) & set(mAH[i].keys()) \
#                & set(mT[i].keys()) & set(T[i].keys())
#            
#    print "city2 after"



    #remove "NA" from the refined data
    r = len(AH)
    for i in range(r):
        for j in city2:
            if AH[i][j] == "NA" or AH[i][j] == "NaN":
                AH[i][j] = -999
            if T[i][j] == "NA" or T[i][j] == "NaN":
                T[i][j] = -999



    #initialize the matrix for the refined data
    ah = np.zeros(shape=(r, len(city2)), dtype=np.float)
    t = np.zeros(shape=(r, len(city2)), dtype=np.float)


    #insert weather data into matrix
    for i in range(r):
        ah[i] = [float(AH[i][x]) for x in city2]
        t[i] = [float(T[i][x]) for x in city2]

    #correct the non-available data, note by "-999"
    ah = correct_data(ah)
    t = correct_data(t)

    #compact the weather(humidity and temperature) data into 6 features
    #calculate week based weather data (average days data)


    for j in range(lnd/7):
        ah1 = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
        t1 = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
        for i in range(len(dates)):
            ah1[i, :] = sum(ah[((i+1)*lnd-7*(j+1)):(i+1)*lnd, :])/(7*(j+1))
            t1[i, :] = sum(t[((i+1)*lnd-7*(j+1)):(i+1)*lnd, :])/(7*(j+1))
        data["weather"].append(ah1)
        data["weather"].append(t1)

    data["city2"] = city2
    




#temperary function for testing all years' model at once time
#@rsv_id: city id in rsv table
#@weather_id: city id in weather table
def store_allXy(data, rsv_id, weather_id):
    dates = data["dates"]
    lnd = data["LND"]
    aXy={}
    #filter data given specific year
    for year in range(2009, 2015):
        start = 0
        end =0
        for i in range (len(dates)):
            d = dates[i].split('-')
            if int(d[0]) == year and d[1] == "08" and start == 0:
                start = i
            if int(d[0]) == year+1 and d[1] == "02" and end == 0:
                end = i
        X = np.zeros(shape=(end-start, lnd/7*2), dtype=np.float)
        y = np.zeros(shape=(end-start, 1), dtype=np.int)


        for i in range(lnd/7*2):
            X[:,i] = data["weather"][i][start:end, weather_id].copy()
        y = data["ylabels"][start:end, rsv_id].copy()


        #-----+----- => -----++++++ solving unblanced data
        for m in y:
            had_one=0
            for i in range (len(y)):
                if y[i] == 1 and had_one ==0:
                    had_one=1
                if had_one == 1:
                    y[i] = 1
        y = y*2-1    # 111000 => 111-1-1-1
        aXy[year] = [X,y]
    return aXy
    
    





#def remove_features(X, aXy):
#    deleted = range(NP/7*2)
#    X=np.delete(X, [1,2,4,5] ,1)
#    for year in range(2009, 2015):
#        aXy[year][0] = np.delete(aXy[year][0], [1,2,4,5] ,1)
#    return [X, aXy]
#



#This function will calculate X and y
#@paras: system parameters
#@data: data structure holding all data
#@address1, file address of hospital data
#@address2, file address of weather data

def load_data(paras, data, address):
    #load all datasets
    load_hospital_data(data, address)
    load_weather_data(data, address)


    city1 = data["city1"]
    city2 = data["city2"]
    allXy = data["allXy"]
    year = paras["year"]


    #retrieve X and y
    print set(city2) & set(city1)
    weather_id = city2.index(paras['city'])
    rsv_id = city1.index(paras['city'])

    aXy = store_allXy(data, rsv_id, weather_id)

    X = aXy[year][0]
    y = aXy[year][1]




#    [X, aXy] = remove_features(X, aXy)
    data["X"] = X
    data["y"] = y
    data["allXy"] = aXy






    ########################
    #per week one feature  --> solving time series
    #This approach doesnot work since: one point one dimension,  
    #there will be a clear hyperplane that almost determined by date info
    ########################
#    m =0
#    while m < len(dates):
#        a = np.zeros(shape=(len(dates), 1), dtype=np.int)
#        a = a - 1
#        a[m]=1
#        m+=1
#        X = np.concatenate((X, a), axis=1)            
#        
