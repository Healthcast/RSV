#!/usr/bin/pyhton


import csv
import numpy as np
import matplotlib.pyplot as plt














def read_raw_data(address):

    l=[]
    with open(address) as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)

    return l






#remove duplicate cities: Berlin2, Essen2
def remove_duplicate_city(city1, hos):
    removed_ind=[]
    for i in range(len(city1)):
        if city1[i][-1] == "2":
            removed_ind.append(i)
            for j in range(len(city1)):
                if city1[i][:-1] == city1[j]:
                    hos[:,j] = hos[:,j] +hos[:,i]
    j=0                     
    for i in removed_ind:
        del city1[i-j]
        j+=1

    hos = np.delete(hos, removed_ind, 1)
    return [city1, hos]


def load_hospital_data(data, address):
    dates = data["dates"]
    hos = data["hospital"]
    ys = data["ylabels"]
    city1 = data["city1"]
    ss = data["season_start"]

    
    #read raw data from csv file
    l = read_raw_data(address+"rsv.csv")

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
    city1.sort()

    #init ylabel and hospital data matrix 
    hos = np.zeros(shape=(len(dates), len(city1)), dtype=np.int)
    ys = np.zeros(shape=(len(dates), len(city1)), dtype=np.int)


    #insert hospital data
    for i in range(len(newl)):
        hos[i,:] = [newl[i][x] for x in city1]


    #if one city have 2 hospital, add data together
    [city1, hos] = remove_duplicate_city(city1, hos)



    #determine the start week of each season: Octorber 01
    for p in range(2009, 2016):
        start=0
        for j in range(len(dates)):
            d = dates[j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                ss.append(j)
                start = 1
    ss = ss + [len(dates)-1]

    #label and insert the "start week" to ylabel
    for j in range (len(city1)):
        for m in range(len(ss)-1):
            had_one=0
            for i in range(ss[m], ss[m+1]-1):
                if hos[i,j]>0 and hos[i+1,j] > hos[i,j] and hos[i+2,j] > hos[i+1,j] \
                   and had_one==0:
                    ys[i,j] = 1
                    had_one=1
                else:    ys[i,j] = 0

    data["hospital"] = hos
    data["ylabels"] = ys
    data["city1"] = city1
    data["dates"] = dates
    data["season_start"] = ss





    





#correct the non_available data (-999 -> based on adjacent data)
def correct_data(d):
    (r, c) = d.shape

    for i in d.T:
        #correct non_available data "-999" as the prefix
        if int(i[0]) == -999:
            for j in range(1,r):
                if int(i[j]) != -999:
                    i[0:j] = np.zeros(shape=(j), dtype=np.float) + i[j]
                    break

        #correct non_available data "-999" as the sufffix
        if int(i[r-1]) == -999:
            for j in range(r-2, -1, -1):
                if int(i[j]) != -999:
                    i[j+1:r] = np.zeros(shape=(r-j-1), dtype=np.float) + i[j]
                    break

        #correct non_available data "-999" in the middle 
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
    weather=data["weather"]
    city2=data["weather"]
    dates = data["dates"]
    lnd = data["LND"]

    #AH --> AbsHumidity
    #T -- > temperature
    AH = read_raw_data(address+"AbsHumidity.csv")
    T = read_raw_data(address+"temperature.csv")
    

    # calculate the min city set that all data-set share
    city2 = set(AH[0].keys()) & set(T[0].keys())
    city2.remove("date")
    city2 = list(city2)
    city2.sort()


    #remove "NA" from the refined data NA -> -999
    r = len(AH)
    for i in range(r):
        for j in city2:
            if AH[i][j] == "NA" or AH[i][j] == "NaN":
                AH[i][j] = -999
            if T[i][j] == "NA" or T[i][j] == "NaN":
                T[i][j] = -999



    #initialize the matrix for holding weather data
    ah = np.zeros(shape=(r, len(city2)), dtype=np.float)
    t = np.zeros(shape=(r, len(city2)), dtype=np.float)


    #insert weather data into matrix
    daht = [] # date index of AH and T
    for i in range(r):
        ah[i, :] = [float(AH[i][x]) for x in city2]
        t[i, :] = [float(T[i][x]) for x in city2]
        daht.append(AH[i]["date"])

    #correct the non-available data, note by "-999"
    ah = correct_data(ah)
    t = correct_data(t)

    #......w1......w2......w3                                                  
    #1234567 1234567 1234567                                                       
    #calculate weekly weather data on average of 7, 14, 21, ... days
    for j in range(lnd/7):
        ah1 = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
        t1 = np.zeros(shape=(len(dates), len(city2)), dtype=np.float)
        for i in range(len(dates)):
            p = daht.index(dates[i])
            ah1[i, :] = sum(ah[(p-7*(j+1)+1):(p+1), :])/(7*(j+1))
            t1[i, :] = sum(t[(p-7*(j+1)+1):(p+1), :])/(7*(j+1))
        #weather data with the index of "dates"        
        data["weather"].append(ah1)
        data["weather"].append(t1)

    data["city2"] = city2
    




#temperary function for testing all years' model at once time
#@rsv_id: city id in rsv table
#@weather_id: city id in weather table
def calc_allXy(data):
    dates = data["dates"]
    city1 = data["city1"]
    city2 = data["city2"]
    lnd = data["LND"]
    aXy={}


    for n in data["jcity"]:
        weather_id = city2.index(n)
        rsv_id = city1.index(n)
        Xy={}
        
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
            Xy[year] = [X,y]
        aXy[n]=Xy
    return aXy
    
    





#def remove_features(X, aXy):
#    deleted = range(NP/7*2)
#    X=np.delete(X, [1,2,4,5] ,1)
#    for year in range(2009, 2015):
#        aXy[year][0] = np.delete(aXy[year][0], [1,2,4,5] ,1)
#    return [X, aXy]
#














def calc_avi_RSV_ss(data):
    hos = data["hospital"]
    city1 = data["city1"]
    city2 = data["city2"]
    ars = data["ars"]
    ss = data["season_start"]
    jcity = data["jcity"]

    for n in city1:
        cid = city1.index(n)
        rc={}
        for m in range(len(ss)-1):
            s = sum(hos[ss[m]:ss[m+1] ,cid])
            if s > 200:
                rc[2009+m] = s
        if rc.keys() != []:
            print rc.keys()
            ars[n] = rc

    fo = open("ava_cities.txt", "w")
    a = ""
    for i in jcity:
        if ars.has_key(i):
            a+=i 
            a+="\n"
            for j in ars[i].keys():
                a+=str(j)
                a+=":    "
                a+=str(ars[i][j])
                a+="\n"
    fo.write(a)            
            
    data["ars"] = ars
    












#This function will calculate X and y
#@paras: system parameters
#@data: data structure holding all data
#@address1, file address of hospital data
#@address2, file address of weather data
def load_data(paras, data, address):


    #load all datasets into matrix
    load_hospital_data(data, address)
    load_weather_data(data, address)

    allXy = data["allXy"]
    city1 = data["city1"]
    city2 = data["city2"]
    year = paras["year"]
    city = paras['city']



    #retrive joint cities of two dataset
    jcity = list(set(city2) & set(city1))
    jcity.sort()
    data["jcity"] = jcity


    #calc_available_RSV season
    calc_avi_RSV_ss(data)



    #retrieve X and y
    aXy = calc_allXy(data)
    #use all "advanced days" as the X
    [X, y] = aXy[city][year]


#    [X, aXy] = remove_features(X, aXy)
    data["X"] = X
    data["y"] = y
    data["allXy"] = aXy






