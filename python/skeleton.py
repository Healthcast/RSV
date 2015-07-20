#!/usr/bin/pyhton

import csv
import numpy as np
import matplotlib.pyplot as plt


## Code skeleton for analysis pipeline
##
## ASSUME the rsv data is in the file ../data/rsv.csv
## I put the data in a separate directory so it is easy to
## keep it out of the repo. We have to be very strict about
## NEVER including the data; Git makes it very hard to remove
## something once it has been added.

## From the file ../data/rsv.csv,

## extract for each city,
##   each season which has recorded activity
##   Preliminary definitions:
##      a season is the period 1 Oct to 1 July, spanning the new year
##      "recorded activity" means at least 3 weeks with > 0 cases

## For each extracted season, flag the start week of the epidemic
##   Preliminary flag:
##      start week is the first week with >0 cases, for which the
##      next two weeks show more cases than the initial week.
##      In each of the following, the first positive integer marks
##      the start week: (0,0,1,2,3) (0,4,12,12)  (0,3,20,17)

## Make a function which plots each season as a line graph (x axis is time,
## y axis is number of cases). Add a mark at each week flagged as start week.
## Use this function to see if the flag definition makes sense.

## Create a test harnass for stat learning algos from  http://scikit-learn.org/




data = np.zeros(shape=(10,10), dtype=np.int)
date = []
city = ["tot","Koln","Berlin","Bonn","Dusseldorf","Freiburg","Koln2","Munster","Hamburg","Hannover","Jena","Regensburg","Wurzburg","Tubingen","Weiden","Basel","Wiesbaden","Munchen","Ulm","Munchen2","Marburg","Bad Oeynhausen","Wien","Giessen","Innsbruck","Bern","Essen","Mainz","Zurich","Groningen","Essen","Aachen"]
ylabel = np.zeros(shape=(10,10), dtype=np.int) #1:yes. 0:not the start week
data = np.zeros(shape=(10,10), dtype=np.int)
season_start_date=[]




def read_data(address):
    global data, date, city, ylabel, season_start_date
    l=[]
    with open(address) as csvfile:
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
        date.append(i[''])
        del i['']

    #init ylabel and date 
    data = np.zeros(shape=(len(date), len(city)), dtype=np.int)
    ylabel = np.zeros(shape=(len(date), len(city)), dtype=np.int)

    #insert patient data
    for i in range(len(newl)):
        data[i] = [newl[i][x] for x in city]

    #determine the start week of each season: Octorber 01
    for p in range(2009, 2016):
        start=0
        for j in range(len(date)):
            d = date[j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                season_start_date.append(j)
                start = 1
    season_start_date = [0] + season_start_date
    season_start_date = season_start_date + [len(date)-1]

    #label and insert the "start week" to ylabel
    for j in range (len(city)):
        for m in range(len(season_start_date)-1):
            had_one=0
            for i in range(season_start_date[m], season_start_date[m+1]-1):
                if data[i,j]>0 and data[i+1,j] > data[i,j] and data[i+2,j] > \
                       data[i,j] and had_one==0:
                    ylabel[i,j] = 1
                    had_one=1
                else:    ylabel[i,j] = 0

                
def plot_seasons():
    global data, date, city, ylabel, season_start_date
    height = 250
    t={}

    #loading data
    for i in range(len(city)):
        for m in range(len(season_start_date)-1):
            t[city[i] + "_" + str(m+2008)] = [ i,m]
            

    #plot data for every season
    for i in t.keys():
        d = data[season_start_date[t[i][1]]:season_start_date[t[i][1]+1]-1, \
                 t[i][0]]
        y = ylabel[season_start_date[t[i][1]]:season_start_date[t[i][1]+1]-1, \
                 t[i][0]]
        
        plt.plot(d, '-', linewidth=2)

        #plot the start week
        for j in range(len(y)):
            if y[j] == 1 :
                plt.plot([j, j],[0, height], 'r-', linewidth=2)
    
        plt.savefig("./image1/"+i+".png")
        plt.close()



    #plot data for all years
    for i in range (len(city)):
        plt.plot(data.T[i], 'b-', linewidth=2)

        # plot season boundry
        for m in season_start_date:
            plt.plot([m, m],[0, height], 'y--', linewidth=1)

        #plot the start week
        for j in range(len(ylabel.T[i])-1):
            if ylabel.T[i,j] == 1 :
                plt.plot([j, j],[0, height], 'r-', linewidth=2)
                plt.text(j,0, "2009")
        
        plt.savefig("./image2/"+city[i]+".png")
        plt.close()
    
        
    
        
    



if __name__ == "__main__":

    read_data("../data/rsv.csv")
    plot_seasons()    
