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
label = []





def read_data(address):
    global data, date, city, label
    l=[]
    with open(address) as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)
            
    #remove data in August and September
    for i in l:
        d=i[''].split('-')
        if int(d[1]) == 8 or int(d[1]) == 9:
            del i
      
    for i in l:
        date.append(i[''])
        del i['']
  
    data = np.zeros(shape=(len(date), len(city)), dtype=np.int)

    for i in range(len(l)):
        data[i] = [l[i][x] for x in city]          
#    print [x for x in city]          
    for j in range (len(city)):
        label.append([])
        had_one=0
        for i in range (len(date)-2):
            if data[i,j]>0 and data[i+1,j] > data[i,j] and data[i+2,j] > \
                   data[i+1,j] and had_one==0:
                label[j].append('Y')
                had_one=1
            else:    label[j].append('N')
    

                
def plot_seasons():
    global data, date, city, label

    t={}
    se=[]

    for p in range(2009, 2015):
        start=0
        end=0
        for j in range(len(date)-1):
            d = date[j].split('-')
            if str(p) == d[0] and d[1] == '10' and start == 0:
                se.append(j)
                start = 1
            d = date[j+1].split('-')
            if str(p+1) == d[0] and d[1] == '07' and end ==0:
                se.append(j)     
                end=1

    se.append(len(date)-1)

    for i in range(len(city)):
        for p in range(2009, 2015):
            c = (p-2009)*2
            t[city[i] + "_" + str(p)] = data[se[c]:se[c+1], i]
            
    
    for i in t.keys():
        plt.plot(t[i], '-', linewidth=2)
        plt.savefig("./image1/"+i+".png")
        plt.close()
       
    for i in range (len(city)):
        plt.plot(data.T[i], '-', linewidth=2)
        plt.savefig("./image2/"+city[i]+".png")
        plt.close()
    
        
    
        
    



if __name__ == "__main__":

    read_data("../data/rsv.csv")
    plot_seasons()    
