#!/usr/bin/pyhton

import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = np.zeros(shape=(10,10), dtype=np.int)
date = []
city = ["tot","Koln","Berlin","Bonn","Dusseldorf","Freiburg","Koln2","Munster","Hamburg","Hannover","Jena","Regensburg","Wurzburg","Tubingen","Weiden","Basel","Wiesbaden","Munchen","Ulm","Munchen2","Marburg","Bad Oeynhausen","Wien","Giessen","Innsbruck","Bern","Essen","Mainz","Zurich","Groningen","Essen2","Aachen"]
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
        d2 = date[season_start_date[t[i][1]]:season_start_date[t[i][1]+1]-1]
        
        plt.plot(d, '-', linewidth=2)

        #plot the start week
        for j in range(len(y)):
            if y[j] == 1 :
                plt.plot([j, j],[0, height], 'r-', linewidth=2)
                plt.text(j, 220, d2[j])
                
                
        plt.savefig("./image1/"+i+".png")
        plt.close()


    #plot data for all years
    for i in range (len(city)):
        plt.plot(data.T[i], 'b-', linewidth=2)

        # plot season boundry
        for m in season_start_date:
            plt.plot([m, m],[0, height], 'y--', linewidth=1)

        for m in range(1, len(season_start_date)-1):
            a = season_start_date[m]
            plt.text(a-5,255, date[a].split('-')[0])                        

        #plot the start week
        up=1
        for j in range(len(ylabel.T[i])-1):
            if ylabel.T[i,j] == 1 :
                plt.plot([j, j],[0, height], 'r-', linewidth=2)
                if up==1:
                    plt.text(j-10, 225, date[j])
                    up=0
                else:
                    plt.text(j-10, 215, date[j])
                    up=1
        plt.savefig("./image2/"+city[i]+".png")
        plt.close()
    
        
    
def screen_confused_data():
    global data, date, city, ylabel, season_start_date

    #all zeros data
    all_0_t=""
    all_0_data={}
    for j in range (len(city)):
        s = sum(data.T[j])
        if s < 10:
            all_0_data[city[j]] = s
            
    for i in all_0_data.keys():
        all_0_t += i + "  " + str(all_0_data[i]) + "\n"

    
    #partial zeros data
    partial_0_t=""
    partial_0_data ={}
    for j in range (len(city)):
        for m in range(len(season_start_date)-1):
            a=sum(data[season_start_date[m]:season_start_date[m+1]-1, j])
            if a < 1:
                if partial_0_data.has_key(city[j]):
                    partial_0_data[city[j]].append(date[season_start_date[m]])
                else:
                    partial_0_data[city[j]] = [date[season_start_date[m]]]

    for i in partial_0_data.keys():
        partial_0_t += i + "\n" + ', '.join(partial_0_data[i]) + "\n"

    #sharp data
    sharp_data={}
    sharp_t=""
    for j in range (len(city)):
        for m in range(len(season_start_date)-1):
            s_data = data[season_start_date[m]:season_start_date[m+1]-1, j]
            for i in range(len(s_data)-2):
                if abs(s_data[i+1] - s_data[i]) > 100 :
                    if sharp_data.has_key(city[j]):
                        sharp_data[city[j]].append(date[season_start_date[m]])
                    else:
                        sharp_data[city[j]] = [date[season_start_date[m]]]
    for i in sharp_data.keys():
        sharp_t += i + "\n" + ', '.join(sharp_data[i]) + "\n"

    
    #print all confused data
    t=""
    t += "#####################################\n"
    t += "#all zeros data\n"
    t += "#Definition: the city i will be reported if the number of patients < 10\n"
    t += "#####################################\n"
    t += all_0_t + "\n\n" 
    t += "#####################################\n"
    t += "#partial zeros data\n"
    t += "#Definition: the city i in season j will be reported" + \
         "if the number of patients of i in j = 0\n"
    t += "#####################################\n"
    t += partial_0_t + "\n\n"
    t += "#####################################\n"
    t += "#sharp data\n"
    t += "#Definition: the city i in season j will be reported" + \
         "if |(the number of patients of week m+1) - (that of m)| > 100\n"
    t += "#####################################\n"
    t +=  sharp_t + "\n\n"
    t += "#####################################\n"
    t += "#available data\n"
    t += "#####################################\n"
    t += ', '.join([x for x in city if x not in partial_0_data.keys() \
                    if x not in all_0_data.keys() if x not in sharp_data.keys()])
    f = open('confused_data', 'w')
    f.write(t)
    f.close()



def test_performance():
    iris = datasets.load_iris()
    iris_X = iris.data
    iris_y = iris.target
    iris_y = iris_y/2

    #simple test
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=.5, \
                                                        random_state=0)
    svc = svm.SVC(kernel='linear')
    scores = svc.fit(X_train, y_train).decision_function(X_test)
    r = svc.predict(X_test)

    #test performence method: accuracy, confusion matrix, precision, recall
    #ROC, AUC, classification report
    
    #accuracy: number of correct prediction / number of all cases
    print "\ntest accuracy:"
    print metrics.accuracy_score(y_test, r)

    #precision: tp/(tp+fp)
    print "\ntest precision:"
    print metrics.precision_score(y_test, r)

    #precision: tp/(tp+fn)
    print "\ntest recall:"
    print metrics.recall_score(y_test, r)

    #confusion matrix:
    print "\nconfusion matrix:"
    print metrics.confusion_matrix(y_test, r)

    #test roc curve and auc
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print "\ntest auc : " + str(roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

    #classification report
    print "\ntest callsification report:"
    print metrics.classification_report(y_test, r)


def test_SVM():
    iris = datasets.load_iris()
    iris_X = iris.data[:,:2]
    iris_y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=.5, \
                                                        random_state=0)
    #modify kernel
    #kernels:'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    #modify C to control the soft margin: large C, more soft
    svc = svm.SVC(kernel='linear', C=10.0)
    svc.fit(X_train, y_train)

    #plot the results
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max,\
    0.02))

    print len(np.c_[xx.ravel(), yy.ravel()])
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
        
    plt.contourf(xx, yy, Z )
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Paired)

    plt.show()                                         
    


def test_knn():
    iris = datasets.load_iris()
    iris_X = iris.data[:,:2]
    iris_y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=.5, \
                                                        random_state=0)
    #n: number of neighbors
    #weights: uniform or distance
    clf = neighbors.KNeighborsClassifier(15, weights='uniform')
    clf.fit(X_train, y_train)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"%(15, 'uniform'))
    plt.show()

def test_RF():
    
    iris = datasets.load_iris()
    iris_X = iris.data[:,:2]
    iris_y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=.5, \
                                                        random_state=0)
    #test max_depth: the max depth of the tree
    #test n_estimators: how many trees used
    #test max_features: how many good features used in each split
    clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    clf.fit(X_train, y_train)
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"%(15, 'uniform'))
    plt.show()
        

def test_weather_data():
    global city, data
    l=[]
    avi_city = []
    a=0
    with open("../data/temperature.csv") as csvfile:
        r = csv.DictReader(csvfile)
        for line in r:
            l.append(line)
            if a >10:
                break
            a+=1
    for x in l[0]:
        if x in city:
            avi_city.append(x)
    t = np.zeros(shape=(len(date), len(avi_city)), dtype=np.int)             
    t2 = np.zeros(shape=(len(date), len(avi_city)), dtype=np.int)             
    w_tempreture = np.zeros(shape=(len(date), len(avi_city)), dtype=np.int)             
    W_meanAbsHum = np.zeros(shape=(len(date), len(avi_city)), dtype=np.int)
    
    for i in range(len(l)):
        for a in l[i].keys():
            if l[i][a] == 'NA':    #handle missing value
                l[i][a] = '-999'
        t[i] = [float(l[i][x]) for x in avi_city]

    print sum(t[0:2, :])
    for i in range(0, len(l), 7):
        t2[i/7] = sum(t[i:i+7, :])
    print len(t2)
    print len(t2[0])
    print len(data)
        
        

if __name__ == "__main__":
    read_data("../data/rsv.csv")
#    plot_seasons()
    screen_confused_data()
#    test_performance()
#    test_SVM()
#    test_knn()
#    test_RF()
    test_weather_data()
