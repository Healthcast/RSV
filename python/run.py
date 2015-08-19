#!/usr/bin/pyhton


import getopt, sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier


paras={
    "clf": "svm",
    "eva": "accuracy",
    "city":"Koln",
    "svm":[10, "linear"],
    "knn":[10, "uniform"],
    "rf":[10, 10, 1],
    "features":["temperature"]
    }


date = []
city = []
city = ["tot","Koln","Berlin","Bonn","Dusseldorf","Freiburg","Koln2","Munster","Hamburg","Hannover","Jena","Regensburg","Wurzburg","Tubingen","Weiden","Basel","Wiesbaden","Munchen","Ulm","Munchen2","Marburg","Bad Oeynhausen","Wien","Giessen","Innsbruck","Bern","Essen","Mainz","Zurich","Groningen","Essen2","Aachen"]
data = np.zeros(shape=(10,10), dtype=np.int)
ylabel = np.zeros(shape=(10,10), dtype=np.int) #1:yes. 0:not the start week
y = np.zeros(shape=(10,10), dtype=np.int) #1:yes. 0:not the start week
X = np.zeros(shape=(10,1), dtype=np.int) 
data = np.zeros(shape=(10,10), dtype=np.int)
season_start_date=[]



def pre_process_data(address):
    global data, date, city, ylabel, season_start_date, paras, X, y
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
                
    X = np.zeros(shape=(len(date), 2), dtype=np.int)
    city_id = city.index(paras['city'])
    X[:,0] = data.T[city_id]
    X[:,1] =[x*x for x in data.T[city_id]] 

    y = np.zeros(shape=(len(date), 1), dtype=np.int)
    y = ylabel.T[city_id]
        

    
def apply_algorithm():
    global paras, X, y
    if paras['clf'] == 'svm':
        clf = svm.SVC(kernel=paras['svm'][1], C=paras['svm'][0])
    elif paras['clf'] == 'knn':
        clf = neighbors.KNeighborsClassifier(paras['knn'][0],\
                                             weights=paras['knn'][1])
    elif paras['clf'] == 'rf':
        clf = RandomForestClassifier(max_depth=paras['rf'][0], \
                                     n_estimators=paras['rf'][1],\
                                     max_features=paras['rf'][2])
    else:
        print str("unknown classifier") 
        sys.exit(2)

    return clf
    
def apply_evaluation(clf):
    global paras, X, y


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, \
                                                        random_state=0)

    clf.fit(X_train, y_train)
    r = clf.predict(X_test)

    if paras['eva'] == 'accuracy':
        print "The accuracy:"
        print metrics.accuracy_score(y_test, r)
    elif paras['eva'] == 'precision':
        print "The precision:"
        print metrics.precision_score(y_test, r)
    elif paras['eva'] == 'recall':
        print "The recall:"
        print metrics.recall_score(y_test, r)
    elif paras['eva'] == 'confusion':
        print "The confusion matrix:"
        print metrics.confusion_matrix(y_test, r)
    elif paras['eva'] == 'report':
        print "The report:"
        print metrics.classification_report(y_test, r)
    elif paras['eva'] == 'roc' and paras['clf'] == 'svm':
        scores = clf.decision_function(X_test)
        print "The auc:"
        fpr, tpr, thresholds = metrics.roc_curve(y_test, scores)
        roc_auc = metrics.auc(fpr, tpr)
        print str(roc_auc)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

        


def main():
    global paras
    long_opts = ["help", "clf=", "city=", "eva=", \
                 "svm_C=", "svm_k=", "knn_n=", "knn_w=", "rf_d=", \
                 "rf_n=", "rf_p="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", long_opts)
    except getopt.GetoptError as err:
        print str(err) 
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            sys.exit()
        elif o in ("--clf"):
            paras["clf"] = a
        elif o in ("--city"):
            paras["city"] = a
        elif o in ("--eva"):
            paras["eva"] = a
        elif o in ("--svm_C"):
            paras["svm"][0] = int(a)
        elif o in ("--svm_k"):
            paras["svm"][1] = a
        elif o in ("--knn_n"):
            paras["knn"][0] = int(a)
        elif o in ("--knn_w"):
            paras["knn"][1] = a
        elif o in ("--rf_d"):
            paras["rf"][0] = int(a)
        elif o in ("--rf_n"):
            paras["rf"][1] = int(a)
        elif o in ("--rf_p"):
            paras["rf"][2] = int(a)
        else:
            assert False, "unhandled option"
    paras["features"] = args

    pre_process_data("../data/rsv.csv")
    clf = apply_algorithm()
    apply_evaluation(clf)

if __name__ == "__main__":
    main()
