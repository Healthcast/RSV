#!/usr/bin/pyhton


import getopt, sys
import numpy as np
import methods #algorithm and evaluation stage
import preprocess #preprocess stage 



#default values
paras={
    "clf": "svm",
    "eva": "accuracy",
    "city":"Munchen",
    "svm":[1, "linear"],
    "knn":[10, "uniform"],
    "rf":[10, 10, 1],
    "year":2012
}


data={
    "date1" : [], #date list for hospital data
    "date2" : [], #date list for weather data
    "city" :  [], #all cities' name in rsv data
    "city2": [], #all cities' name in weather data
    "hospital" : np.zeros(shape=(10,10), dtype=np.int), #hospital data
    "weather" : [], #weather data
    "ylabels" : np.zeros(shape=(10,10), dtype=np.int), #y labels for all hospital data
    "y" : np.zeros(shape=(10,1), dtype=np.int), #1:yes. 0:not the start week
    "X" : np.zeros(shape=(10,10), dtype=np.int), 
    "season_start" : [] #start date of each season (October)
}


def print_help():
    print \
    "Parameters explaination\n" +\
    "--clf    classifier: svm, knn, rf\n" +\
    "--eva    evauation method: accuracy, precision, confusion, report, roc\n" +\
    "--city   city name\n" +\
    "--svm_C  parameter of svm, control the softmaigin\n" +\
    "--svm_k  kernel function: linear, poly, rbf, sigmoid\n" +\
    "--knn_n  the number of neighbors\n"+\
    "--knn_w  the weight of neighbors: uniform, distance\n" +\
    "--rf_d   max depth of the tree\n" + \
    "--rf_n   number of estimators\n" +\
    "--rf_p   number of good features in each split\n" +\
    "--year   2009~2014\n"
def main():
    global paras
    long_opts = ["help", "clf=", "city=", "eva=", \
                 "svm_C=", "svm_k=", "knn_n=", "knn_w=", "rf_d=", \
                 "rf_n=", "rf_p=", "year="]
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", long_opts)
    except getopt.GetoptError as err:
        print str(err) 
        sys.exit(2)
    for o, a in opts:
        if o in ("-h", "--help"):
            print_help()
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
        elif o in ("--year"):
            paras["year"] = int(a)
        else:
            assert False, "unhandled option"

    preprocess.load_data(paras, data,"../data/")
    clf = methods.apply_algorithm(paras, data)
    methods.apply_evaluation(paras, clf, data)

if __name__ == "__main__":
    main()
