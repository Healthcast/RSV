#!/usr/bin/pyhton


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier



def apply_algorithm(paras, X, y):

    if paras['clf'] == 'svm':
        clf = svm.SVC(kernel=paras['svm'][1], C=paras['svm'][0], probability=True)
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

    
def apply_evaluation(paras, X, y, clf, data):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, \
                                                        random_state=0)

    clf.fit(X_train, y_train)
    r = clf.predict(X_test)


    d = clf.decision_function(X)
    p = clf.predict_proba(X).T[1]*3
    h = data["hospital"].T[data["city"].index(paras["city"])]
    h1 = h.astype(float)
    m = max(h1)
    h1=h1/m*4

    plt.figure()
#    plt.plot(d)
    plt.plot(y)
    plt.plot(h1)
    plt.plot(p)


#    height = 4
#    bottom = -2
#    ss = data["season_start"]
#    date=data["date1"]
#    c_id = data["city"].index(paras["city"])
#    ylabel = data["ylabels"]
#    for m in ss:
#        plt.plot([m, m],[bottom, height], 'y--', linewidth=1)
#
#    for m in range(1, len(ss)-1):
#        a = ss[m]
#        plt.text(a-5,height, date[a].split('-')[0])                        
#
#   #plot the start week
#    up=1
#    for j in range(len(ylabel.T[c_id])-1):
#        if ylabel.T[c_id,j] == 1 :
#            plt.plot([j, j],[bottom, height], 'k-', linewidth=2)
#            if up==1:
#                plt.text(j-10, height-1, date[j])
#                up=0
#            else:
#                plt.text(j-10, height-2, date[j])
#                up=1
#

    plt.show()




    #plot the results
#    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
#    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
#
#    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
#    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#    Z = Z.reshape(xx.shape)
#
#    plt.figure()
#    plt.pcolormesh(xx, yy, Z)
#    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
#    plt.xlim(xx.min(), xx.max())
#    plt.ylim(yy.min(), yy.max())
#    plt.title("binary classification classification")
#    plt.show()
#

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
