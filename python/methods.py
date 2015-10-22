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
    
def apply_evaluation(paras, X, y, clf):


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, \
                                                        random_state=0)

    clf.fit(X_train, y_train)
    r = clf.predict(X_test)

    #plot the results
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.pcolormesh(xx, yy, Z)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("binary classification classification")
    plt.show()


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
