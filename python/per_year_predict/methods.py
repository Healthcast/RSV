#!/usr/bin/pyhton


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA




def apply_algorithm(paras, data):
    X = data["X"]
    y = data["y"]

    if paras['clf'] == 'svm':
        clf = svm.SVC(kernel=paras['svm'][1], C=paras['svm'][0], probability=True)
    elif paras['clf'] == 'knn':
        clf = neighbors.KNeighborsClassifier(paras['knn'][0],\
                                             weights=paras['knn'][1])
    elif paras['clf'] == 'rf':
        clf = RandomForestClassifier(max_depth=paras['rf'][0], \
                                     n_estimators=paras['rf'][1],\
                                     max_features=paras['rf'][2])
    elif paras['clf'] == 'lr':
        clf = linear_model.LogisticRegression(C=0.5)
    else:
        print str("unknown classifier") 
        sys.exit(2)
    

    return clf



def plot_results(r, clf, data, paras):
    X = data["X"]
    y = data["y"]

    d = clf.decision_function(X)
    p = clf.predict_proba(X).T[1]
    h = data["hospital"].T[data["city"].index(paras["city"])]
    h1 = h.astype(float)
    m = max(h1)
    h1=h1/m*4

    plt.figure(1)
#    pd = plt.plot(d, label="distance")
    plt.plot(y, label="y")
#    plt.plot(h1)
    plt.plot((p-0.5)*2, label="probability")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.figure(2, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
    plt.xlabel(' AbsHumidity')
    plt.ylabel('meanAbsHumidity')
    plt.xticks(())
    plt.yticks(())


    plt.figure(3, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.Paired)
    plt.xlabel('AbsHumidity')
    plt.ylabel('temperature')
    plt.xticks(())
    plt.yticks(())
    
    plt.figure(4, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Paired)
    plt.xlabel('meanAbsHumidity')
    plt.ylabel('meanTemperature')
    plt.xticks(())
    plt.yticks(())
    
    fig = plt.figure(5, figsize=(8, 6))
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(X)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap=plt.cm.Paired)
    ax.set_title("First three PCA directions")
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])

    plt.show()


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
    

    
def apply_evaluation(paras,  clf, data):
    X = data["X"]
    y = data["y"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, \
                                                        random_state=0)

    clf.fit(X_train, y_train)
    r = clf.predict(X_test)
    plot_results(r, clf, data, paras)
    

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
