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
    h = data["hospital"].T[data["city1"].index(paras["city"])]
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
    plt.scatter(X[:, 0], X[:, 3], c=y, cmap=plt.cm.Paired)
    plt.xlabel('ah1')
    plt.ylabel('t1')



    plt.figure(3, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 1], X[:, 4], c=y, cmap=plt.cm.Paired)
    plt.xlabel('ah2')
    plt.ylabel('t2')

    
    plt.figure(4, figsize=(8, 6))
    plt.clf()
    plt.scatter(X[:, 2], X[:, 5], c=y, cmap=plt.cm.Paired)
    plt.xlabel('ah3')
    plt.ylabel('t3')

    

    plt.figure(6, figsize=(8, 6))
    plt.plot(X[:,0], label="ah1")
    plt.plot(X[:,1], label="ah2")
    plt.plot(X[:,2], label="ah3")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


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


    


def testAllXyModel(data):
    X = data["X"]
    y = data["y"]
    aXy = data["allXy"]
    lnd = data["LND"]
    p=12
    accus = []


    xs = np.hsplit(X,X.shape[1]/2)
    print len(xs)
    clfs=[]
    for i in range(lnd/7):
        clf = linear_model.LogisticRegression(C=0.5)
        clf.fit(xs[i],y)
        clfs.append(clf)

    for i in range(len(clfs)):
        s=[]
        for year in range(2009, 2015):
            x =aXy[year][0]
            xx = np.hsplit(x,x.shape[1]/2)
            yy =aXy[year][1]
            print "The year: " + str(year)
            if (yy == 1).all() or (yy == -1).all():
                print "only one class"
            else:
                r = clfs[i].predict(xx[i])
                print "Accuracy:"
                s.append(metrics.accuracy_score(yy, r))
                print metrics.accuracy_score(yy, r)
        print "average accuracy"
        accus.append(sum(s)/len(s))
    print accus
    
    for i in range(lnd/7):
        plt.figure(year-2008, figsize=(24, 18))
        for year in range(2009, 2015):
            xx =aXy[year][0]
            yy =aXy[year][1]
            if (yy == 1).all() or (yy == -1).all():
                print "only one class"
            else:
                plt.plot(xx[:,i*2], label=str(year))
                first1 = np.where(yy>0)[0][0]
                plt.plot(first1, xx[:,i*2][first1], 'rD')

        plt.ylabel('ah on average of ' + str((i+1)*7))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("./image3/"+str((i+1)*7)+".png")
        plt.close()
        

    
def apply_evaluation(paras,  clf, data):

    X = data["X"]
    y = data["y"]

    testAllXyModel(data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, \
                                                        random_state=0)

    clf.fit(X_train, y_train)
    r = clf.predict(X_test)
#    plot_results(r, clf, data, paras)


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
    elif paras['eva'] == 'roc':
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
