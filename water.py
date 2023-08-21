# linear regression
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
data=pd.read_csv('pure.csv')
print(data.head())
print(data.describe())
X=data[['ph']]
Y=data[['purity']]
Y_=data[['y/n']]
linear_regressor=LinearRegression()
linear_regressor.fit(X,Y)
pred=linear_regressor.predict([[9]])
print("predicted value:",pred[0])
print("accuracy:",linear_regressor.score(X[:20],Y[:20])*100)
plt.scatter(X['ph'],Y,color='b')
plt.plot(X['ph'],linear_regressor.predict(X),color='black',linewidth=3)
plt.xlabel('ph')
plt.ylabel('purity')
plt.show()
#logistic regression
from sklearn.linear_model import LogisticRegression
mdl = LogisticRegression()
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (LGR): ",pred[0])
print("Accuracy (LGR): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('purity') 
plt.show()

#SVR
from sklearn.svm import SVR
mdl = SVR(kernel = 'rbf')
mdl.fit(X, Y)
pred = mdl.predict([[9]])
print("Predicted value (SVR): ",pred[0])
print("Accuracy (SVR): ",mdl.score(X[:20], Y[:20])*100)

plt.scatter(X['ph'], Y, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')
plt.ylabel('purity')
plt.show()



#SVC
from sklearn.svm import SVC
mdl = SVC(kernel='poly')
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (SVC): ",pred[0])
print("Accuracy (SVC): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('y/n') 
plt.show()


#NB
from sklearn.naive_bayes import GaussianNB
mdl = GaussianNB()
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (NB): ",pred[0])
print("Accuracy (NB): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('y/n') 
plt.show()


#KNN
from sklearn.neighbors import KNeighborsClassifier
mdl = KNeighborsClassifier()
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (KNN): ",pred[0])
print("Accuracy (KNN): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('y/n') 
plt.show()


#RANDOM FOREST CLASSIFICATION
from sklearn.ensemble import RandomForestClassifier
mdl = RandomForestClassifier(criterion='entropy')
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (RFC): ",pred[0])
print("Accuracy (RFC): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('y/n') 
plt.show()


#RANDOM FOREST REGRESSION
from sklearn.ensemble import RandomForestRegressor
mdl = RandomForestRegressor(n_estimators=100,max_depth=6)
mdl.fit(X, Y)
pred = mdl.predict([[9]])
print("Predicted value (RFR): ",pred[0])
print("Accuracy (RFR): ",mdl.score(X[:20], Y[:20])*100)

plt.scatter(X['ph'], Y, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('purity') 
plt.show()


#DECISION TREE CLASSIFICATION
from sklearn.tree import DecisionTreeClassifier
mdl = DecisionTreeClassifier(max_leaf_nodes=3, random_state=1)
mdl.fit(X, Y_)
pred = mdl.predict([[9]])
print("Predicted value (DTC): ",pred[0])
print("Accuracy (DTC): ",mdl.score(X[:20], Y_[:20])*100)

plt.scatter(X['ph'], Y_, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('y/n') 
plt.show()


#DECISION TREE REGRESSION
from sklearn.tree import DecisionTreeRegressor
mdl =  DecisionTreeRegressor(max_depth=3)
mdl.fit(X, Y)
pred = mdl.predict([[9]])
print("Predicted value (DTR): ",pred[0])
print("Accuracy (DTR): ",mdl.score(X[:20], Y[:20])*100)

plt.scatter(X['ph'], Y, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('purity') 
plt.show()


#KMEANS
from sklearn.cluster import KMeans
k = 2
mdl = KMeans(n_clusters=k)
mdl.fit(data.iloc[1:])
centroids = mdl.cluster_centers_
print("Centroids: ",centroids)
pred = mdl.predict([[1,9,8,9,2,5,4,0]])
print("Predicted value (KM): ",pred[0])

labels = mdl.labels_
colors = ['blue','red','green','black','purple','yellow','orange','grey']
y = 0
for x in labels:
    # plot the points acc to their clusters
    # and assign different colors
    plt.scatter(data.iloc[y,0], data.iloc[y,1],color=colors[x])
    y+=1
        
for x in range(k):
    #plot the centroids
    lines = plt.plot(centroids[x,0],centroids[x,1],'kx')    
    #make the centroid larger    
    plt.setp(lines,ms=15.0)
    plt.setp(lines,mew=2.0)
    
title = ('No of clusters (k) = {}').format(k)
plt.title(title)
plt.xlabel('eruptions (mins)')
plt.ylabel('waiting (mins)')
plt.show()

#MULTIPLE LINEAR REGRESSION
X = data[["ph","lead"]]
mdl = LinearRegression()
mdl.fit(X, Y)
pred = mdl.predict([[6, 9]])
print("Predicted value (MLR): ",pred[0])
print("Accuracy (MLR): ",mdl.score(X[:20], Y[:20])*100)

plt.scatter(X['ph'], Y, color='b')
plt.plot(X['ph'], mdl.predict(X),color='black',linewidth=3)
plt.xlabel('ph')  
plt.ylabel('purity') 
plt.show()