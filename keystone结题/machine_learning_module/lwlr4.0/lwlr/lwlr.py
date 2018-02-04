from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
#from sklearn.linear_model.ridge import Ridge


def loadDataSet(fileName):
    numFeat = len(open(fileName).readlines()) - 1
    timeArr =[];temperatureArr =[];idArr=[]
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip('\n')
        idArr.append(float(curLine.split()[0]))
        timeArr.append(float(curLine.split()[1]))
        temperatureArr.append(float(curLine.split()[2]))
    return idArr,temperatureArr

def try_different_method(model):
    model.fit(resultArr,yArr)
    score = model.score(resultArr, yArr)
    result = model.predict(resultArr)
    return result,score
    

xArr, yArr = loadDataSet('moment_area_data_100_1315_3(1).txt')
xArr1=[];yArr1=[];xArr2=[]
xArr1=xArr[:];yArr1=yArr[:];xArr2=xArr[:]
testArr=yArr[:]
resultArr=[]
while xArr:resultArr.append([xArr.pop(0),xArr2.pop(0)])
###SVM回归——红####
from sklearn import svm
model_SVR = svm.SVR(C=3)
result_SVR,score_SVR=try_different_method(model_SVR)
####决策树回归——绿####
from sklearn import tree
model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
result_DecisionTreeRegressor,score_DecisionTreeRegressor=try_different_method(model_DecisionTreeRegressor)
###线性回归——青####
from sklearn import linear_model
model_LinearRegression = linear_model.LinearRegression()
result_LinearRegression,score_LinearRegression=try_different_method(model_LinearRegression)
###KNN回归——品红####
from sklearn import neighbors
model_KNeighborsRegressor = neighbors.KNeighborsRegressor(n_neighbors=20)
result_KNeighborsRegressor,score_KNeighborsRegressor=try_different_method(model_KNeighborsRegressor)
##随机森林回归——黄####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=50)#这里使用20个决策树
result_RandomForestRegressor,score_RandomForestRegressor=try_different_method(model_RandomForestRegressor)
###GBRT回归——黑色####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=1000)#这里使用100个决策树
result_GradientBoostingRegressor,score_GradientBoostingRegressor=try_different_method(model_GradientBoostingRegressor)
plt.figure(figsize=(15,10))

ax1=plt.subplot(231)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax1.set_title('SVR \n score: %f'%score_SVR)
plt.plot(xArr1,result_SVR,'r-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)



ax2=plt.subplot(232)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax2.set_title('Decision Tree Regression \n score: %f'%score_DecisionTreeRegressor)
plt.plot(xArr1,result_DecisionTreeRegressor,'g-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax3=plt.subplot(233)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax3.set_title('Linear Regression \n score: %f'%score_LinearRegression)
plt.plot(xArr1,result_LinearRegression,'c-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax4=plt.subplot(234)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax4.set_title('K Neighbors Regression \n score: %f'%score_KNeighborsRegressor)
plt.plot(xArr1,result_KNeighborsRegressor,'m-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax5=plt.subplot(235)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax5.set_title('Random Forest Regression \n score: %f'%score_RandomForestRegressor)
plt.plot(xArr1,result_RandomForestRegressor,'y-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax6=plt.subplot(236)
plt.xlim(15070000, 15070001)
plt.ylim(4, 22)
ax6.set_title('Gradient Boosting Regression \n score: %f'%score_GradientBoostingRegressor)
plt.plot(xArr1,result_GradientBoostingRegressor,'k-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

plt.legend()
plt.savefig('C:\Users\lh.Lenovo-PC\Desktop\MyFig.png', dpi=300)
plt.show()
