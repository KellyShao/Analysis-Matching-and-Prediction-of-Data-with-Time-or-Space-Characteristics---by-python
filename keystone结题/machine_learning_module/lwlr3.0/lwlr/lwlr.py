from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D
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
    return timeArr,temperatureArr,idArr

def try_different_method(model):
    model.fit(resultArr,yArr)
    score = model.score(resultArr,yArr)
    result = model.predict(resultArr)
    return result,score
    

xArr, yArr,zArr = loadDataSet('tcp2_datatest2_3d_whole_2000.txt')
xArr1=[];yArr1=[];zArr1=[];
xArr1=xArr[:];yArr1=yArr[:];zArr1=zArr[:]
resultArr=[]
while xArr:resultArr.append([zArr.pop(0),xArr.pop(0)])
#变换数据
#while yArr:testArr.append([yArr.pop(0),yArr1.pop(0)])


###SVM回归——红####
from sklearn import svm
model_SVR = svm.SVR(degree=5)
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
model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
result_KNeighborsRegressor,score_KNeighborsRegressor=try_different_method(model_KNeighborsRegressor)
##随机森林回归——黄####
from sklearn import ensemble
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
result_RandomForestRegressor,score_RandomForestRegressor=try_different_method(model_RandomForestRegressor)
###GBRT回归——黑色####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
result_GradientBoostingRegressor,score_GradientBoostingRegressor=try_different_method(model_GradientBoostingRegressor)
plt.figure(figsize=(15,10))

ax1=plt.subplot(231,projection='3d')
ax1.set_title('SVR \n score: %f'%score_SVR)
plt.plot(xArr1,zArr1,result_SVR,'r-')
plt.scatter(xArr1,zArr1,yArr,color='blue',marker='o',linewidth=1,alpha=0.8)

ax2=plt.subplot(232,projection='3d')
ax2.set_title('Decision Tree Regression \n score: %f'%score_DecisionTreeRegressor)
plt.plot(xArr1,zArr1,result_DecisionTreeRegressor,'g-')
plt.scatter(xArr1, zArr1,yArr,color='blue',marker='o',linewidth=1,alpha=0.8)

ax3=plt.subplot(233,projection='3d')
ax3.set_title('Linear Regression \n score: %f'%score_LinearRegression)
plt.plot(xArr1,zArr1,result_LinearRegression,'c-')
plt.scatter(xArr1,zArr1, yArr,color='blue',marker='o')

ax4=plt.subplot(234,projection='3d')
ax4.set_title('K Neighbors Regression \n score: %f'%score_KNeighborsRegressor)
plt.plot(xArr1,zArr1,result_KNeighborsRegressor,'m-')
plt.scatter(xArr1,zArr1, yArr,color='blue',marker='o',linewidth=1,alpha=0.8)

ax5=plt.subplot(235,projection='3d')
ax5.set_title('Random Forest Regression \n score: %f'%score_RandomForestRegressor)
plt.plot(xArr1,zArr1,result_RandomForestRegressor,'y-')
plt.scatter(xArr1,zArr1, yArr,color='blue',marker='o',linewidth=1,alpha=0.8)

ax6=plt.subplot(236,projection='3d')
ax6.set_title('Gradient Boosting Regression \n score: %f'%score_GradientBoostingRegressor)
plt.plot(xArr1,zArr1,result_GradientBoostingRegressor,'k-')
plt.scatter(xArr1,zArr1, yArr,color='blue',marker='o',linewidth=1,alpha=0.8)

plt.legend()
plt.savefig('C:\Users\lh.Lenovo-PC\Desktop\MyFig.png', dpi=300)
plt.show()
