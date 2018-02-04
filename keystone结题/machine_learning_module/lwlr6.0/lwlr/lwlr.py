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
        idArr.append(float(curLine.split()[2]))
        timeArr.append(float(curLine.split()[0]))
        temperatureArr.append(float(curLine.split()[1]))
    return timeArr,temperatureArr

def try_different_method(model):
    model.fit(resultArr,yArr)
    score = model.score(resultArr, yArr)
    result = model.predict(resultArr)
    return result,score
    


#path1=raw_input("Input the file path:")
#path2=".txt";
#path=path1+path2;
xArr, yArr = loadDataSet('111.txt')
xArr1=[];xArr2=[];xArr3=[]
xArr1=xArr[:]
xArr1.sort()
xArr2=xArr1[:];xArr3=xArr1[:]
while xArr:resultArr.append([xArr1.pop(0),xArr2.pop(0)])
i = 0
while i < len(xArr):
    i=i+1

p=xArr.index(value)
yArr1=yArr[:]
testArr=yArr[:]
resultArr=[]
while xArr:resultArr.append([xArr.pop(0),xArr2.pop(0)])
###SVM回归——红####
from sklearn import svm
model_SVR = svm.SVR()
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
model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=10)#这里使用20个决策树
result_RandomForestRegressor,score_RandomForestRegressor=try_different_method(model_RandomForestRegressor)
###GBRT回归——黑色####
from sklearn import ensemble
model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=50)#这里使用100个决策树
result_GradientBoostingRegressor,score_GradientBoostingRegressor=try_different_method(model_GradientBoostingRegressor)

plt.figure(figsize=(15,10))

ax1=plt.subplot(231)
ax1.set_title('SVR \n score: %f'%score_SVR)
plt.plot(xArr1,result_SVR,'r-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax2=plt.subplot(232)
ax2.set_title('Decision Tree Regression \n score: %f'%score_DecisionTreeRegressor)
plt.plot(xArr1,result_DecisionTreeRegressor,'g-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax3=plt.subplot(233)
ax3.set_title('Linear Regression \n score: %f'%score_LinearRegression)
plt.plot(xArr1,result_LinearRegression,'c-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax4=plt.subplot(234)
ax4.set_title('K Neighbors Regression \n score: %f'%score_KNeighborsRegressor)
plt.plot(xArr1,result_KNeighborsRegressor,'m-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax5=plt.subplot(235)
ax5.set_title('Random Forest Regression \n score: %f'%score_RandomForestRegressor)
plt.plot(xArr1,result_RandomForestRegressor,'y-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

ax6=plt.subplot(236)
ax6.set_title('Gradient Boosting Regression \n score: %f'%score_GradientBoostingRegressor)
plt.plot(xArr1,result_GradientBoostingRegressor,'k-',linewidth=2)
plt.scatter(xArr1, yArr,color='blue',marker='o',linewidth=0.2,alpha=0.8)

plt.legend()
plt.savefig('C:MyFig.png', dpi=300)
plt.show()

#predict_value=raw_input("Input the predict value:")
#predict_SVR_value=[[predict_value,predict_value]]
#predict_SVR,predict_SVR=try_different_method(predict_SVR_value)
#print (" %s"%("predict_SVR_value="))
#print("%f"%(predict_SVR))