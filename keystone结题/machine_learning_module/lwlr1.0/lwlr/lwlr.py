from numpy import *
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    numFeat = len(open(fileName).readlines()) - 1 
    #timeMat = []; temperatureMat = [];idMat=[]
    timeArr =[];temperatureArr =[];idArr=[]
    fr = open(fileName)
    for line in fr.readlines():
        #timeArr =[];temperatureArr =[];idArr=[]
        curLine = line.strip('\n')
        #for i in range(numFeat):#i till numFeat
        #time=curLine.split()[0]
        #temperature=curLine.split()[1]
        #id=curLine.split()[2]
        idArr.append(float(curLine.split()[0]))
        timeArr.append(float(curLine.split()[1]))
        temperatureArr.append(float(curLine.split()[2]))
        #timeMat.append(timeArr)
        #temperatureMat.append(temperatureArr)
        #idMat.append(idArr)
    #return timeMat; temperatureMat#;idMat
    return timeArr,temperatureArr


##calculate regression coefcient
#def standRegres(xArr,yArr):
#    xMat = mat(xArr)
#    yMat = mat(yArr).T #invert matrix
#    xTx = xMat.T * xMat
#    if linalg.det(xTx) == 0.0:
#        print 'This matrix is singular, cannot do inverse'
#        ws=np.linalg.pinv(xTx) 
#    else: ws = xTx.I * (xMat.T * yMat) #coefcient w=[(xt x)-1 xt y]
#    return ws

### Front stage-wise Regression ###  
def stageWise(xArr, yArr, step=0.01, numIt=100) :  
    xMat = mat(xArr)  
   # xMat = regularize(xMat)  
    yMat = mat(yArr).T  
    yMean = mean(yMat)  
    yMat = yMat - yMean  
    N, n = shape(xMat)  
    returnMat = zeros((numIt, n))  
    ws = zeros((n,1))  
    wsTest = ws.copy()  
    weMax = ws.copy()  
    for ii in range(numIt) :  
        print ws.T  
        lowestErr = inf  
        for jj in range(n) :  
            for sign in [-1,1] :  
                wsTest = ws.copy()  
                wsTest[jj] += step*sign  
                yTest = xMat*wsTest  
                rssE = rssError(yMat.A, yTest.A)  
                if rssE < lowestErr :  
                    lowestErr = rssE  
                    wsMax = wsTest  
        ws = wsMax.copy()  
        returnMat[ii,:] = ws.T  
    return returnMat


#show plot
#def plotStandRegres(xArr,yArr,ws):
#    import matplotlib.pyplot as plt 
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot([i[1] for i in xArr],yArr,'ro')
#    xCopy = xArr
#    print type(xCopy)
#    xCopy.sort()
#    yHat = xCopy*ws
#    ax.plot([i[1] for i in xCopy],yHat)
#    plt.show()


#correlation
def calcCorrcoef(xArr,yArr,ws):
    xMat = mat(xArr)
    yMat = mat(yArr)
    yHat = xMat*ws
    return corrcoef(yHat, yMat)

#set k
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = mat(xArr); yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) #diagnose matrix
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse"
        ws=np.linalg.pinv(xTx) 
    else:ws = xTx.I * (xMat.T * (weights * yMat))
    a=testPoint * ws
    return a

##define k
#def lwlrTest(testArr,xArr,yArr,k=1.0):
#    m = shape(testArr)[0]
#    yHat = zeros(m)
#    for i in range(m):
#        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
#    return yHat


#set k
def lwlrTestPlot(xArr,yArr,k=1.0):
    import matplotlib.pyplot as plt
    x=shape(yArr)
    yHat = zeros((x.x))
    xCopy = mat(xArr)
    xCopy.sort(0)
    for i in range(shape(xArr)[0]):
        yHat[i,:] = lwlr(xCopy[i],xArr,yArr,k)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([i[1] for i in xArr],[i[1] for i in yArr],'ro')
    ax.plot(xCopy,yHat)
    ax.scatter(xCopy[:,1].flatten().A[0],mat(yArr).T[:,0].flatten().A[0],s=2,c='red') 
    plt.show()
    #return yHat,xCopy

     
def rssError(yArr,yHatArr): #yArr and yHatArr both need to be arrays
    return ((yArr-yHatArr)**2).sum()


def main():
    #regression
    xArr=[];yArr=[]
    xArr,yArr = loadDataSet('tcp2_1.txt')
    #ws = standRegres(xArr,yArr)
    #设置图表字体为华文细黑，字号15
    plt.rc('font', family='STXihei', size=15)
#绘制散点图，广告成本X，点击量Y，设置颜色，标记点样式和透明度等参数
    plt.scatter(xArr,yArr,60,color='blue',marker='o',linewidth=1,alpha=0.8)
#添加x轴标题
    plt.xlabel('time')
#添加y轴标题
    plt.ylabel('temperature')
#添加图表标题
    plt.title('temperature/day Analyze')
#设置背景网格线颜色，样式，尺寸和透明度
    plt.grid(color='#95a5a6',linestyle='--', linewidth=1,axis='both',alpha=0.4)
#显示图表
    plt.show()
    ws=stageWise(xArr, yArr, step=0.01, numIt=100)
    print "ws(regression coefficient):",ws
    #plotStandRegres(xArr,yArr,ws)
    print "correlation:",calcCorrcoef(xArr,yArr,ws)
    #lwlr
    lwlrTestPlot(xArr,yArr,k=1)

if __name__ == '__main__':
    main()