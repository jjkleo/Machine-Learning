from numpy import *
from sys import *
import os
import operator

def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

#k-近邻算法
def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0] #获取dataSet的个数
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1) #按行相加
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	#print(sortedClassCount)
	return sortedClassCount[0][0]

#将文本记录转化成NumPy
def file2matrix(filename):
	fr=open(filename)
	arrayOLines=fr.readlines()
	numberOfLines=len(arrayOLines) #得到文件行数
	returnMat=zeros((numberOfLines,3)) #创建numberOfLines×3的零矩阵
	classLabelVector=[]
	index=0
	for line in arrayOLines:
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat,classLabelVector

#数据归一化
def autoNorm(dataSet):
	#每列最小值
	minVals=dataSet.min(0)
	#每列最大值
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals

#分类器针对约会网站的测试代码
def datingClassTest():
	hoRatio=0.10
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print ("the classifier came back with: %d, the real answer is %d" %(classifierResult,datingLabels[i]))
		if classifierResult!=datingLabels[i]:
			errorCount+=1.0
	print ("the total error rate is: %f" %(errorCount/float(numTestVecs)))

#约会网站预测函数
def classifyPerson():
	resultList=['not at all','in small doses','in large doses']
	percentTats=float(input("percentage of time spent playing video games?"))
	ffMiles=float(input("frequent flier miles earned per year?"))
	iceCream=float(input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=array([ffMiles,percentTats,iceCream])
	classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print ("You will probably like this persion: ", resultList[classifierResult-1])
	
def img2vector(filename):
	returnVect=zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect

#手写数字识别系统的测试代码
def handwritingClassTest():
	hwLabels=[]
	trainingFileList=os.listdir('trainingDigits')
	m=len(trainingFileList)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i:]=img2vector('trainingDigits/%s' %fileNameStr)
	testFileList=os.listdir('testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%s' %fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print ("the classifier came back with: %d, the real answer is %d" %(classifierResult,classNumStr))
		if classifierResult!=classNumStr:
			errorCount+=1.0
	print ('the total number of errors is: %d' % errorCount)
	print ("the total error rate is: %f" %(errorCount/float(mTest)))
	
	

	
	
	
	
	
	
	

	