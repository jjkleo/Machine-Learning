from math import log
import operator

def createDataSet():
	dataSet=[[1,1,'yes'],
			 [1,1,'yes'],
			 [1,0,'no'],
			 [0,1,'no'],
			 [0,1,'no']]
	labels=['no surfacing','flippers']
	return dataSet,labels

#计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	#数据集大小
	numEntries=len(dataSet)
	labelCounts={}
	for featVec in dataSet:
		currentLabel =featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	shannonEnt=0.0
	for key in labelCounts:
		prop=float(labelCounts[key])/numEntries
		shannonEnt -=prop*log(prop,2)
	return shannonEnt
#按照给定特征划分数据集
def splitDataSet(dataSet,axis,value):
	retDataSet=[]
	for featVec in dataSet:
		if featVec[axis]==value:
			reducedFeatVec=featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet
#选择最好的数据集划分
def chooseBestFeatureToSplit(dataSet):
	numFeatures=len(dataSet[0])-1
	baseEntropy=calcShannonEnt(dataSet)
	bestInfoGain=0.0;bestFeature=-1
	for i in range(numFeatures):
		featList=[example[i] for example in dataSet]
		uniqueVals=set(featList)
		newEntropy=0.0
		for value in uniqueVals:
			subDataSet=splitDataSet(dataSet,i,value)
			prop=len(subDataSet)/float(len(dataSet))
			newEntropy+=prop*calcShannonEnt(subDataSet)
		infoGain=baseEntropy-newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain=infoGain
			bestFeature=i
	return bestFeature

#多数表决
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote]=0
		classCount[vote]+=1
	sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

#创建树
def createTree(dataSet,labels):
	classList=[example[-1] for example in dataSet]
	if classList.count(classList[0])==len(classList):
		return classList[0]
	if len(dataSet[0])==1:
		return majorityCnt(classList)
	bestFeat=chooseBestFeatureToSplit(dataSet)
	bestFeatLabel=labels[bestFeat]
	myTree={bestFeatLabel:{}}
	del(labels[bestFeat])
	featValues=[example[bestFeat] for example in dataSet]
	uniqueVals=set(featValues)
	for value in uniqueVals:
		subLabels=labels[:]
		myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
	return myTree

#使用决策树的分类函数
def classify(inputTree,featLabels,testVec):
	firstStr=list(inputTree.keys())[0]
	secondDict=inputTree[firstStr]
	featIndex=featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex]==key:
			if type(secondDict[key]).__name__=='dict':
				classLabel=classify(secondDict[key],featLabels,testVec)
			else:
				classLabel=secondDict[key]
	return classLabel

#使用pickle模块存储决策树
def storeTree(inputTree,filename):
	import pickle
	fw=open(filename,'wb')
	pickle.dump(inputTree,fw)
	fw.close()

def grabTree(filename):
	import pickle
	fr=open(filename,'rb')
	return pickle.load(fr)



		
		
		
		
		
		
		
		
		
		
		
		