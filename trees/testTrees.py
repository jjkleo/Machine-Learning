import trees
import treePlotter

#测试决策树的运用
#myDat,labels=trees.createDataSet()
#print(labels)
#myTree=treePlotter.retrieveTree(0)
#print(myTree)
#print(trees.classify(myTree,labels,[1,0]))
#print(trees.classify(myTree,labels,[1,1]))

#测试决策树的绘制
#myTree=treePlotter.retrieveTree(0)
#myTree['no surfacing'][3]='mybe'
#treePlotter.createPlot(myTree)

#测试决策树的存储
#trees.storeTree(myTree,'classifierStorage.txt')
#print(trees.grabTree('classifierStorage.txt'))

#使用决策树预测隐形眼镜类型
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)



