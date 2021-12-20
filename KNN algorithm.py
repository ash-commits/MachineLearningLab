from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
print("Iris Dataset Loaded")

xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target,test_size=0.2)
print("Dataset is split into training and testing")
print('Size of training data and its label', xtrain, ytrain)
print('Size of testing data and its label', xtest, ytest)

for i in range(len(iris.target_names)):
    print('Label',i,'-',str(iris.target_names[i]))
classifier = KNeighborsClassifier(n_neighbors=1)
classifier.fit(xtrain,ytrain)
ypred = classifier.predict(xtest)
print("results of classification using KNN with k = 1: ")
for r in range(0,len(xtest)):
    print("sample",str(xtest[r]),"actual-label",ytest[r],"Predicted label",ypred[r])
print("Classification Accuracy: ",classifier.score(xtest,ytest))

from sklearn.metrics import classification_report,confusion_matrix
print("Confusion Matrix: \n",confusion_matrix(ytest,ypred))
print("Accuracy Metrics: \n",classification_report(ytest,ypred))
