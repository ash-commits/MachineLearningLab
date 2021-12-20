import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

msg = pd.read_csv(r"C:\Users\sivaa\Downloads\text_classification.csv",names = ['message','label'])
print('Dimension of dataset', msg.shape)

msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})
x = msg.message
y = msg.labelnum

x_train, x_test, y_train, y_test = train_test_split(x,y)
print("\n total no.of training data: ", y_train.shape)
print("\n total no.of testing data: ", y_test.shape)

cv = CountVectorizer()
x_train_dtm = cv.fit_transform(x_train)
x_test_dtm = cv.transform(x_test)
print("\n no.of words or tokens in document: ", cv.get_feature_names())
df = pd.DataFrame(x_train_dtm.toarray(),columns=cv.get_feature_names())
clf = MultinomialNB().fit(x_train_dtm,y_train)
predicted = clf.predict(x_test_dtm)

print("\n Accuracy of classifier is: ",metrics.accuracy_score(y_test,predicted))
print("\n Confusion Matrix")
print(metrics.confusion_matrix(y_test,predicted))
print("\n Value of Precision: ",metrics.precision_score(y_test,predicted))
print("\n Value of Recall: ",metrics.recall_score(y_test,predicted))



