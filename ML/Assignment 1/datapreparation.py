import pandas as pd
import matplotlib as plt
import seaborn as sns
df = pd.read_csv('Heart.csv')
print(df.head())

#1.Print Shape of Data
print(df.shape)

#2.Finding Null values or missing values using isnull()
print(df.isnull().sum())
#or
print(df.count())

#3. Find the datatype of the columns
print(df.dtypes)

#4. Finding out Zeros
print("Total Number of Zeros=",df[df == 0].count())

#5. Finding mean age of patients
print("Mean=",df['Age'].mean())

#Extract only Age, Sex, Chestpain, RestBP, Chol. 
#Randomly divide dataset in training(75%) and test(25%)
newdf = df[['Age','Sex','ChestPain','RestBP','Chol']]
print(newdf)

#cross validation
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, random_state=0, test_size=0.25)
print("Train Data",train.shape)
print("Test Set",test.shape)


print("________________________________________________________________________________")


#part b
import numpy as np
actual = list(np.ones(45)) + list(np.zeros(55))
print("ACTUAL->")
print(np.array(actual))

predicted = list(np.ones(40)) + list(np.zeros(52)) + list(np.ones(8))
print("PREDICTED->")
print(np.array(predicted))

#Plotting the confusion metrics
import matplotlib.pyplot as plt
from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()

#Classification report (Precision, Recall, F1-Score, Accuracy)
from sklearn.metrics import classification_report
print(classification_report(actual, predicted))

from sklearn.metrics import accuracy_score
print(accuracy_score(actual, predicted))
