import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv('BankNote_Authentication.csv')
print(df.head())


variance = df["variance"]
score = df["class"]
variance_train, variance_test, score_train, score_test = train_test_split(variance, score, test_size = 0.25, random_state = 0)


X = np.reshape(variance_train.ravel(), (len(variance_train), 1))
Y = np.reshape(score_train.ravel(), (len(score_train), 1))

classifier = LogisticRegression(random_state = 0)
classifier.fit(X, Y)

X_test = np.reshape(variance_test.ravel(), (len(variance_test), 1))
Y_test = np.reshape(score_test.ravel(), (len(score_test), 1))

heart_attack_prediction = classifier.predict(X_test)

predicted_values = []
for i in heart_attack_prediction:
    if i == 0:
        predicted_values.append("Forged")            
    else:
        predicted_values.append("Authorized")

actual_values = []
for i in Y_test.ravel():
    if i == 0:
        actual_values.append("Forged")
    else:
        actual_values.append("Authorized")


labels = ["Authorized", "Forged"]

cm = confusion_matrix(actual_values, predicted_values)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
plt.show()