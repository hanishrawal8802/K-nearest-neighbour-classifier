from turtle import color
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
df = pd.read_csv('Classified Data')
print(df.head())
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()                                ## To select the nearest K value we first standarize our data and sort them ##
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_feature = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_feature,columns=df.columns[:-1])
print(df_feat.head())
print(df_feat.tail())
X =df_feat
y = df['TARGET CLASS']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
Knn= KNeighborsClassifier(n_neighbors=1)
Knn.fit(X_train,y_train)
pred = Knn.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))

error_rate = []                                        ## To check the error rate between the range of 1 to 40 by plotting##
for i in range(1,40):
    Knn = KNeighborsClassifier(n_neighbors=i)
    Knn.fit(X_train,y_train)
    pred_i = Knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,'b--',marker='o',markersize =10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error rate')
plt.show()

Knn= KNeighborsClassifier(n_neighbors=30)       ## at K=30 we got the minimum error from plot and accuracy is also increased by 3% ,by comparing the previous result where k=1
Knn.fit(X_train,y_train)
pred = Knn.predict(X_test)
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))