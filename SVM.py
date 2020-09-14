import pandas as pd 
cancer_data = pd.read_csv("C:/Users/uamarh1/Pictures/train.csv")

cancer_data.head()

cancer_clear=cancer_data.dropna()

cancer_clear.head(20)

cancer_clear = cancer_clear.iloc[:,~cancer_clear.columns.isin(['tripid'])]
cancer_clear = cancer_clear.iloc[:,~cancer_clear.columns.isin(['drop_time'])]
cancer_clear = cancer_clear.iloc[:,~cancer_clear.columns.isin(['pickup_time'])]

cancer_clear.head()

X = cancer_clear.drop('label', axis=1)
Y = cancer_clear['label']
Y.head()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)

X_train = scaler.transform(X)

Y = Y.replace(to_replace=['correct', 'incorrect'], value=[1, 0])

Y.head()

from sklearn.svm import SVC # "Support vector classifier"  
classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(X_train,Y) 

test_data = pd.read_csv("C:/Users/uamarh1/Pictures/test.csv")
test_data = test_data.iloc[:,~test_data.columns.isin(['tripid'])]
test_data = test_data.iloc[:,~test_data.columns.isin(['drop_time'])]
test_data = test_data.iloc[:,~test_data.columns.isin(['pickup_time'])]

y_pred= classifier.predict(test_data) 

y_pred.head()

y_pred

final = pd.DataFrame(data=y_pred, columns=["prediction"])

final.head()

final.to_csv(r'C:/Users/uamarh1/Pictures/final.csv')

