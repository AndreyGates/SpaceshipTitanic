from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.svm import SVC
from data_prep import prepared_data
import pandas as pd
import csv

def fill_csv(PassengerId_test, y_test):
    '''FILLING IN CSV FOR SUBMISSION'''
    header = ['PassengerId', 'Transported']
    data = zip(PassengerId_test, y_test)

    with open('../SpaceTitanic/src/predictions.csv', 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(data)

# modelling
X_train, y_train, X_test, y_test = prepared_data()

model = AdaBoostClassifier(learning_rate=0.1, n_estimators=250)

svc = model.fit(X_train, y_train)
y_test = svc.predict(X_test)

# filling csv for submission
df = pd.read_csv('../SpaceTitanic/src/test.csv')
PassengerId_test = df['PassengerId'].tolist()
y_test = list(map(lambda x : x == 1, y_test)) # converting to boolean
fill_csv(PassengerId_test, y_test)
