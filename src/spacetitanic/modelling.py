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
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                          colsample_bylevel=0.7940834865361239, colsample_bynode=1,
                          colsample_bytree=0.7565648948818255, early_stopping_rounds=None,
                          enable_categorical=False, eval_metric=None,
                          gamma=0.9781032170729997, gpu_id=-1, grow_policy='depthwise',
                          importance_type=None, interaction_constraints='',
                          learning_rate=0.058171277286333936, max_bin=256,
                          max_cat_to_onehot=4, max_delta_step=0, max_depth=1, max_leaves=0,
                          min_child_weight=3, monotone_constraints='()',
                          n_estimators=3000, n_jobs=1, num_parallel_tree=1,
                          predictor='auto', random_state=2, reg_alpha=0.052678020132594046,
                          reg_lambda=1.4283842692755349)

svc = model.fit(X_train, y_train)
y_test = svc.predict(X_test)

# filling csv for submission
df = pd.read_csv('../SpaceTitanic/src/test.csv')
PassengerId_test = df['PassengerId'].tolist()
y_test = list(map(lambda x : x == 1, y_test)) # converting to boolean
fill_csv(PassengerId_test, y_test)
