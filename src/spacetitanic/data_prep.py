import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector


PassengerId_test = [] # global variable for submission goals


class MyException(Exception): # custom exception
    pass


def data_extract(csv_path):
    '''DATA EXTRACTION'''
    df = pd.read_csv(csv_path)
    X = df.drop(['PassengerId', 'Name'], axis=1)
    y = None

    if csv_path == '../SpaceTitanic/src/train.csv':
        X = X.drop(['Transported'], axis=1)
        y = df['Transported']

    if csv_path != '../SpaceTitanic/src/train.csv' and csv_path != '../SpaceTitanic/src/test.csv':
        MyException('Cannot extract train or test data! The wrong file!')

    return X, y


def manual_transformer(X):
    '''CUSTOM COLUMN TRANSFORMATION'''
    cabin_mapping = lambda s : 0 if s[-1] == 'P' else 1
    X['Cabin'] = X['Cabin'].map(cabin_mapping, na_action='ignore')
    return X


def data_preprocessor():
    '''NUMERIC AND CATEGORICAL VALUES HANDLING (EXCEPT ORDINAL - DONE SEPARATELY)'''
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # ("pca", PCA(n_components=6))
    ])

    '''HANDLING CATEGORICAL DATA'''
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(categories='auto', drop='first'))
    ])

    '''OVERALL PREPROCESSING - COLUMN TRANSFORMATION'''
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, selector(dtype_exclude=object)),
        ('cat', categorical_transformer, selector(dtype_include=object))
    ])

    return preprocessor


def prepared_data():
    '''PREPROCESSING TRAIN AND TEST DATA'''
    prep = data_preprocessor()
    X_train, y_train = data_extract('../SpaceTitanic/src/train.csv')
    X_train = manual_transformer(X_train)
    X_train = prep.fit_transform(X_train)

    X_test, y_test = data_extract('../SpaceTitanic/src/test.csv')
    X_test = manual_transformer(X_test)
    X_test = prep.transform(X_test)

    return X_train, y_train, X_test, y_test
