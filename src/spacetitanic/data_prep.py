import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_selector as selector


def data_extract(csv_path):
    '''DATA EXTRACTION'''
    df = pd.read_csv(csv_path)

    # df = df.dropna(axis=1, thresh=len(df.values)/1.5) # dropping columns where 1/3 is Nan or more)
    # df = df.fillna(df.mean()) # imputing missing numerical values with feature means
    # df = df.drop(1379) # df[1379]['Electrical'] is the only empty string in the column

    X = df.drop(['PassengerId', 'Name'], axis=1)
    # X = X.drop(ordinal_columns_str, axis=1)
    y = None

    if csv_path == '../SpaceTitanic/src/train.csv':
        X = X.drop(['Transported'], axis=1)
        y = df['Transported']

    return X, y


def manual_transformer(X):
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
    prep = data_preprocessor()
    X, y = data_extract('../SpaceTitanic/src/train.csv')
    X = manual_transformer(X)
    X = prep.fit_transform(X)

    return X, y
