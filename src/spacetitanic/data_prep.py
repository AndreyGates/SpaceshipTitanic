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

    X = df.drop(['Id'], axis=1)
    # X = X.drop(ordinal_columns_str, axis=1)
    y = None

    if csv_path == '..SpaceTitanic/src/train.csv':
        X = X.drop(['SalePrice'], axis=1)
        y = df['SalePrice']

    return X, y


