# -*- coding: utf-8 -*-
"""
Spyder Editor

Author:Vatsal Mehta.
"""

import numpy as np
import pandas as pd

df=pd.read_csv('Data.csv')
print("\n")

print(df.columns)

print("\n")

print(df.shape)

print("\n")

print(df.corr())
print("\n")

print("Max Salary value is in row number: ",df['Salary'].idxmax()) #it will print the  row number of maximum value in that column

print("Salary to age ratio is",df['Salary']/df['Age'])

print("\n")
print(df.isnull().sum())

print("\n")
print(df['Salary'].isnull())


#method 1

df.loc[4,'Salary']=40000

print(df)

#method 2

df['Age'].fillna(45,inplace=True)

print(df)

#method 3

df.dropna(thresh=3)


#Segregation of input and output variables

X=df.iloc[:,0:3].values
y=df.iloc[:,3].values


#method 4

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

print("\n")


"""Encoding Categorical Data for input data present in X variable """

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X=np.array(ct.fit_transform(X))

print(X)

"""One hot Encoder basically splits one single column into multiple columns having numerical values"""

print("\n")

"""Encoding Categorical Data for output data present in y variable """


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)

print(y)

"""Label Encoder basically labels string values into numerical values and does not split the columns"""






















