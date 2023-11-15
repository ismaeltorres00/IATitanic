import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


dataset = pd.read_csv("train.csv")

# Imputación para 'Age' Faltan 177 por lo que hago la media ya que es una factor determinante
imputer = SimpleImputer(strategy='median')
dataset['Age'] = imputer.fit_transform(dataset[['Age']])

#fdvj Extrayendo títulos de 'Name'
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Revisión de la columna 'Ticket'
ticket_unique = dataset['Ticket'].nunique()

# Eliminar columnas innecesarias
dataset = dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1) 

# Codificación de variables categóricas
label_encoders = {}
columns_encode = ['Embarked', 'Sex', 'Title']

for column in columns_encode:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])

X = dataset.drop('Survived', axis=1) 
y = dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

resultado = model.score(X_test, y_test)

#print(resultado)

pickle.dump(model, open('model.pkl','wb'))

