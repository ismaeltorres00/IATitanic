import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv('train.csv')

dataset = dataset.dropna()

print(dataset.info)

X = dataset.drop(['Name', 'Ticket', 'Cabin'], axis=1) 
y = dataset['Survived']

label_encoders = {}
columns_encode = ['Embarked','Sex']

for column in columns_encode:
    label_encoders[column] = LabelEncoder()
    dataset[column] = label_encoders[column].fit_transform(dataset[column])
    
for column in columns_encode:
    original_mapping = {label: original_label for label, original_label in zip(label_encoders[column].classes_, label_encoders[column].inverse_transform(dataset[column].unique()))}
    print(f'Mapeo original para {column}: {original_mapping}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()

model.fit(X_train, y_train)

resultado = model.score(X_test,y_test)

print(resultado)
#y_pred = model.predict(X_test)

