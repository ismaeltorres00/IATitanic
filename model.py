from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import pickle

data = pd.read_csv("train.csv")

# Imputar valores faltantes para 'Age' y 'Embarked'
imputer_age = SimpleImputer(strategy='median')
data['Age'] = imputer_age.fit_transform(data[['Age']])

imputer_embarked = SimpleImputer(strategy='most_frequent')
data['Embarked'] = imputer_embarked.fit_transform(data[['Embarked']])[:, 0]

# Ingeniería de características: Crear una nueva característica 'FamilySize'
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
data['Sex'] = data['Sex'].map({'female': 0, 'male': 1})

# Mostrar las primeras filas del dataset después de las transformaciones
data.head()

# Selección de características (excluyendo 'Name', 'Ticket', y 'PassengerId')
features = data.drop(['Survived', 'Name', 'Ticket', 'PassengerId','Cabin'], axis=1)
target = data['Survived']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [50, 100, 150],  # Diferentes números de árboles
    'max_depth': [None, 10, 20, 30],  # Diferentes profundidades máximas
    'min_samples_split': [2, 5, 10],  # Diferentes cantidades mínimas de muestras para dividir un nodo
    'min_samples_leaf': [1, 2, 4]  # Diferentes cantidades mínimas de muestras en las hojas
}

'''
# Crear y entrenar el modelo de Bosque Aleatorio
model = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(features, target)
print("Mejores hiperparámetros:", grid_search.best_params_)
print("Mejor Puntaje de Validación Cruzada:", grid_search.best_score_)

scores = cross_val_score(model, features, target, cv=5)
print("Puntajes de Validación Cruzada:", scores)
# Calcular el puntaje promedio de validación cruzada
mean_score = scores.mean()
print("Puntaje Promedio de Validación Cruzada:", mean_score)
'''
model = RandomForestClassifier(max_depth= 20, min_samples_leaf= 1, min_samples_split= 10, n_estimators=100)

model.fit(X_train, y_train)

# Puntuación
score = model.score(X_test, y_test)
print(score)

# Exportamos
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Cabeceras para saber los datos que tenemos para predecir
feature_names = X_train.columns
print(feature_names)