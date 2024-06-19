import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

data = pd.read_csv('water_potability.csv')
print(data.shape)

data = data.fillna(data.mean())

x = data.drop('Potability',axis = 1)
y = data['Potability']

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,shuffle=True,random_state=None)

# Random forest....
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
random_forest = rf.predict(x_test)
random_forest_score = accuracy_score(random_forest,y_test)*100
print(f'Random Forest Accuracy: {random_forest_score:.2f}%')

joblib.dump(rf, 'model.pkl')
