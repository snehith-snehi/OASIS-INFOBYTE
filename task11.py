import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv("Iris.csv")

# EDA
print(df['Species'].value_counts())
sns.pairplot(df, hue='Species')
plt.show()
# FIXED CORR:
print(df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].corr()['PetalLengthCm'].sort_values(ascending=False))

# Rest unchanged...
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['Species'])
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['species_encoded']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(confusion_matrix(y_test, y_pred))

new_flower = np.array([[5.1, 3.5, 1.4, 0.2]])
pred_species = le.inverse_transform(model.predict(new_flower))
print('Predicted:', pred_species[0])