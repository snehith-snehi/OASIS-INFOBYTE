import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Load YOUR file
df = pd.read_csv('car data.csv')

# EDA
print(df.head())
print(df['Selling_Price'].describe())
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm')
plt.show()

# Feature Prep
df['Age'] = 2026 - df['Year']  # Current year
df['Price_Depreciation'] = df['Present_Price'] - df['Selling_Price']
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Encode categoricals
le = LabelEncoder()
cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Model data
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost (best for tabular)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print(f'R²: {r2_score(y_test, y_pred):.3f}')  # ~0.85-0.90
print(f'MAE: Rs.{mean_absolute_error(y_test, y_pred):.2f}L')
print('Top Features:', pd.Series(model.feature_importances_, X.columns).sort_values(ascending=False))

# Predict new: e.g., 2020 Swift, Present=7L, 30k km, Petrol/Dealer/Manual/0
new_car = pd.DataFrame({
    'Present_Price': [7.0], 'Driven_kms': [30000], 'Fuel_Type': ['Petrol'],
    'Selling_type': ['Dealer'], 'Transmission': ['Manual'], 'Owner': [0],
    'Age': [6], 'Price_Depreciation': [0]  # Temp
})
for col in ['Fuel_Type', 'Selling_type', 'Transmission', 'Owner']:
    new_car[col] = le.fit_transform(new_car[col])  # Refit for new
print('Predicted Selling_Price: Rs.{:.2f}L'.format(model.predict(new_car)[0]))