import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Load data (replace with your path)
df = pd.read_csv('Advertising.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)  # Clean index column if present

# EDA
print(df.describe())
sns.pairplot(df)
plt.show()
print(df.corr()['Sales'].sort_values(ascending=False))

# Prepare data
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print('MSE:', mean_squared_error(y_test, y_pred))
print('R2:', r2_score(y_test, y_pred))
print('Coefficients:', dict(zip(X.columns, model.coef_)))

# Predict new sales
new_ads = np.array([[100, 25, 10]])  # TV=100k, radio=25k, newspaper=10k
print('Predicted sales:', model.predict(new_ads))