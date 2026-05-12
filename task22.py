import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load CSV (download from Kaggle)
df = pd.read_csv('Unemployment in India.csv')

# Clean
df['Date'] = pd.to_datetime(df['Date'])  # Fix if needed
print(df.head())
print(df.describe())

# Viz 1: Line chart
fig = px.line(df, x='Date', y='EstimatedUnemploymentRate', color='Region',
              title='Unemployment Rate: COVID Spike')
fig.show()

# FIXED Heatmap
pivot = df.pivot_table(index='Region', columns='Frequency', 
                       values='EstimatedUnemploymentRate', aggfunc='mean')
sns.heatmap(pivot, annot=True, cmap='YlGnBu', fmt='.2f')
plt.title('Avg Unemp Rate by Region/Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Top states
top = df.nlargest(10, 'EstimatedUnemploymentRate')[['Region', 'EstimatedUnemploymentRate', 'Date']]
print(top)

# Boxplot
plt.figure(figsize=(12,5))
sns.boxplot(data=df, x='Region', y='EstimatedLabourParticipationRate')
plt.xticks(rotation=90)
plt.show()