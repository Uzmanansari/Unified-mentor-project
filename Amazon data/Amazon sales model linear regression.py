# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 21:00:25 2024

@author: uzman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import plotly.express as px

# Load the dataset
data = pd.read_csv('Amazon Sales data.csv')

# Generate a profiling report
profile = ProfileReport(data, title="Amazon Sales Data Report")
profile.to_file("amazon_sales_data_report.html")

# Data Cleaning and Transformation
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

# Create new features
data['Order Month'] = data['Order Date'].dt.month
data['Order Year'] = data['Order Date'].dt.year

# Fill missing values if any
data.fillna(0, inplace=True)

# ETL: Transform and Load (Store the transformed data)
transformed_data = data.copy()
transformed_data.to_csv('transformed_amazon_sales_data.csv', index=False)

# Sales Trend Analysis
# Month-wise Analysis
monthly_sales = data.groupby(['Order Year', 'Order Month'])['Total Revenue'].sum().reset_index()
monthly_sales['Date'] = pd.to_datetime(monthly_sales['Order Year'].astype(str) + '-' + monthly_sales['Order Month'].astype(str))

# Visualize monthly sales trends
plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='Date', y='Total Revenue')
plt.title('Monthly Sales Trend')
plt.xlabel('Date')
plt.ylabel('Total Revenue')
plt.show()

# Year-wise Analysis
yearly_sales = data.groupby('Order Year')['Total Revenue'].sum().reset_index()

# Visualize yearly sales trends
plt.figure(figsize=(12, 6))
sns.barplot(data=yearly_sales, x='Order Year', y='Total Revenue')
plt.title('Yearly Sales Trend')
plt.xlabel('Year')
plt.ylabel('Total Revenue')
plt.show()

# Yearly Month-wise Analysis
yearly_monthly_sales = data.groupby(['Order Year', 'Order Month'])['Total Revenue'].sum().reset_index()

# Visualize the combined trend to identify seasonal patterns
plt.figure(figsize=(12, 6))
sns.lineplot(data=yearly_monthly_sales, x='Order Month', y='Total Revenue', hue='Order Year', marker='o')
plt.title('Yearly Month-wise Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Revenue')
plt.legend(title='Year')
plt.show()

# Key Metrics Identification
total_sales = data['Total Revenue'].sum()
average_order_value = data['Total Revenue'].mean()
profit_margin = data['Total Profit'].sum() / data['Total Revenue'].sum()

top_selling_items = data.groupby('Item Type')['Total Revenue'].sum().sort_values(ascending=False).head(10)
top_selling_regions = data.groupby('Region')['Total Revenue'].sum().sort_values(ascending=False).head(10)
top_sales_channels = data.groupby('Sales Channel')['Total Revenue'].sum().sort_values(ascending=False)

print(f"Total Sales: {total_sales}")
print(f"Average Order Value: {average_order_value}")
print(f"Profit Margin: {profit_margin}")
print("Top Selling Items:\n", top_selling_items)
print("Top Selling Regions:\n", top_selling_regions)
print("Top Sales Channels:\n", top_sales_channels)

# Relationships Analysis
# Correlation analysis between attributes
correlation_matrix = data[['Units Sold', 'Unit Price', 'Unit Cost', 'Total Revenue', 'Total Cost', 'Total Profit']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Machine Learning Models
# Regression Models
X = data[['Units Sold', 'Unit Price', 'Unit Cost']]
y = data['Total Revenue']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Clustering Models
# K-Means Clustering
clustering_features = data[['Units Sold', 'Total Revenue', 'Total Profit']]
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(clustering_features)

# Visualizing Clusters
plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='Total Revenue', y='Total Profit', hue='Cluster', palette='viridis')
plt.title('K-Means Clustering')
plt.xlabel('Total Revenue')
plt.ylabel('Total Profit')
plt.legend(title='Cluster')
plt.show()

# Save the transformed data with clusters
data.to_csv('/mnt/data/clustered_amazon_sales_data.csv', index=False)
