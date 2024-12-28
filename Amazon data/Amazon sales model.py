# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 20:37:14 2024

@author: uzman
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load the dataset
data = pd.read_csv('Amazon Sales data.csv')

# Data cleaning and preprocessing
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

data

#pip install ydata-profiling

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ydata_profiling import ProfileReport

# Load your dataset
data

# Generate a profiling report
profile = ProfileReport(data, title="Amazon Sales Data Report")

# Save the report to an HTML file
profile.to_file("amazon_sales_data_report.html")

# To view the report in a Jupyter notebook, you can use:
profile.to_notebook_iframe()

# Visualizations
sns.histplot(data['Total Revenue'])
plt.show()

data['Order Month'] = data['Order Date'].dt.month
data['Order Year'] = data['Order Date'].dt.year

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

# Regression Example
X = data[['Units Sold', 'Unit Price', 'Unit Cost']]
y = data['Total Revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

import plotly.express as px

fig = px.line(data, x='Order Date', y='Total Revenue', title='Sales Trend Over Time')
fig.show()
