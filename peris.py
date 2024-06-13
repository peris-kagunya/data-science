import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# Load the data
data = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Displaying the first few rows
print(data.head())

# Print column names and info
print(data.columns)
print(data.info())

# Data Cleaning and Preparation
data['CustomerID'] = data['CustomerID'].astype(str)
data['Amount'] = data['Quantity'] * data['UnitPrice']

# Data Analysis
# Step 1: Identify the most bought product
most_bought_product = data.groupby('Description')['Quantity'].sum().idxmax()
print("Most bought product:", most_bought_product)

# Step 2: Sort the data by Amount and Quantity
sorted_by_amount = data.groupby('Description')['Amount'].sum().sort_values(ascending=False)
print("Top 5 products by amount:\n", sorted_by_amount.head())

sorted_by_quantity = data.groupby('Description')['Quantity'].sum().sort_values(ascending=False)
print("Top 5 products by quantity:\n", sorted_by_quantity.head())

# Step 3: Sort the data by Country and Quantity
sorted_by_country_quantity = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
print("Top 5 countries by quantity:\n", sorted_by_country_quantity.head())

#Frequently sold item
most_frequent_item = data['Description'].value_counts().idxmax
print("Most frequently sold item:", most_frequent_item)

#Total sales
from datetime import date

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')

# Display the converted DataFrame
print(data.head())
# Find the maximum date in 'InvoiceDate'
max_date = data['InvoiceDate'].max()
print("Maximum date in InvoiceDate:", max_date)
# Find the maximum date in 'InvoiceDate'
min_date = data['InvoiceDate'].min()
print("Minimum date in InvoiceDate:", min_date)
# Calculate the difference in days between the maximum and minimum dates
date_difference = (max_date - min_date).days
print("Difference in days between max and min dates:", date_difference)

# Calculate the total sales
total_sales = data['Amount'].sum()
print("Total sales:", total_sales)

#The day that the maximum was recorded
date =max_date - pd.Timedelta(days=30)
print(date)
days = max_date-date
print(days) 

# Calculate the sales for the last 30 days from the maximum date
sales_last_30_days = data[data['InvoiceDate'] >= max_date - pd.DateOffset(days=30)]['Amount'].sum()
print("Sales for the last 30 days:", sales_last_30_days)




# Select relevant features for clustering
X = data[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Display the first few rows with cluster labels
print(data.head())

# Analyze the clusters
cluster_counts = data['Cluster'].value_counts()
print("Cluster Counts:\n", cluster_counts)

# Load the data
data = pd.read_csv('OnlineRetail.csv', encoding='ISO-8859-1')

# Data Cleaning and Preparation
data['CustomerID'] = data['CustomerID'].astype(str)
data['Amount'] = data['Quantity'] * data['UnitPrice']

# Select relevant features for clustering
X = data[['Quantity', 'UnitPrice']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))

# Scatter plot of the data points colored by cluster
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=200, c='red', label='Centroids')
plt.title('K-Means Clustering')
plt.xlabel('Quantity (Standardized)')
plt.ylabel('Unit Price (Standardized)')
plt.legend()
plt.grid(True)
plt.show()

# Bar chart of cluster counts
plt.figure(figsize=(8, 5))
cluster_counts = data['Cluster'].value_counts()
cluster_counts.plot(kind='bar', color='skyblue')
plt.title('KMeans: Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()


def time_series_analysis():

    # Convert 'InvoiceDate' column to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    # Set 'InvoiceDate' as index
    data.set_index('InvoiceDate', inplace=True)

    # Aggregate data by day
    daily_sales = data.resample('D').sum()

    # Plot the time series
    plt.figure(figsize=(10, 6))
    plt.plot(daily_sales.index, daily_sales['Quantity'], color='blue')
    plt.title('Time Series: Daily Sales')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.show()

    # Perform decomposition to analyze trends, seasonality, and residuals
    result = seasonal_decompose(daily_sales['Quantity'], model='additive', period=30)
    result.plot()
    plt.show()



time_series_analysis()




