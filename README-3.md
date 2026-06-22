# Customer Segmentation using K-Means

## Project Overview
This project demonstrates how K-Means clustering can be used to group customers based on their income and spending behavior.

## Dataset
The dataset contains the following features:

- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## Goal
The goal is to identify customer segments without predefined labels using unsupervised machine learning.

## Machine Learning Technique
### K-Means Clustering

K-Means is an unsupervised learning algorithm that divides data into K clusters by minimizing the distance between data points and cluster centroids.

## Project Steps

1. Load the dataset
2. Perform exploratory data analysis (EDA)
3. Select relevant features
4. Use the Elbow Method to find the optimal number of clusters
5. Train a K-Means model
6. Visualize customer segments
7. Interpret cluster characteristics

## Example Code

```python
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("Mall_Customers.csv")

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

model = KMeans(n_clusters=5, random_state=42)
df["Cluster"] = model.fit_predict(X)

print(df.head())
```

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Expected Outcome

The model groups customers into different segments such as:

- High income, high spending
- High income, low spending
- Low income, high spending
- Low income, low spending
- Average customers

## Author

GitHub portfolio project for demonstrating clustering and unsupervised learning skills.
