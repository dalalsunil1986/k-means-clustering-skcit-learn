import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("car_sales.csv")

data = data.values[:, [7, 13]]

kmeans = KMeans(n_clusters=2)
kmeans.fit(data)

def cluster_data():
    result = kmeans.fit_predict(data)
    cluster_1 = []
    cluster_2 = []
    for i in range(len(data)):
        if result[i] == 0:
            cluster_1.append(data[i])
        else:
            cluster_2.append(data[i])
    return cluster_1, cluster_2
