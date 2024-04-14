# %% [markdown]
# Heirarchical Clustersing

# %% [markdown]
# importing the libraries

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# importing the dataset

# %%
df=pd.read_csv("Mall_Customers.csv")
df.head()

# %%
x=df.iloc[:,[3,4]].values
x

# %% [markdown]
# Using the Dendogram to find the optimal cluster numbers

# %%
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method="ward"))
plt.xlabel("Customers")
plt.ylabel("Euclidean distance")
plt.title("Dendrogram")
plt.savefig("Dendogram.png")
plt.show()

# %%
#? can use 3 or 5 for clustering by measuring the length of the cluster


# %% [markdown]
# Training the Hierarchical Clustering Algorithm

# %%
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
y_hc

# %% [markdown]
# Visualizing the Hierarchical Clustering

# %%
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c="red",label="cluster 1" , s=100)
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c="blue",label="cluster 2" , s=100)
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c="green",label="cluster 3" , s=100)
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],c="orange",label="cluster 4" , s=100)
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],c="purple",label="cluster 5" , s=100)
plt.title("Cluster of Customers")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (0-100)")
plt.legend(loc="upper right")
plt.savefig("Cluster of Customers - 5")
plt.show()

# %% [markdown]
# Now trying for clusters = 3

# %% [markdown]
# Training the Hierarchical Clustering Algorithm

# %%
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
y_hc

# %% [markdown]
# Visualizing the Hierarchical Clustering

# %%
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],c="red",label="cluster 1" , s=100)
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],c="blue",label="cluster 2" , s=100)
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],c="green",label="cluster 3" , s=100)
plt.title("Cluster of Customers")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score (0-100)")
plt.legend(loc="upper right")
plt.savefig("Cluster of Customers - 3")
plt.show()


