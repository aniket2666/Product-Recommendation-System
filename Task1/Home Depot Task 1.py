# Importing Libraries

# %%
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# %%
product_description = pd.read_csv('product_descriptions.csv')
product_description.shape

# %%
product_description = product_description.dropna()
product_description.shape
product_description.head()

# %%
product_description1 = product_description.head(500)
product_description1["product_description"].head()

# %%
vectorizer = TfidfVectorizer(stop_words = 'english')
x1 = vectorizer.fit_transform(product_description1["product_description"])
x1

# %%
x=x1
kmeans = KMeans(n_clusters=10, init = 'k-means++')
y_kmeans = kmeans.fit_predict(x)
plt.plot(y_kmeans, ".")
plt.show()

# %%
def print_cluster(i):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

# %%
true_k = 10

model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(x1)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print_cluster(i)

# %%
def show_recommendations(product):
    y = vectorizer.transform([product])
    prediction = model.predict(y)
    print_cluster(prediction[0])
# Try these Outputs
# %%
show_recommendations('water')

# %%
show_recommendations('spray paint')


