import pandas as pd

url = "\Users\chett\Downloads\mcdonalds.csv"
mcdonalds = pd.read_csv(url, index_col=0)

print(mcdonalds.columns.tolist())

print(mcdonalds.shape)

print(mcdonalds.head(3))

import numpy as np


MD_x = mcdonalds.iloc[:, 0:11]

MD_x = (MD_x == "Yes").astype(int)

col_means = MD_x.mean().round(2)

print(col_means)

from sklearn.decomposition import PCA

MD_x = (MD_x == "Yes").astype(int)

pca = PCA()
MD_pca = pca.fit_transform(MD_x)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

summary_df = pd.DataFrame({
    'Standard Deviation': np.sqrt(pca.explained_variance_),
    'Proportion of Variance': explained_variance,
    'Cumulative Proportion': cumulative_variance
})

print(summary_df.round(4))

std_devs = np.sqrt(pca.explained_variance_)
print(np.round(std_devs, 1))

rotation_matrix = pca.components_.T

rotation_df = pd.DataFrame(rotation_matrix, index=MD_x.columns, 
                           columns=[f'PC{i+1}' for i in range(rotation_matrix.shape[1])])

print(rotation_df.round(3))
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

MD_pca = pca.fit_transform(MD_x)

plt.scatter(MD_pca[:, 0], MD_pca[:, 1], color='grey')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA Plot')

rotation_matrix = pca.components_.T

scale_factor = 3  
for i, feature in enumerate(MD_x.columns):
    plt.arrow(0, 0, rotation_matrix[i, 0] * scale_factor, rotation_matrix[i, 1] * scale_factor, 
              color='r', head_width=0.05, head_length=0.1)
    plt.text(rotation_matrix[i, 0] * scale_factor * 1.1, rotation_matrix[i, 1] * scale_factor * 1.1, 
             feature, color='g', ha='center', va='center')

plt.show()

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


np.random.seed(1234)

scaler = StandardScaler()
MD_x_scaled = scaler.fit_transform(MD_x)

n_clusters_range = range(2, 9)
best_kmeans = None
best_inertia = np.inf

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    kmeans.fit(MD_x_scaled)
    if kmeans.inertia_ < best_inertia:
        best_inertia = kmeans.inertia_
        best_kmeans = kmeans

MD_km28 = best_kmeans

labels = pd.Series(MD_km28.labels_)
relabel_map = {old_label: new_label for new_label, old_label in enumerate(labels.unique())}
labels = labels.map(relabel_map)

MD_km28.labels_ = labels.values

print(MD_km28.labels_)

from sklearn.cluster import KMeans

np.random.seed(1234)

n_clusters_range = range(2, 9)
models = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234, verbose=0)
    kmeans.fit(MD_x)
    models.append(kmeans)

best_model = min(models, key=lambda model: model.inertia_)

labels = pd.Series(best_model.labels_)
relabel_map = {old_label: new_label for new_label, old_label in enumerate(labels.unique())}
labels = labels.map(relabel_map)
best_model.labels_ = labels.values

inertia_values = [model.inertia_ for model in models]
plt.plot(n_clusters_range, inertia_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of segments')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.grid(True)
plt.show()

from sklearn.metrics import adjusted_rand_score

np.random.seed(1234)

n_clusters_range = range(2, 9)
models = []
ari_scores = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234, verbose=0)
    kmeans.fit(MD_x)
    models.append(kmeans)

    ari_boot = []
    for _ in range(100):
        indices = np.random.choice(len(MD_x), size=len(MD_x), replace=True)
        X_boot = MD_x.iloc[indices]
        labels_boot = kmeans.predict(X_boot)
        ari_boot.append(adjusted_rand_score(kmeans.labels_, labels_boot))
    ari_scores.append(np.mean(ari_boot))

plt.plot(n_clusters_range, ari_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of segments')
plt.ylabel('Adjusted Rand index')
plt.title('Adjusted Rand Index for KMeans Clustering')
plt.grid(True)
plt.show()

best_model_4 = models[2]  

plt.hist(best_model_4.labels_, bins=np.arange(0.5, 5.5, 1), edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title('Cluster Assignments for 4-cluster Solution')
plt.xlim(0, 4)
plt.show()
from sklearn.metrics import silhouette_samples

stability_scores = silhouette_samples(MD_x, best_model_4.labels_)
plt.bar(range(4), [np.mean(stability_scores[best_model_4.labels_ == i]) for i in range(4)], color='blue')
plt.ylim(0, 1)
plt.xlabel('Segment number')
plt.ylabel('Segment stability')
plt.title('Segment Stability')
plt.show()

from sklearn.mixture import GaussianMixture

gmm_models = []
for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=1234)
    gmm.fit(MD_x)
    gmm_models.append(gmm)

for n_clusters, gmm in zip(n_clusters_range, gmm_models):
    print(f'{n_clusters} clusters: Log Likelihood={gmm.lower_bound_:.3f}, AIC={gmm.aic(MD_x):.3f}, BIC={gmm.bic(MD_x):.3f}')

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_samples
from sklearn.tree import DecisionTreeClassifier
from partykit import ctree
from statsmodels.formula.api import ols

np.random.seed(1234)

n_clusters_range = range(2, 9)
models = []
ari_scores = []

for n_clusters in n_clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=1234)
    kmeans.fit(MD_x)
    models.append(kmeans)

    ari_boot = []
    for _ in range(100):
        indices = np.random.choice(len(MD_x), size=len(MD_x), replace=True)
        X_boot = MD_x.iloc[indices]
        labels_boot = kmeans.predict(X_boot)
        ari_boot.append(adjusted_rand_score(kmeans.labels_, labels_boot))
    ari_scores.append(np.mean(ari_boot))

plt.plot(n_clusters_range, ari_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of segments')
plt.ylabel('Adjusted Rand index')
plt.title('Adjusted Rand Index for KMeans Clustering')
plt.grid(True)
plt.show()

best_model_4 = models[2]  

plt.hist(best_model_4.labels_, bins=np.arange(0.5, 5.5, 1), edgecolor='black')
plt.xlabel('Cluster')
plt.ylabel('Frequency')
plt.title('Cluster Assignments for 4-cluster Solution')
plt.xlim(0, 4)
plt.show()

stability_scores = silhouette_samples(MD_x, best_model_4.labels_)
plt.bar(range(4), [np.mean(stability_scores[best_model_4.labels_ == i]) for i in range(4)], color='blue')
plt.ylim(0, 1)
plt.xlabel('Segment number')
plt.ylabel('Segment stability')
plt.title('Segment Stability')
plt.show()

gmm_models = []
for n_clusters in n_clusters_range:
    gmm = GaussianMixture(n_components=n_clusters, n_init=10, random_state=1234)
    gmm.fit(MD_x)
    gmm_models.append(gmm)

for n_clusters, gmm in zip(n_clusters_range, gmm_models):
    print(f'{n_clusters} clusters: Log Likelihood={gmm.lower_bound_:.3f}, AIC={gmm.aic(MD_x):.3f}, BIC={gmm.bic(MD_x):.3f}')

mcdonalds['Like.n'] = 6 - mcdonalds['Like']
formula = 'Like.n ~ yummy + convenient + spicy + fattening + greasy + fast + cheap + tasty + expensive + healthy + disgusting'
model = ols(formula, data=mcdonalds).fit()
print(model.summary())

plt.figure(figsize=(10, 6))
plt.scatter(visit, like, s=10 * female, alpha=0.5)
plt.xlim(2, 4.5)
plt.ylim(-3, 3)
plt.xlabel('Average Visit Frequency')
plt.ylabel('Average Like Rating')
plt.title('Segment Characteristics')
for i, txt in enumerate(range(1, 5)):
    plt.text(visit[i], like[i], txt)
plt.show()

tree = DecisionTreeClassifier(random_state=1234)
X = mcdonalds[['Like.n', 'Age', 'VisitFrequency', 'Gender']]
y = mcdonalds['k4']  
tree.fit(X, y)


plt.figure(figsize=(12, 8))
plot_tree(tree, feature_names=X.columns, class_names=True, filled=True)
plt.title('Decision Tree for Segment Classification')
plt.show()

