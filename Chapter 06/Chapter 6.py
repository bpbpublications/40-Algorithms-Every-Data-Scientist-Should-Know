# Basic Unsupervised Learning Algorithms

# K-means Clustering
# Let us implement the K-Means algorithm using Python and NumPy. We will generate some synthetic data and apply K-Means to cluster the data.
	import numpy as np
	
	def k_means_clustering(X, K, max_iters=100):
	    # X: Input data (numpy array of shape [num_samples, num_features])
	    # K: Number of clusters
	    # max_iters: Maximum number of iterations (default: 100)
	    
	    num_samples, num_features = X.shape
	    
	    # Step 1: Randomly initialize K centroids from the data points
	    initial_centroids_idx = np.random.choice(num_samples, K, replace=False)
	    centroids = X[initial_centroids_idx]
	    
	    for _ in range(max_iters):
	        # Step 2: Assign data points to the nearest centroid
	        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)  # Compute Euclidean distances
	        cluster_assignments = np.argmin(distances, axis=-1)
	        
	        # Step 3: Update centroids
	        new_centroids = np.array([X[cluster_assignments == k].mean(axis=0) for k in range(K)])
	        
	        # Check for convergence
	        if np.allclose(centroids, new_centroids):
	            break
	        
	        centroids = new_centroids
	    
	    return centroids, cluster_assignments
	
	# Generate synthetic data
	np.random.seed(42)
	data = np.random.rand(100, 2)  # 100 data points with 2 features
	
	# Apply K-Means with K=3 clusters
	K = 3
	centroids, cluster_assignments = k_means_clustering(data, K)
	
	# Print the resulting centroids and cluster assignments
	print("Final Centroids:")
	print(centroids)
	print("Cluster Assignments:")
	print(cluster_assignments)

# Real world coding example
# Install the required libraries 
# First, make sure you have scikit-learn and OpenCV installed. If you don't have them, you can install them using pip:
	pip install scikit-learn
	pip install opencv-python
#
# Perform image compression using the following code:
 	import numpy as np
	import cv2
	import matplotlib.pyplot as plt
	from sklearn.cluster import KMeans
	
	def image_compression(image_path, num_colors=16):
	    # Load the image
	    image = cv2.imread(image_path)
	    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    
	    # Reshape the image to be a 2D array of pixels (rows, columns) x 3 (RGB channels)
	    pixels = image.reshape(-1, 3)
	    
	    # Apply K-means clustering to the pixels
	    kmeans = KMeans(n_clusters=num_colors, random_state=42)
	    kmeans.fit(pixels)
	    
	    # Replace each pixel's color with the nearest centroid color
	    compressed_pixels = kmeans.cluster_centers_[kmeans.labels_]
	    
	    # Reshape the compressed pixels back to the original image shape
	    compressed_image = compressed_pixels.reshape(image.shape)
	    
	    return compressed_image
	
	# Load the image and perform compression
	input_image_path = "path_to_your_image.jpg"
	compressed_image = image_compression(input_image_path, num_colors=16)
	
	# Display the original and compressed images
	plt.figure(figsize=(8, 4))
	plt.subplot(1, 2, 1)
	plt.title("Original Image")
	plt.imshow(cv2.imread(input_image_path))
	plt.axis('off')
	
	plt.subplot(1, 2, 2)
	plt.title("Compressed Image")
	plt.imshow(compressed_image)
	plt.axis('off')
	
	plt.show()

# Hierarchical clustering
# Here is a detailed coding example in Python:
# Let us implement the Agglomerative Hierarchical Clustering algorithm using Python and the scikit-learn library. We will generate some synthetic data and apply hierarchical clustering to cluster the data.
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.datasets import make_blobs
	from sklearn.cluster import AgglomerativeClustering
	from scipy.cluster.hierarchy import dendrogram, linkage
	
	# Generate synthetic data with three clusters
	X, y = make_blobs(n_samples=300, centers=3, random_state=42)
	
	# Apply Hierarchical Clustering
	agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
	agg_clustering.fit(X)
	
	# Plot the Dendrogram
	linked = linkage(X, method=’ward’)
	dendrogram(linked, truncate_mode=’lastp’, p=12, show_contracted=True)
	plt.xlabel(“Sample Index”)
	plt.ylabel(“Distance”)
	plt.title(“Dendrogram”)
	plt.show()

# Real world coding example
# In this example, we will use hierarchical clustering in Python to perform disease subtyping using gene expression data. We will use the scikit-learn library for hierarchical clustering and matplotlib for visualization.
# 1.	Install required libraries: First, make sure you have scikit-learn and matplotlib installed. If you don't have them, you can install them using pip:
	pip install scikit-learn
 	pip install matplotlib
#
# 2.	Perform disease subtyping.
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.datasets import make_blobs
 	from sklearn.cluster import AgglomerativeClustering
 	from scipy.cluster.hierarchy import dendrogram, linkage
 	
 	# Generate synthetic gene expression data with three disease subtypes
 	X, y = make_blobs(n_samples=300, centers=3, random_state=42)
 	
 	# Apply Hierarchical Clustering
 	agg_clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0)
 	agg_clustering.fit(X)
 	
 	# Plot the Dendrogram
 	linked = linkage(X, method='ward')
 	dendrogram(linked, truncate_mode='lastp', p=12, show_contracted=True)
 	plt.xlabel("Sample Index")
 	plt.ylabel("Distance")
 	plt.title("Dendrogram")
 	plt.show()

# Principal Component Analysis 
# Following is a detailed coding example in Python:
# Let us implement PCA using Python and the scikit-learn library. We will use the famous Iris dataset for this example.
 	import numpy as np
 	import pandas as pd
 	import matplotlib.pyplot as plt
 	from sklearn.decomposition import PCA
 	from sklearn.datasets import load_iris
 	from sklearn.preprocessing import StandardScaler
 	
 	# Load the Iris dataset
 	iris = load_iris()
 	X = iris.data
	y = iris.target
 	
 	# Standardize the data
 	scaler = StandardScaler()
 	X_scaled = scaler.fit_transform(X)
 	
 	# Apply PCA
 	pca = PCA(n_components=2)
 	X_pca = pca.fit_transform(X_scaled)
 	
 	# Create a DataFrame for the transformed data
 	pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
 	pca_df['Target'] = y
 	
 	# Plot the transformed data
 	plt.figure(figsize=(8, 6))
 	targets = [0, 1, 2]
 	colors = ['r', 'g', 'b']
  	for target, color in zip(targets, colors):
 	    indices = pca_df['Target'] == target
 	    plt.scatter(pca_df.loc[indices, 'Principal Component 1'], pca_df.loc[indices, 'Principal Component 2'], c=color, s=50)
 	plt.xlabel('Principal Component 1')
 	plt.ylabel('Principal Component 2')
 	plt.title('PCA of Iris Dataset')
 	plt.legend(iris.target_names)
 	plt.show()

# Real world principal coding example
# In this example, we will use PCA to analyse a dataset from semiconductor manufacturing, where multiple process variables are measured during the fabrication of semiconductor wafers. We will use PCA to identify patterns and variations in the data and detect any anomalies in the manufacturing process.
# 1.	Install the required libraries Make sure you have the required libraries installed:
 	pip install numpy pandas matplotlib scikit-learn
#
#2.	Perform quality control using PCA.
 	import numpy as np
 	import pandas as pd
 	import matplotlib.pyplot as plt
 	from sklearn.decomposition import PCA
 	from sklearn.preprocessing import StandardScaler
 	
 	# Load the semiconductor manufacturing dataset
 	data = pd.read_csv("path_to_your_dataset.csv")
 	
 	# Separate the features (process variables) from the target (quality labels)
 	X = data.drop("Quality", axis=1)
 	y = data["Quality"]
 	
 	# Standardize the data
  	scaler = StandardScaler()
 	X_scaled = scaler.fit_transform(X)
 	
 	# Apply PCA
 	pca = PCA(n_components=2)
 	X_pca = pca.fit_transform(X_scaled)
 	
 	# Create a DataFrame for the transformed data
 	pca_df = pd.DataFrame(data=X_pca, columns=['Principal Component 1', 'Principal Component 2'])
 	pca_df['Quality'] = y
 	
  	# Plot the transformed data
  	plt.figure(figsize=(8, 6))
 	targets = pca_df['Quality'].unique()
 	colors = ['r', 'g', 'b']  # Assuming there are three quality labels
 	for target, color in zip(targets, colors):
 	    indices = pca_df['Quality'] == target
 	    plt.scatter(pca_df.loc[indices, 'Principal Component 1'], pca_df.loc[indices, 'Principal Component 2'], c=color, s=50)
 	plt.xlabel('Principal Component 1')
 	plt.ylabel('Principal Component 2')
 	plt.title('PCA for Quality Control in Semiconductor Manufacturing')
 	plt.legend(targets)
 	plt.show()

# t-Distributed Stochastic Neighbour Embedding
# Following is a detailed coding example in Python:
# Let us implement t-SNE using Python and the scikit-learn library. We will use the famous Iris dataset for this example.
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.datasets import load_iris
 	from sklearn.manifold import TSNE
 	
 	# Load the Iris dataset
 	iris = load_iris()
 	X = iris.data
 	y = iris.target
 	
 	# Apply t-SNE
 	tsne = TSNE(n_components=2, random_state=42)
 	X_tsne = tsne.fit_transform(X)
 	
 	# Plot the transformed data
 	plt.figure(figsize=(8, 6))
 	targets = [0, 1, 2]
 	colors = ['r', 'g', 'b']
  	for target, color in zip(targets, colors):
 	    indices = y == target
 	    plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], c=color, s=50)
 	plt.xlabel('t-SNE Component 1')
 	plt.ylabel('t-SNE Component 2')
 	plt.title('t-SNE Visualization of Iris Dataset')
  	plt.legend(iris.target_names)
 	plt.show()

# Real world coding example
# A real-world example of using t-SNE for drug discovery involves visualizing and analysing molecular structures and chemical properties of compounds. In this example, we will use t-SNE to explore a dataset of molecular fingerprints and identify clusters of similar compounds, which can aid in the discovery of potential drug candidates or understanding chemical relationships.
# 1.	Install the required libraries. Make sure you have the required libraries installed:
 	pip install numpy pandas matplotlib scikit-learn rdkit
#
# 2.	Perform Drug Discovery using t-SNE.
 	import numpy as np
 	import pandas as pd
 	import matplotlib.pyplot as plt
 	from sklearn.manifold import TSNE
 	from rdkit import Chem
 	from rdkit.Chem import AllChem
 	
 	# Load the molecular data (SMILES strings)
 	molecules = ['CCO', 'CCN', 'CCOCC', 'CN(C)C', 'CC(=O)O', 'CNC', 'CCS', 'CC(=O)NC']
 	mols = [Chem.MolFromSmiles(smiles) for smiles in molecules]
 	
 	# Calculate molecular fingerprints (MACCS keys)
 	fps = [AllChem.GetMACCSKeysFingerprint(mol) for mol in mols]
 	
 	# Convert fingerprints to a numpy array
 	fp_array = np.array([list(fp) for fp in fps])
 	
 	# Apply t-SNE
 	tsne = TSNE(n_components=2, random_state=42)
  	fp_tsne = tsne.fit_transform(fp_array)
 	
 	# Create a DataFrame for the transformed data
 	tsne_df = pd.DataFrame(data=fp_tsne, columns=['t-SNE Component 1', 't-SNE Component 2'])
 	tsne_df['Molecule'] = molecules
 	
 	# Plot the transformed data
 	plt.figure(figsize=(8, 6))
 	plt.scatter(tsne_df['t-SNE Component 1'], tsne_df['t-SNE Component 2'], s=50)
 	for i, txt in enumerate(tsne_df['Molecule']):
 	    plt.annotate(txt, (tsne_df['t-SNE Component 1'].iloc[i], tsne_df['t-SNE Component 2'].iloc[i]))
 	plt.xlabel('t-SNE Component 1')
 	plt.ylabel('t-SNE Component 2')
 	plt.title('t-SNE Visualization of Molecular Fingerprints for Drug Discovery')
 	plt.show()

# Association Rule Mining (A priori Algorithm)
# Here is a detailed coding example in Python:
# Let us implement the A Priori algorithm for ARM using Python and the ‘mlxtend’ library. We will use a small example dataset for demonstration:
# 1.	Install the required libraries: Make sure you have the required library installed:
 	pip install mlxtend
#
# 2.	Perform ARM using A Priori Algorithm.
 	import pandas as pd
 	from mlxtend.preprocessing import TransactionEncoder
 	from mlxtend.frequent_patterns import apriori, association_rules
 	
 	# Sample transaction data
 	transactions = [[‘milk’, ‘bread’, ‘eggs’],
 	                [‘milk’, ‘bread’],
 	                [‘milk’, ‘butter’],
 	                [‘bread’, ‘eggs’],
 	                [‘milk’, ‘bread’, ‘butter’, ‘eggs’]]
 	
 	# Encode the transaction data
 	te = TransactionEncoder()
 	te_ary = te.fit(transactions).transform(transactions)
 	df = pd.DataFrame(te_ary, columns=te.columns_)
 	
 	# Generate frequent itemsets with minimum support of 0.4
 	frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
 	
 	# Generate association rules with minimum confidence of 0.7
 	association_rules = association_rules(frequent_itemsets, metric=”confidence”, min_threshold=0.7)
 	
  	# Display frequent itemsets and association rules
 	print(“Frequent Itemsets:”)
 	print(frequent_itemsets)
 	print(“\nAssociation Rules:”)
 	print(association_rules)

# Real world coding example
# Let us imagine a hypothetical scenario where we have a dataset of reported incidents in a city. This dataset includes details about the type of crime, the location, the time of day, and other relevant factors. Our goal is to uncover patterns in these criminal activities.
# Here is a hypothetical dataset structure:
 	crime_data.csv:
 	----------------
 	incident_id, crime_type, location, time_of_day, weapon_used
 	1, robbery, downtown, night, gun
 	2, assault, suburbs, day, knife
 	3, robbery, downtown, night, gun
 	... and so on

# We will use the A priori algorithm to find associations among these attributes.
# Here is a Python code with mlxtend library:
 	import pandas as pd
 	from mlxtend.frequent_patterns import apriori
 	from mlxtend.frequent_patterns import association_rules
 	
 	# Load the dataset
 	data = pd.read_csv('crime_data.csv')
 	
 	# Convert the dataset into one-hot encoded format
 	one_hot = pd.get_dummies(data[['crime_type', 'location', 'time_of_day', 'weapon_used']])
 	
 	# Find frequent itemsets using Apriori
 	frequent_itemsets = apriori(one_hot, min_support=0.05, use_colnames=True)
 	
 	# Generate association rules
 	rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
 	
 	# Filter rules for better clarity
 	filtered_rules = rules[ (rules['lift'] >= 1.2) & (rules['confidence'] >= 0.7) ]
 	
 	print(filtered_rules)

# Exercises and solutions
# Using K-Means Clustering to segment mall customers based on spending habits
# Objective: Use the Mall Customers dataset to group customers into distinct segments based on their annual income and spending score, helping to target marketing campaigns more effectively.
# Dataset features:
# •	CustomerID: Unique ID assigned to the customer.
# •	Gender: Customer's gender.
# •	Age: Customer's age.
# •	Annual Income (k$): Customer's annual income in thousand dollars.
# •	Spending Score (1-100): Score assigned by the mall based on customer behaviour and spending nature.
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the Mall Customers dataset.
#   c.	Extract the 'Annual Income' and 'Spending Score' columns for clustering.
# 2.	Model building:
#   a.	Determine an optimal number of clusters (k) using the Elbow Method.
#   b.	Train a K-means clustering model using the optimal k.
#   c.	Visualize the formed clusters.
# 3.	Interpretation:
#   a.	Analyse and describe the characteristics of each customer segment.
#
# Code implementation:
 	# Import necessary libraries
 	import pandas as pd
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.cluster import KMeans
 	
 	# Load the Mall Customers dataset
 	df = pd.read_csv('Mall_Customers.csv')
 	
 	# Extract the relevant columns
 	X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
 	
 	# Determine optimal number of clusters using the Elbow Method
 	wcss = []  # Within-cluster sum of squares
 	for i in range(1, 11):
 	    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
 	    kmeans.fit(X)
 	    wcss.append(kmeans.inertia_)
 	
 	plt.plot(range(1, 11), wcss)
 	plt.title('The Elbow Method')
 	plt.xlabel('Number of clusters')
 	plt.ylabel('WCSS')
 	plt.show()
 	
 	# Train K-means model based on the Elbow point (for this exercise, assume it's 5)
 	kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
 	y_kmeans = kmeans.fit_predict(X)
 	
 	# Visualize the clusters
 	plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], label='Cluster 1')
 	# ... (repeat for other clusters)
 	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
 	plt.title('Clusters of customers')
  	plt.xlabel('Annual Income (k$)')
 	plt.ylabel('Spending Score (1-100)')
 	plt.legend()
 	plt.show()

# Task for the student:
# •	Describe the nature of each customer segment based on the formed clusters.
# •	Experiment with clustering using more than two features and observe if the segmentation changes.
# •	Use the Silhouette Score as another metric to determine the optimal number of clusters.

# Categorizing iris flowers based on features using hieracrchical clustering
# Objective: Use the Iris dataset, which contains measurements of iris flowers from three species, to cluster the flowers and understand the natural groupings.
# Dataset features:
# •	sepal length: Length of the sepal.
# •	sepal width: Width of the sepal.
# •	petal length: Length of the petal.
# •	petal width: Width of the petal.
# •	species: Species of the iris flower (Setosa, Versicolour, Virginica).
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the Iris dataset.
#   c.	Exclude the 'species' column for clustering, as it's a label.
# 2.	Dendrogram creation:
#   a.	Create a dendrogram to visualize the hierarchical structure and determine an optimal number of clusters.
# 3.	Model building:
#   a.	Use Agglomerative Clustering with the optimal number of clusters identified.
#   b.	Visualize the resulting clusters in a 2D space using a pair of features.
# 4.	Interpretation:
#   a.	Compare the formed clusters with the actual species column to evaluate how closely the hierarchical clustering managed to group the actual species.
#Code implementation:
 	# Import necessary libraries
 	import pandas as pd
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from scipy.cluster.hierarchy import dendrogram, linkage
 	from sklearn.cluster import AgglomerativeClustering
 	from sklearn.datasets import load_iris
 	
 	# Load the Iris dataset
 	iris = load_iris()
 	data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
 	data['species'] = iris.target
 	
  	# Extract the features for clustering
 	X = data.iloc[:, :-1].values
 	
 	# Create a dendrogram
 	Z = linkage(X, 'ward')  # Using Ward linkage method
 	dendrogram(Z, truncate_mode='lastp', p=10)  # Displaying only the last p merged clusters
 	plt.title('Dendrogram')
 	plt.xlabel('Iris Flowers')
 	plt.ylabel('Euclidean Distance')
 	plt.show()
 	
 	# Build Agglomerative Clustering model based on dendrogram observation (for this exercise, assume 3 clusters)
 	agg_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
 	clusters = agg_cluster.fit_predict(X)
 	
 	# Visualize the clusters using sepal length and sepal width
 	plt.scatter(X[clusters == 0, 0], X[clusters == 0, 1], label='Cluster 1')
 	# ... (repeat for other clusters)
 	plt.title('Clusters of Iris Flowers')
 	plt.xlabel('Sepal Length')
 	plt.ylabel('Sepal Width')
 	plt.legend()
 	plt.show()

# Task for the student:
# •	Describe the nature of each cluster by examining the feature values of its members.
# •	Experiment with different linkage methods (e.g., single, complete, average) and observe how the clustering changes.
# •	Compare the clusters obtained from hierarchical clustering with the actual species. How well did the algorithm perform in segregating different species?

# Dimensionality reduction for breast cancer data visualization using PCA
# Objective: Use the Breast Cancer dataset (available in scikit-learn) to visualize the data in a 2-dimensional space after reducing its dimensionality with PCA, allowing for better understanding and visualization of data clusters.
# Dataset features: The Breast Cancer dataset contains features computed from a digitized image of a Fine Needle Aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image.
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the Breast Cancer dataset.
#   c.	Standardize the dataset to have zero mean and unit variance.
# 2.	PCA implementation:
#   a.	Apply PCA to reduce the data dimensions to 2 principal components.
#   b.	Transform the data to the first two principal components.
# 3.	Visualization:
#   a.	Visualize the 2D projection of the data, color-coded based on the target variable (benign or malignant).
#
# Code implementation:
 	# Import necessary libraries
 	import pandas as pd
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.decomposition import PCA
 	from sklearn.datasets import load_breast_cancer
 	from sklearn.preprocessing import StandardScaler
 	
 	# Load the Breast Cancer dataset
 	data = load_breast_cancer()
 	df = pd.DataFrame(data.data, columns=data.feature_names)
 	target = data.target
 	
 	# Standardize the dataset
 	scaler = StandardScaler()
 	scaled_data = scaler.fit_transform(df)
 	
 	# Implement PCA
 	pca = PCA(n_components=2)
 	pca_result = pca.fit_transform(scaled_data)
 	
 	# Visualize the 2D projection
 	plt.figure(figsize=(10, 6))
  	plt.scatter(pca_result[:, 0], pca_result[:, 1], c=target, cmap='rainbow')
 	plt.xlabel('First Principal Component')
 	plt.ylabel('Second Principal Component')
 	plt.colorbar()
 	plt.title('2D PCA of Breast Cancer Dataset')
 	plt.show()

# Task for the student:
# •	Examine the variance explained by each of the two principal components. Which component explains more variance?
# •	Investigate the loadings of the original features on each of the principal components. Which features seem most important?
# •	Experiment with different numbers of components and observe the cumulative explained variance.
# •	After visualization, try to cluster the data in the reduced space using K-means clustering. Do the clusters match the original labels?

# Visualizing MNIST dataset in 2D using t-SNE
# Objective: Use the MNIST dataset (a large dataset of handwritten digits) to visualize the data in a 2-dimensional space using t-SNE, enabling a better understanding of the clusters and groupings of different digits.
# Dataset features: The MNIST dataset contains 28x28 pixel images of handwritten digits (0 through 9). Each image is represented as a 784-dimensional vector (28x28 pixels).
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load a subset of the MNIST dataset (using a subset due to the computational demand of t-SNE).
#   c.	Optionally, standardize the dataset.
# 2.	t-SNE implementation:
#   a.	Apply t-SNE to reduce the data dimensions to 2.
#   b.	Transform the data into this 2-dimensional t-SNE space.
# 3.	Visualization:
#   a.	Visualize the 2D t-SNE representation, color-coded based on the digit labels.
#
# Code implementation:
 	# Import necessary libraries
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.manifold import TSNE
 	from sklearn.datasets import fetch_openml
 	from sklearn.preprocessing import StandardScaler
 	
 	# Load a subset of the MNIST dataset
 	mnist = fetch_openml('mnist_784')
 	X = mnist.data[:5000]  # Using a subset for faster computation
 	y = mnist.target[:5000]
 	
 	# Optionally, standardize the dataset
  	scaler = StandardScaler()
 	X_scaled = scaler.fit_transform(X)
 	
 	# Implement t-SNE
 	tsne = TSNE(n_components=2, random_state=42)
  	X_tsne = tsne.fit_transform(X_scaled)
 	
 	# Visualize the 2D t-SNE representation
 	plt.figure(figsize=(10, 8))
 	scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='jet', alpha=0.5)
 	plt.colorbar(scatter)
 	plt.title('2D t-SNE of MNIST Dataset')
 	plt.show()

# Task for the student:
# •	Observe the 2D visualization. Can you identify distinct clusters for different digits?
# •	Experiment with different t-SNE parameters, such as perplexity and learning rate. How do they impact the visualization?
# •	Compare t-SNE visualization with PCA for the same dataset. Which one seems to separate clusters more distinctly?
# •	Implement a simple classifier (like k-NN) on the 2D t-SNE representation and check its performance. How well does it classify?

# Analysing retail transactions using Association Rule Mining
# Objective: Use a dataset of retail transactions to discover association rules which can help in identifying frequently co-purchased items, aiding in marketing and store arrangement decisions.
# Sample dataset features:
# •	The dataset should contain transactions where each transaction is a list of products purchased in a single shopping trip.
# •	Example: [['Milk', 'Bread', 'Eggs'], ['Milk', 'Diapers', 'Beer'], ...]
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the retail transactions dataset.
#   c.	Convert the transactions into a one-hot encoded Data Frame.
# 2.	Frequent itemset generation:
#   a.	Use the apriori method to find frequent item sets.
#   b.	Decide on a support threshold to filter out less common item sets.
# 3.	Association rule discovery:
#   a.	Generate association rules from the frequent item sets.
#   b.	Decide on metrics (e.g., confidence, lift) and their thresholds to select significant rules.
# Code implementation:
 	# Import necessary libraries
 	import pandas as pd
 	from mlxtend.preprocessing import TransactionEncoder
 	from mlxtend.frequent_patterns import apriori, association_rules
 	
 	# Sample transactions (Replace with actual data loading step if you have a dataset)
 	transactions = [['Milk', 'Bread', 'Eggs'],
 	                ['Milk', 'Diapers', 'Beer'],
 	                ['Milk', 'Bread', 'Diapers'],
 	                ['Bread', 'Eggs', 'Beer'],
 	                ['Milk', 'Bread', 'Diapers', 'Beer']]
 	
 	# Convert transactions to one-hot encoded DataFrame
 	encoder = TransactionEncoder()
 	onehot = encoder.fit_transform(transactions)
  	df = pd.DataFrame(onehot, columns=encoder.columns_)
 	
 	# Generate frequent itemsets
 	frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
 	
 	# Generate association rules
 	rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.2)
 	
 	# Display significant rules
 	print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Tasks for the student:
# •	Analyse the discovered association rules. Which items are most purchased together?
# •	Experiment with different support and confidence thresholds. How do they impact the discovered rules?
# •	Investigate the lift metric. Why might it be useful in this context?
# •	Extend the dataset or use a larger dataset and re-run the association rule mining. What new insights emerge?
# •	How might a retailer use these insights in practice, e.g., for store arrangement or promotions?