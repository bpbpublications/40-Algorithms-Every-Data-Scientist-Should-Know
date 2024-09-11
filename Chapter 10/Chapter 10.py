# Basic Semi-Supervised Learning Algorithms
# Self-training
# Let us use the self-training concept with a simple classifier on a toy dataset.
	import numpy as np
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	
	# Create a toy dataset
	X, y = datasets.make_classification(n_samples=1000, n_features=20, random_state=42)
	
	# Split the dataset: 80% unlabeled, 20% labeled
	X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, test_size=0.8, random_state=42)
	
	clf = RandomForestClassifier()
	
	# Initial training on the labeled data
	clf.fit(X_labeled, y_labeled)
	
	# Self-training iterations
	for iteration in range(5):
	    # Predict pseudo-labels for the unlabeled data
	    probs = clf.predict_proba(X_unlabeled)
	    predictions = clf.predict(X_unlabeled)
	
	    # Select instances where the model is confident: probability > 0.9
	    confident_data_mask = np.max(probs, axis=1) > 0.9
	    X_confident = X_unlabeled[confident_data_mask]
	    y_confident = predictions[confident_data_mask]
	
	    # Add the confident instances to the labeled dataset
	    X_labeled = np.concatenate((X_labeled, X_confident))
	    y_labeled = np.concatenate((y_labeled, y_confident))
	
	    # Remove the confident instances from the unlabeled dataset
	    X_unlabeled = X_unlabeled[~confident_data_mask]
	
	    # Retrain the model
	    clf.fit(X_labeled, y_labeled)
	
	    # Check the performance (optional)
	    print(f"Iteration {iteration + 1}, Accuracy: {accuracy_score(y_unlabeled, clf.predict(X_unlabeled))}")
	
# Real world coding example
# Active learning and self-training are both semi-supervised learning techniques that aim to use unlabeled data to improve model performance. While they have distinct methodologies, they can be combined in certain scenarios. Here is a conceptual blend of both approaches:
# Scenario
# Suppose we are building a text classifier to categorize user reviews into positive or negative sentiments. We have a small, labeled dataset and a much larger unlabeled dataset. We'll use active learning to query the most uncertain samples to be labeled by a human expert and use self-training to automatically label the most confident ones.
# Example steps:
# 1.	Train an initial model on the small, labeled dataset.
# 2.	Use this model to predict labels on the unlabeled dataset.
# 3.	Identify steps:
#           •	Most uncertain samples (for active learning) – these are the samples where the model predictions are close to 0.5 (assuming binary classification with sigmoid activation).
#           •	Most confident samples (for self-training) – samples where the model's predictions are close to 0 or 1.
# 4.	Add the confidently predicted samples to the training set with their pseudo-labels.
# 5.	Request human expert labels for the uncertain samples and add them to the training set.
# 6.	Retrain the model on the updated training set.
# 7.	Repeat the process until a stopping criterion is met, e.g., a certain performance level, or a maximum number of iterations.
# 
# This Python example will be a conceptual demonstration using the Scikit-learn library:
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.model_selection import train_test_split
	import numpy as np
	
	# Fetch data
	newsgroups = fetch_20newsgroups(subset='all', categories=['rec.autos', 'sci.med'])
	X, y = newsgroups.data, newsgroups.target
	
	# Split data into small labeled set and large unlabeled set
	X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, train_size=0.1, stratify=y)
	
	vectorizer = CountVectorizer()
	X_labeled_vec = vectorizer.fit_transform(X_labeled)
	X_unlabeled_vec = vectorizer.transform(X_unlabeled)
	
	# Initial model training
	clf = MultinomialNB()
	clf.fit(X_labeled_vec, y_labeled)
	
	for iteration in range(5):  # Just as an example, we run 5 iterations
	    probs = clf.predict_proba(X_unlabeled_vec)
	    
	    # Identify uncertain samples and confident samples
	    uncertain_samples = np.argsort(np.abs(probs[:, 1] - 0.5))[:10]  # Top 10 uncertain samples
	    confident_samples = np.where((probs[:, 1] > 0.95) | (probs[:, 1] < 0.05))[0]  # Very confident predictions
	    
	    # For this example, we're pseudo-labeling confident samples.
	    # In a real-world application, you'd get true labels for the uncertain samples from an expert.
	    X_labeled.extend([X_unlabeled[i] for i in confident_samples])
	    y_labeled.extend(clf.predict([X_unlabeled_vec[i] for i in confident_samples]))
	    
	    # Remove used samples from the unlabeled pool
	    X_unlabeled = [x for i, x in enumerate(X_unlabeled) if i not in confident_samples and i not in uncertain_samples]
	    X_unlabeled_vec = vectorizer.transform(X_unlabeled)
	    
	    # Retrain model
	    X_labeled_vec = vectorizer.transform(X_labeled)
	    clf.fit(X_labeled_vec, y_labeled)
	    
	    print(f"Iteration {iteration+1} done!")

# Co-training
# Let us illustrate co-training using a simple synthetic dataset and scikit-learn:
	import numpy as np
	from sklearn.datasets import make_moons
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import KNeighborsClassifier
	import matplotlib.pyplot as plt
	
	# Create a synthetic dataset: 2D points shaped as two intertwined moons
	X, y = make_moons(n_samples=300, noise=0.3, random_state=42)
	
	# Split the data: 20 labeled, 280 unlabeled
	X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X, y, train_size=20, stratify=y, random_state=42)
	
	# Split the features into two views (each view is one dimension in this case)
	X_labeled_view1 = X_labeled[:, 0].reshape(-1, 1)
	X_labeled_view2 = X_labeled[:, 1].reshape(-1, 1)
	
	X_unlabeled_view1 = X_unlabeled[:, 0].reshape(-1, 1)
	X_unlabeled_view2 = X_unlabeled[:, 1].reshape(-1, 1)
	
	# Initial training
	clf1 = KNeighborsClassifier(n_neighbors=3).fit(X_labeled_view1, y_labeled)
	clf2 = KNeighborsClassifier(n_neighbors=3).fit(X_labeled_view2, y_labeled)
	
	n_iterations = 50
	n_samples = 5  # number of samples to label per iteration
	
	for iteration in range(n_iterations):
	    # Use classifiers to predict labels for the unlabeled data
	    probs1 = clf1.predict_proba(X_unlabeled_view1)
	    probs2 = clf2.predict_proba(X_unlabeled_view2)
	
	    # Get the most confident predictions for each classifier
	    top_samples_clf1 = np.argsort(-probs1[:, 1])[:n_samples]
	    top_samples_clf2 = np.argsort(-probs2[:, 1])[:n_samples]
	
	    # Add these to the labeled set for the other classifier
	    X_labeled_view2 = np.vstack([X_labeled_view2, X_unlabeled_view2[top_samples_clf1]])
	    y_labeled = np.concatenate([y_labeled, clf1.predict(X_unlabeled_view1[top_samples_clf1])])
	
	    X_labeled_view1 = np.vstack([X_labeled_view1, X_unlabeled_view1[top_samples_clf2]])
	    y_labeled = np.concatenate([y_labeled, clf2.predict(X_unlabeled_view2[top_samples_clf2])])
	
	    # Remove labeled samples from the unlabeled pool
	    X_unlabeled = np.delete(X_unlabeled, np.concatenate([top_samples_clf1, top_samples_clf2]), axis=0)
	    X_unlabeled_view1 = np.delete(X_unlabeled_view1, np.concatenate([top_samples_clf1, top_samples_clf2]), axis=0)
	    X_unlabeled_view2 = np.delete(X_unlabeled_view2, np.concatenate([top_samples_clf1, top_samples_clf2]), axis=0)
	
	    # Retrain the classifiers
	    clf1.fit(X_labeled_view1, y_labeled)
	    clf2.fit(X_labeled_view2, y_labeled)
	    
	    print(f"Iteration {iteration+1} done!")
	
	# After co-training, the classifiers can be used for predictions or further evaluations.

# Real world coding example
# Remote sensing involves analysing images or data collected from satellites or aircraft to observe and study large areas of the earth's surface. Co-training can be particularly useful for remote sensing tasks, especially when labeled data is scarce.
# Here is a simplified example where we will use co-training for a binary classification problem in remote sensing: classifying an area as either urban or non-urban based on two types of features (or views):
# •	Spectral features: These are features derived from the spectrum of light reflected off surfaces. This might include information across multiple bands (e.g., infrared, red, green, blue).
# •	Texture features: These could be features that represent patterns, variability, and relationships of pixel values in an image, like entropy or contrast.
# For the sake of brevity, this example will be illustrative and may require adaptation to real-world datasets:
	import numpy as np
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.datasets import make_classification
	from cotrain import CoTrainingClassifier # Note: You might need to install or implement a co-training module
	
	# Simulated remote sensing data
	X, y = make_classification(n_samples=1000, n_features=6, random_state=42)
	
	# Splitting the features into two views (3 features for spectral and 3 for texture)
	X_spectral = X[:, :3]
	X_texture = X[:, 3:]
	
	# Let's assume that only a small portion of the data is labeled
	n_labeled = 50
	X_labeled_spectral = X_spectral[:n_labeled]
	y_labeled = y[:n_labeled]
	
	X_labeled_texture = X_texture[:n_labeled]
	X_unlabeled_spectral = X_spectral[n_labeled:]
	X_unlabeled_texture = X_texture[n_labeled:]
	
	# Setting up co-training with KNN as the base classifier for each view
	clf1 = KNeighborsClassifier()
	clf2 = KNeighborsClassifier()
	
	co_train_clf = CoTrainingClassifier(clf1, clf2)
	co_train_clf.fit(X_labeled_spectral, X_labeled_texture, y_labeled, 
	                 X_unlabeled_spectral, X_unlabeled_texture)
	
	# Predict on new data (for demonstration purposes, we're using the same data)
	y_pred = co_train_clf.predict(X_spectral, X_texture)
	
	print(y_pred)

# Multi-view learning
# In this coding example, we will use the sklearn library to generate synthetic data and perform co-training:
	import numpy as np
	from sklearn.datasets import make_multilabel_classification
	from sklearn.semi_supervised import SelfTrainingClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split
	
	# Create a synthetic dataset with 2 views
	X, y = make_multilabel_classification(n_samples=1000, n_features=40, n_classes=2, n_labels=1, random_state=42, length=10)
	
	# Splitting the features into two views
	X1 = X[:, :20]
	X2 = X[:, 20:]
	
	# Split data
	X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, random_state=42)
	
	# Assuming only 50 samples are labeled
	rng = np.random.default_rng(42)
	random_unlabeled_points = rng.choice(np.arange(y_train.shape[0]), size=950, replace=False)
	y_train[random_unlabeled_points] = -1
	
	# Co-training using SelfTraining with RandomForest
	rf = RandomForestClassifier()
	self_training_model = SelfTrainingClassifier(rf)
	
	# Train on the first view
	self_training_model.fit(X1_train, y_train)
	y_pred_view1 = self_training_model.predict(X1_test)
	
	# Train on the second view
	self_training_model.fit(X2_train, y_train)
	y_pred_view2 = self_training_model.predict(X2_test)
	
	# You can combine the predictions from both views in several ways, like voting, to get the final prediction.

# Real world coding example
# Remote sensing often involves analysing data from multiple sources, such as multi-spectral, hyperspectral, and LiDAR data. Here is a simple conceptual example to demonstrate how one might use multi-view learning for remote sensing, focusing on land cover classification using two views: multi-spectral data and elevation data.
# Scenario
# We want to classify different land cover types (e.g., forest, urban, water) using both multi-spectral satellite imagery and elevation data from a Digital Elevation Model (DEM).

# For this example, let's consider co-training, a popular multi-view semi-supervised algorithm. To do so, follow the below mentioned steps:
# 1.	Data preparation: Suppose we have a limited set of labeled data and a large set of unlabeled data whicha re describes as below:
# •	‘multi_spectral_data’: Multi-spectral images of shape (num_samples, height, width, num_bands).
# •	‘elevation_data’: Elevation data of the same spatial resolution as the multi-spectral data but with only one channel, i.e., shape (num_samples, height, width, 1).
# 2.	Feature extraction: We might want to extract features from raw data to represent them in a more compact and informative form. For example:
# •	For multi-spectral data: Principal Component Analysis (PCA).
# •	For elevation data: Statistical metrics like mean, variance, or texture features.
# 3.	Classifier initialization: Initialize two classifiers, one for each view.
	from sklearn.neighbors import KNeighborsClassifier
	from co_training import CoTrainingClassifier  # You might need a library that supports co-training
	
	clf1 = KNeighborsClassifier()
	clf2 = KNeighborsClassifier()
	
	co_train_clf = CoTrainingClassifier(clf1, clf2)

#	Co-training: Use the labeled data to start the co-training process, and let the classifiers iteratively label the most confident samples in the unlabeled pool.
	# Assume X1_labeled, X2_labeled are the labeled data for the two views and y_labeled are the corresponding labels
	# X1_unlabeled, X2_unlabeled are the unlabeled data for the two views
	
	co_train_clf.fit(X1_labeled, X2_labeled, y_labeled, X1_unlabeled, X2_unlabeled)

#	Prediction and evaluation:
	y_pred = co_train_clf.predict(X1_test, X2_test)  # X1_test and X2_test are test data for the two views

# You can then evaluate ‘y_pred’ using metrics suitable for classification tasks, e.g., accuracy, F1 score, etc.

# Expectation-Maximization
# Python Example: Using EM with GMM on 1D data:
# ’et's see an example with synthetic data.
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.mixture import GaussianMixture
	
	# Generate synthetic data: a mix of two Gaussian distributions
	np.random.seed(42)
	data_1 = np.random.normal(loc=5, scale=2, size=300)
	data_2 = np.random.normal(loc=15, scale=3, size=700)
	data = np.concatenate([data_1, data_2])
	
	# Plot the data
	plt.hist(data, bins=50, density=True, alpha=0.6, col’r’'b')
	plt.tit“e("Histogram of D”ta")
	plt.show()
	
	# Use EM with Gaussian Mixture Model
	gmm = GaussianMixture(n_components=2, max_iter=100, random_state=42)  #’We're trying to fit two Gaussians
	gmm.fit(data.reshape(-1, 1))
	
	# Retrieve Gaussian parameters
	means = gmm.means_
	covariances = gmm.covariances_
	weights = gmm.weights_
	
	pri“t("Mea”s:", means)
	pri“t("Covarianc”s:", covariances)
	pri“t("Weights (or proportion”):", weights)
	
	# Plot the fitted Gaussians
	x = np.linspace(data.min(), data.max(), 1000)
	for (mean, covariance, weight) in zip(means, covariances, weights):
	    plt.plot(x, weight * stats.norm.pdf(x, mean, np.sqrt(covariance)).ravel())
	
	plt.hist(data, bins=50, density=True, alpha=0.6, col’r’'b')
	plt.tit“e("Fitted Gaussians using”EM")
	plt.show()

# Real World Coding Example
# Medical image segmentation is one of the common applications of the EM algorithm. Let's consider MRI (Magnetic Resonance Imaging) scans, where we might want to segment the image into different tissue types, like white matter (WM), grey matter (GM), and cerebrospinal fluid (CSF).
# If the intensities of the pixels corresponding to each tissue type can be modelled as Gaussian distributions, we can use a Gaussian Mixture Model (GMM) combined with EM to segment the MRI scan.
# Here's a simple example using the ‘GaussianMixture’ class from the ‘scikit-learn’ library:
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.mixture import GaussianMixture
	import nibabel as nib   # for reading MRI data
	
	# Load an MRI scan
	img = nib.load('path_to_mri_scan.nii.gz')  # Provide the path to your MRI scan file
	data = img.get_fdata()
	data_flat = data.reshape(-1)
	
	# Reshape and normalize data
	X = data_flat.reshape(-1, 1)
	X = X / np.max(X)
	
	# Apply Gaussian Mixture Model with EM
	gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
	labels = gmm.predict(X)
	
	# Reshape the labels to the original image shape
	segmented = labels.reshape(data.shape)
	
	# Visualize the segmented MRI
	fig, axes = plt.subplots(1, 4, figsize=(15, 5))
	
	axes[0].imshow(data[:, :, data.shape[2] // 2], cmap="gray")
	axes[0].set_title("Original MRI")
	
	for i, label in enumerate(["CSF", "GM", "WM"]):
	    axes[i+1].imshow(segmented[:, :, data.shape[2] // 2] == i, cmap="gray")
	    axes[i+1].set_title(label)
	
	plt.tight_layout()
	plt.show()

# In this example, we:
	# Load the MRI data using ‘nibabel’, which is a Python library for working with neuroimaging data.
	# Flatten and normalize the MRI data to be between 0 and 1.
	# Apply the EM algorithm with a Gaussian Mixture Model for 3 components (CSF, GM, WM).
	# Segment the MRI image based on the most probable tissue type for each voxel (3D pixel).
    # Visualize the segmented results.
# This example provides a basic illustration, and there are many nuances and additional techniques (like spatial regularization) that can be applied for better results.

# Graph-based methods
# Let us take an example where we have some labeled and unlabeled points in two-dimensional space, and we want to predict the labels for the unlabeled points using Label Propagation.

    import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.semi_supervised import LabelPropagation
	from sklearn.model_selection import train_test_split
	
	# Generate toy data
	X, y = datasets.make_moons(n_samples=300, noise=0.1, random_state=0)
	
	# Split the data into labeled and unlabeled samples
	X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.9, random_state=0)
	
	# Create the graph-based model
	label_prop_model = LabelPropagation()
	
	# Fit the model on the labeled data
	label_prop_model.fit(X_labeled, y_labeled)
	
	# Predict the labels for the entire dataset
	y_predicted = label_prop_model.predict(X)
	
	# Visualize the labeled and unlabeled points
	plt.figure(figsize=(10, 6))
	plt.scatter(X_labeled[:, 0], X_labeled[:, 1], c=y_labeled, marker='o', label='Labeled', cmap='RdBu')
	plt.scatter(X_unlabeled[:, 0], X_unlabeled[:, 1], c=y_predicted[len(y_labeled):], marker='x', label='Unlabeled', cmap='RdBu')
	plt.title('Graph-based Semi-supervised Learning')
	plt.legend()
	plt.show()

# Real world coding example
# In this example, we'll use the Python library ‘NetworkX’ to implement a simple graph-based semi-supervised learning model for anomaly detection. We'll use synthetic data for illustration purposes.
# First, let us install the required packages:
	pip install networkx scikit-learn numpy matplotlib

# Now, let us create some synthetic data:
	import numpy as np
	import matplotlib.pyplot as plt
	
	# Create synthetic data (two clusters for normal behavior and one for anomalies)
	normal_data_1 = np.random.normal(loc=0.0, scale=0.5, size=(40, 2))
	normal_data_2 = np.random.normal(loc=2.0, scale=0.5, size=(40, 2))
	anomalous_data = np.random.normal(loc=5.0, scale=0.5, size=(20, 2))
	
	data = np.vstack([normal_data_1, normal_data_2, anomalous_data])
	labels = np.array([0]*80 + [-1]*20)  # Labeled normal as 0, anomalous as -1
	
	# Plot the data
	plt.scatter(normal_data_1[:, 0], normal_data_1[:, 1], label='Normal 1')
	plt.scatter(normal_data_2[:, 0], normal_data_2[:, 1], label='Normal 2')
	plt.scatter(anomalous_data[:, 0], anomalous_data[:, 1], label='Anomalous')
	plt.legend()
	plt.show()

# Next, build the similarity graph:
	import networkx as nx
	from sklearn.metrics.pairwise import euclidean_distances
	
	# Compute pairwise Euclidean distances
	dist_matrix = euclidean_distances(data)
	
	# Create a graph
	G = nx.Graph()
	for i in range(len(data)):
        for j in range(i+1, len(data)):
	        # Add edges between similar nodes
	        if dist_matrix[i, j] < 1.5:
	            G.add_edge(i, j, weight=1.0 / dist_matrix[i, j])

# Perform graph-based semi-supervised learning to label the nodes:
	def propagate_labels(G, labels):
	    new_labels = labels.copy()
	    for node in G.nodes():
	        if labels[node] == -1:  # If the node is anomalous
	            neighbor_labels = [labels[n] for n in G.neighbors(node)]
	            # Most common label among neighbors
	            if len(neighbor_labels) > 0:
	                new_labels[node] = max(set(neighbor_labels), key=neighbor_labels.count)
	    return new_labels
	
	new_labels = propagate_labels(G, labels)

# Let us evaluate our method:
	from sklearn.metrics import classification_report
	
	# For the sake of this example, we’ll assume that the true anomalous labels are 1
	true_labels = np.array([0]*80 + [1]*20)
	
	print(“Classification Report:”)
	print(classification_report(true_labels, new_labels))
