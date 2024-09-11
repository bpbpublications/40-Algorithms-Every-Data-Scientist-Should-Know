# Advanced Unsupervised Learning Algorithms

# Density-Based Spatial Clustering of Applications with Noise
# Here is the Python example using scikit-learn:
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.cluster import DBSCAN
 	from sklearn.datasets import make_moons
 	
 	# Generate sample data (2D points in shape of two crescent moons)
 	X, _ = make_moons(n_samples=1000, noise=0.05)
 	
 	# Apply DBSCAN
 	db = DBSCAN(eps=0.3, min_samples=5).fit(X)
 	
 	# Get labels (cluster assignments) for each point
 	labels = db.labels_
 	
 	# Plot the clustering results
 	plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
 	plt.title("DBSCAN Clustering")
 	plt.show()

# Real world coding example
# Let us use DBSCAN to analyze a hypothetical dataset of earthquake epicenters to cluster areas of seismic activity. This example will provide an outline of the process. If you are working with a real dataset, you would need to modify this as per your data. Here are the steps to be followed: 
#	Acquiring data: For this example, we are assuming that we have a dataset with columns "Latitude", "Longitude", and "Magnitude" representing the earthquake epicentres.
#	Using DBSCAN: Let us proceed to use DBSCAN on this dataset.
 	import numpy as np
 	import pandas as pd
 	from sklearn.cluster import DBSCAN
 	import matplotlib.pyplot as plt
 	
 	# Load your data (needs to be downloaded from the web via a simple search)
 	# df = pd.read_csv('earthquake_data.csv')
 	
 	# For demonstration, let's create a synthetic dataset
 	np.random.seed(0)
 	data1 = np.random.randn(100, 2) + [2, 2]
 	data2 = np.random.randn(100, 2) + [-2, -2]
 	
 	df = pd.DataFrame(np.vstack([data1, data2]), columns=['Longitude', 'Latitude'])
 	
 	# Extract features
 	X = df[['Longitude', 'Latitude']]
 	
 	# DBSCAN clustering
  	db = DBSCAN(eps=0.5, min_samples=10).fit(X)  # Adjust parameters as necessary
 	labels = db.labels_
 	
 	# Add cluster labels to our dataframe
 	df['Cluster'] = labels
 	
 	# Plotting
 	plt.figure(figsize=(10, 7))
 	plt.scatter(df['Longitude'], df['Latitude'], c=df['Cluster'], cmap='plasma', s=50)
 	plt.title('DBSCAN Clustering of Earthquake Epicenters')
 	plt.xlabel('Longitude')
 	plt.ylabel('Latitude')
 	plt.colorbar()
 	plt.show()

# Gaussian Mixture Models 
# Here is a coding example with Python's Scikit-learn:
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.mixture import GaussianMixture
 	from sklearn.datasets import make_blobs
 	
 	# Create synthetic data
 	X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.60, random_state=0)
 	X = X[:, ::-1]  # flip axes for better plotting
 	
 	# Fit a Gaussian Mixture Model
 	gmm = GaussianMixture(n_components=4, covariance_type='full').fit(X)
 	labels = gmm.predict(X)
 	
 	# Plot the data and the clustering
 	plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
 	probs = gmm.predict_proba(X)
 	
 	# Plot the centroids (means) of the GMM
 	plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='X', s=200, linewidth=2)
 	plt.title("Gaussian Mixture Model")
 	plt.xlabel("Feature 1")
 	plt.ylabel("Feature 2")
 	plt.show()

# Real world coding example
# Motion capture data usually consists of 3D coordinates of markers or joints over time. One common way to use GMMs for activity recognition with such data is to extract meaningful features from the data and then use GMMs to model the distribution of these features for different activities. When a new motion sequence comes in, you'd then determine the likelihood of it under each GMM to classify the activity.
# Here is a basic example to illustrate the concept:
# 1.	Data preparation: Suppose you have motion capture data for two activities - walking and jumping. Let us assume each activity has data from 100 sequences, and from each sequence, you have extracted a feature vector (e.g., velocities, angles, etc.).
# For simplicity, let us use 2D feature vectors, where:
# •	X_walking has shape (100, 2) representing 100 walking sequences.
# •	X_jumping has shape (100, 2) representing 100 jumping sequences.
#
 	import numpy as np
 	from sklearn.mixture import GaussianMixture
 	
 	# Simulated data (you'd replace this with real motion capture features)
 	X_walking = np.random.randn(100, 2)
 	X_jumping = np.random.randn(100, 2) + 2

# 2.	Model training: Train a GMM for each activity:
 	gmm_walking = GaussianMixture(n_components=1).fit(X_walking)
 	gmm_jumping = GaussianMixture(n_components=1).fit(X_jumping)

# 3.	Activity recognition: When a new motion sequence comes in, and you extract its feature vector ‘x_new’, you can determine the activity by comparing the likelihoods under each GMM:
 	x_new = np.array([[1.5, 1.5]])  # A new feature vector
 	
 	log_prob_walking = gmm_walking.score_samples(x_new)
 	log_prob_jumping = gmm_jumping.score_samples(x_new)
 	
 	if log_prob_walking > log_prob_jumping:
 	    print("Activity: Walking")
 	else:
 	    print("Activity: Jumping")

# Autoencoders
# Simple Autoencoder with TensorFlow and Keras
# Here is the code snippet for autoencoder with TensorFlow and Keras:
 	import numpy as np
 	import tensorflow as tf
 	from tensorflow.keras.layers import Input, Dense
 	from tensorflow.keras.models import Model
 	from tensorflow.keras.datasets import mnist
 	import matplotlib.pyplot as plt
 	
 	# Load MNIST dataset
 	(x_train, _), (x_test, _) = mnist.load_data()
 	
 	# Normalize all values between 0 and 1
 	x_train = x_train.astype('float32') / 255.
 	x_test = x_test.astype('float32') / 255.
 	
 	# Flatten images
 	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
 	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
 	
 	# Size of encoded representations
 	encoding_dim = 32  
 	
 	# Input placeholder
 	input_img = Input(shape=(784,))
 	
 	# Encoded representation of the input
 	encoded = Dense(encoding_dim, activation='relu')(input_img)
 	
 	# Reconstruction of the input
 	decoded = Dense(784, activation='sigmoid')(encoded)
 	
 	# This model maps an input to its reconstruction
 	autoencoder = Model(input_img, decoded)
 	
 	# Separate encoder model
 	encoder = Model(input_img, encoded)
  	
 	# Training
 	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
 	autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
 	
 	# Testing
 	encoded_imgs = encoder.predict(x_test)
 	decoded_imgs = autoencoder.predict(x_test)
 	
 	# Visualize
 	n = 10
 	plt.figure(figsize=(20, 4))
 	for i in range(n):
 	    # Original
 	    ax = plt.subplot(2, n, i + 1)
 	    plt.imshow(x_test[i].reshape(28, 28))
 	    plt.gray()
 	    ax.get_xaxis().set_visible(False)
 	    ax.get_yaxis().set_visible(False)
 	
 	    # Reconstruction
 	    ax = plt.subplot(2, n, i + 1 + n)
 	    plt.imshow(decoded_imgs[i].reshape(28, 28))
 	    plt.gray()
 	    ax.get_xaxis().set_visible(False)
 	    ax.get_yaxis().set_visible(False)
 	plt.show()

# Real world coding example
# Autoencoders can be used in a variety of applications in the realm of art, including style transfer, image denoising, and image generation. Here is a simple example of using an autoencoder to generate a kind of abstract version of an image:
# Using Autoencoders for image abstraction
# The prerequisites needed are: 
# •	Python
# •	TensorFlow and Keras
# •	Matplotlib for visualization
#
# We will use a simple convolutional autoencoder, which consists of an encoder and a decoder.
  	import tensorflow as tf
 	from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
 	from tensorflow.keras.models import Model
 	
 	input_img = Input(shape=(128, 128, 3))  # Input image size
 	
 	# Encoding
 	x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
 	x = MaxPooling2D((2, 2), padding='same')(x)
 	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
 	encoded = MaxPooling2D((2, 2), padding='same')(x)
 	
 	# Decoding
  	x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
 	x = UpSampling2D((2, 2))(x)
  	x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
 	x = UpSampling2D((2, 2))(x)
 	decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
 	
 	autoencoder = Model(input_img, decoded)
 	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Let us use an image as our dataset. For simplicity, we'll use a single image, but ideally, you'd train on multiple images to obtain a generalized model.
 	from tensorflow.keras.preprocessing.image import load_img, img_to_array
 	
 	# Load the image
 	img = load_img('path_to_image.jpg', target_size=(128, 128))
 	img_array = img_to_array(img)
 	img_array = img_array.astype('float32') / 255.
 	
 	# Since we're using a single image, we need to expand the dimensions
  	img_array = tf.expand_dims(img_array, 0)

# For training the autoencoder, we will train the autoencoder on the same image to abstract itself. In practice, you would use a proper dataset.
 	autoencoder.fit(img_array, img_array, epochs=5000, shuffle=True)

# Let us now visualize the results:
 	import matplotlib.pyplot as plt
 	
 	decoded_img = autoencoder.predict(img_array)
 	
 	plt.figure(figsize=(10, 4))
 	
 	# Original Image
 	plt.subplot(1, 2, 1)
 	plt.imshow(img_array[0])
 	plt.title('Original Image')
 	
 	# Abstracted Image
 	plt.subplot(1, 2, 2)
 	plt.imshow(decoded_img[0])
 	plt.title('Abstracted Image')
 	
 	plt.show()

# Anomaly detection (Outlier detection)
# Coding example using Scikit-learn: Isolation forest
# One popular anomaly detection algorithm is the Isolation Forest. It is an ensemble method that isolates anomalies instead of profiling and constructing normal points. It works on the principle that anomalies are data points that are few and different, and they are easier to isolate compared to regular points.
# Here is a basic example of how to use the Isolation Forest algorithm from Scikit-learn:
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.ensemble import IsolationForest
 	
 	# Generate sample data
 	rng = np.random.RandomState(42)
 	
 	# Generating normal (X_train) and anomalies (X_outliers)
 	X_train = 0.3 * rng.randn(100, 2)
 	X_train = np.r_[X_train + 2, X_train - 2]
 	X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
 	
 	# Fit the model
 	clf = IsolationForest(max_samples=100, random_state=rng)
 	clf.fit(X_train)
 	
 	# Predictions
 	y_pred_train = clf.predict(X_train)
 	y_pred_outliers = clf.predict(X_outliers)
 	
 	# Plot the results
 	xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
  	Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
 	Z = Z.reshape(xx.shape)
 	
 	plt.title("IsolationForest")
 	plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
 	
 	b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white', s=20, edgecolor='k')
 	b2 = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red', s=20, edgecolor='k')
 	plt.legend([b1, b2], ["normal data", "anomalies"], loc="upper left")
 	plt.show()

# Real world coding example
# Let us discuss a real-world example where we utilize anomaly detection for pollution monitoring. Imagine we have a dataset with hourly recordings of particulate matter (PM2.5) levels, which are tiny particles in the air that reduce visibility and can harm the human lungs.
# In this example we will, detect unusual spikes in PM2.5 levels, which might indicate abnormal pollution events, using the Isolation Forest anomaly detection method.
# The set up is as follows:
# 1.	Acquire data: For simplicity, let us assume a dataset with two columns: ‘timestamp’ (hourly recordings) and ‘pm2_5_level’.
# 2.	Pre-process data: Check for missing values, and possibly interpolate or remove them.
# 3.	Implement Anomaly detection.
#
# The Python implementation is as follows:
 	import pandas as pd
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.ensemble import IsolationForest
 	
 	# Load data
 	df = pd.read_csv('pm25_data.csv')
 	
 	# Simple preprocessing: fill missing values using interpolation
 	df['pm2_5_level'].interpolate(inplace=True)
 	
 	# Apply Isolation Forest
 	iso_forest = IsolationForest(contamination=0.05)  # Assuming ~5% of the data might be anomalies
 	df['anomaly'] = iso_forest.fit_predict(df[['pm2_5_level']])
 	
 	# Plotting
  	plt.figure(figsize=(14, 7))
 	plt.plot(df['timestamp'], df['pm2_5_level'], label='PM2.5 Level', color='blue')
 	plt.scatter(df[df['anomaly'] == -1]['timestamp'], df[df['anomaly'] == -1]['pm2_5_level'], color='red', label='Anomaly')
 	plt.legend()
 	plt.title("PM2.5 Levels and Detected Anomalies")
 	plt.xlabel("Timestamp")
 	plt.ylabel("PM2.5 Level")
 	plt.xticks(rotation=45)
 	plt.tight_layout()
 	plt.show()

# Latent Dirichlet Allocation 
# Here is a basic Python example using the Gensim library, one of the most popular libraries for topic modelling:
 	import gensim
 	from gensim import corpora
 	from pprint import pprint
 	
 	# Sample data
 	documents = ["Human machine interface for lab abc computer applications",
 	             "A survey of user opinion of computer system response time",
 	             "The EPS user interface management system",
 	             "System and human system engineering testing of EPS",
 	             "Relation of user-perceived response time to error measurement"]
 	
 	# Pre-process the data
 	texts = [[word for word in document.lower().split()] for document in documents]
 	
 	# Create a dictionary from the data
 	dictionary = corpora.Dictionary(texts)
 	
 	# Convert document (a list of words) into the bag-of-words format = list of (token_id, token_count) 2-tuples.
 	corpus = [dictionary.doc2bow(text) for text in texts]
 	
 	# LDA model
 	lda_model = gensim.models.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)
 	
 	# Print topics
 	topics = lda_model.print_topics(num_words=4)
 	for topic in topics:
 	    print(topic)

# Real world coding example
# Let us consider a simple example where we aim to analyse trends in tech news articles over several years. The goal is to see how major technological trends have evolved.
# Imagine we have a dataset containing tech news articles from 2010 to 2023. Each article has a timestamp and content. For simplicity, let us consider this dataset as a list of dictionaries:
 	data = [
 	    {'date': '2010-01-15', 'content': '...'},
 	    {'date': '2010-02-12', 'content': '...'},
 	    ...
 	    {'date': '2023-06-22', 'content': '...'},
 	]
# Here is the approach to be followed:
# 1.	Pre-process the data (tokenization, stop word removal, stemming).
# 2.	Extract articles year-wise.
# 3.	For each year, run LDA to determine the major topics.
# 4.	Visualize the most significant topics for each year to observe the trend.
#
#Code implementation:
 	import gensim
 	from gensim import corpora
 	from nltk.corpus import stopwords
 	from nltk.stem import PorterStemmer
 	from nltk.tokenize import word_tokenize
 	import matplotlib.pyplot as plt
 	import pandas as pd
 	
 	# Step 1: Preprocess the data
 	stop_words = set(stopwords.words('english'))
  	ps = PorterStemmer()
 	
 	def preprocess(document):
 	    tokens = word_tokenize(document)
 	    tokens = [word for word in tokens if word not in stop_words]
 	    tokens = [ps.stem(word) for word in tokens]
 	    return tokens
 	
 	# Convert to DataFrame for easier manipulation
 	df = pd.DataFrame(data)
 	df['date'] = pd.to_datetime(df['date'])
 	df['year'] = df['date'].dt.year
 	df['tokens'] = df['content'].apply(preprocess)
 	
 	# Step 2: Extract articles year-wise
 	years = sorted(df['year'].unique())
 	
 	# Step 3: Run LDA for each year and gather top topics
 	top_topics_per_year = {}
 	
 	for year in years:
 	    year_data = df[df['year'] == year]
 	    dictionary = corpora.Dictionary(year_data['tokens'])
 	    corpus = [dictionary.doc2bow(text) for text in year_data['tokens']]
 	    lda = gensim.models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
 	    top_topics = lda.print_topics(num_words=5)
 	    top_topics_per_year[year] = top_topics
 	
 	# Step 4: Visualization (printing here for simplicity)
 	for year, topics in top_topics_per_year.items():
 	    print(f"Year: {year}")
  	    for topic in topics:
 	        print(topic)
 	    print("\n")

# Exercises and solutions
# DBSCAN Exercise: Clustering geographical data
# Objective: Use a dataset of geographical coordinates (latitude and longitude) to identify dense clusters of locations, helping in understanding areas of high activity or interest.
# Sample dataset features: The dataset should contain latitude and longitude of various locations.
# Example: [['40.7128', '-74.0060'], ['34.0522', '-118.2437'], ...] (Coordinates for New York City and Los Angeles respectively)
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the geographical dataset.
#   c.	Plot the coordinates to visually inspect the distribution.
# 2.	DBSCAN implementation
#   a.	Apply DBSCAN to cluster the geographical data.
#   b.	Experiment with the eps (maximum distance between two samples for one to be considered as in the neighbourhood of the other) and min_samples parameters to get meaningful clusters.
# 3.	Visualization
#   a.	Visualize the clustered data using different colours for each cluster and a distinct marker for noise points.
#   b.	Interpret the clusters in terms of potential areas of interest or activity.
#
# Code implementation:
 	# Import necessary libraries
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.cluster import DBSCAN
 	
 	# Sample geographical data (Replace with actual data loading step if you have a dataset)
 	data = np.array([[40.7128, -74.0060], 
 	                 [34.0522, -118.2437], 
 	                 # ... (more data)
 	                ])
 	
 	# Plot raw data
 	plt.scatter(data[:, 0], data[:, 1], s=10)
 	plt.title('Raw Geographical Data')
 	plt.xlabel('Latitude')
 	plt.ylabel('Longitude')
 	plt.show()
 	
 	# Implement DBSCAN
 	dbscan = DBSCAN(eps=0.5, min_samples=5)
 	clusters = dbscan.fit_predict(data)
 	
 	# Visualize the clustered data
 	plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', s=10)
 	plt.title('DBSCAN Clustering of Geographical Data')
 	plt.xlabel('Latitude')
 	plt.ylabel('Longitude')
 	plt.show()

# Task for the student:
# •	Interpret the clusters: Can you identify regions of high activity?
# •	Adjust the eps and min_samples parameters of DBSCAN. How do they influence the clustering result?
# •	Compare DBSCAN results with another clustering algorithm (e.g., K-means). What differences do you observe?
# •	If you have additional information about each location (e.g., type of place), can you further interpret the nature of the clusters?

# GMM Exercise: Clustering Customer Spending Data
# Objective: Use a dataset of customer spending behaviours across different product categories to segment customers into different groups, potentially guiding marketing, or sales strategies.
# Sample dataset features
# •	The dataset should contain columns representing amounts spent by customers on various product categories.
# •	Example: [['Electronics', 'Apparel', 'Groceries'], [150, 60, 200], [80, 20, 300], ...]
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the customer spending dataset.
#   c.	Standardize the data to have zero mean and unit variance.
# 2.	GMM Implementation
#   a.	Apply GMM to cluster the customer data.
#   b.	Experiment with the number of components to find the optimal number of customer segments.
# 3.	Analysis
#   a.	Analyse the means and covariances of the GMM components to understand the spending behaviours of each segment.
#   b.	Visualize the customer segments using dimensionality reduction techniques (like PCA) if the dataset has more than two features.
#
# Code implementation:
 	# Import necessary libraries
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from sklearn.mixture import GaussianMixture
 	from sklearn.preprocessing import StandardScaler
 	from sklearn.decomposition import PCA
 	
 	# Sample spending data (Replace with actual data loading step if you have a dataset)
 	data = np.array([[150, 60, 200], 
 	                 [80, 20, 300], 
 	                 # ... (more data)
 	                ])
 	
 	# Standardize the data
 	scaler = StandardScaler()
 	scaled_data = scaler.fit_transform(data)
 	
 	# Apply GMM
 	gmm = GaussianMixture(n_components=3)
 	clusters = gmm.fit_predict(scaled_data)
 	
 	# PCA for visualization
 	pca = PCA(n_components=2)
 	reduced_data = pca.fit_transform(scaled_data)
 	
 	# Visualize the clustered data
 	plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', s=10)
 	plt.title('Customer Segments via GMM')
 	plt.xlabel('Principal Component 1')
 	plt.ylabel('Principal Component 2')
 	plt.show()

# Task for the student:
# •	Interpret the customer segments: What characterizes the spending behaviour of each cluster?
# •	Adjust the number of GMM components. How does it impact the segmentation?
# •	Compare GMM results with another clustering algorithm (e.g., K-means). What differences or similarities do you observe?
# •	If you were a marketing executive, how might you tailor different marketing campaigns for each segment?

# Autoencoder exercise: Image denoising
# Objective: Use an autoencoder to denoise images. The autoencoder will learn to encode the clean version of an image and will be tested on its ability to reconstruct a clean image from a noisy one.
# Dataset: Ideally, use a dataset like MNIST or CIFAR-10, where images are simple and widely recognized.
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the image dataset.
#   c.	Normalize the pixel values (e.g., between 0 and 1).
#   d.	Add noise to the images to create a noisy dataset. Use Gaussian noise or salt-and-pepper noise.
# 2.	Build the Autoencoder
#   a.	Construct an encoder network that compresses the input image.
#   b.	Construct a decoder network that reconstructs the original image from its encoded representation.
#   c.	Combine the encoder and decoder to form the autoencoder.
# 3.	Training
#   a.	Use the clean images as both the input and target for the autoencoder.
#   b.	Train the autoencoder to minimize the difference between the input and the reconstructed image.
# 4.	Evaluation
#   a.	After training, feed the noisy images to the autoencoder.
#   b.	Visualize the noisy input, the autoencoder's reconstruction, and the original clean image side-by-side.
#
# Code implementation:
 	import numpy as np
 	import matplotlib.pyplot as plt
 	from keras.datasets import mnist
 	from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
 	from keras.models import Model
 	
 	# Load and preprocess data
 	(x_train, _), (x_test, _) = mnist.load_data()
 	x_train = x_train.astype('float32') / 255.
 	x_test = x_test.astype('float32') / 255.
 	x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
 	x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
 	
 	# Introduce noise
 	noise_factor = 0.5
 	x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
 	x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
 	
 	# Define the autoencoder (this example uses a simple convolutional structure)
 	input_img = Input(shape=(28, 28, 1))
 	x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
 	x = MaxPooling2D((2, 2), padding='same')(x)
 	encoded = x
 	
 	x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
 	x = UpSampling2D((2, 2))(x)
 	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
 	
 	autoencoder = Model(input_img, decoded)
 	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
 	
 	# Train
 	autoencoder.fit(x_train_noisy, x_train, epochs=50, batch_size=128, validation_data=(x_test_noisy, x_test))
  	
 	# Evaluate and visualize
 	decoded_imgs = autoencoder.predict(x_test_noisy)
 	
 	# Plotting code here to visualize original, noisy, and reconstructed images...

# Task for the Student:
# •	Analyse how different types of noise (amount and nature) affect the autoencoder's performance.
# •	Explore deeper or more complex autoencoder architectures. How do they impact the reconstruction quality?
# •	Try using autoencoders for other tasks, such as anomaly detection. How might you approach this?
#
# Anomaly detection exercise: Detecting fraudulent transactions
# Objective: Use a dataset of financial transactions to identify potentially fraudulent ones by detecting outliers or anomalies in the data.
# Sample dataset features:
# •	The dataset should contain columns representing attributes of financial transactions, like Amount, Timestamp, MerchantID, and Location.
# •	Most of the transactions are legitimate, with a small fraction being fraudulent.
#
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the transactions dataset.
#   c.	Process timestamp to extract useful features like HourOfDay.
#   d.	Standardize or normalize the numerical features.
# 2.	Anomaly detection with isolation forest
#   a.	Isolation Forest is an effective algorithm for anomaly detection. Train an Isolation Forest model on your data.
#   b.	Predict anomalies using the trained model. This will assign a score to each transaction indicating its likelihood of being an anomaly.
# 3.	Evaluation
#   a.	Visualize the distribution of anomaly scores. Potentially fraudulent transactions should ideally have higher scores.
#   b.	If the dataset contains labelled data, evaluate the model's effectiveness using appropriate metrics (e.g., F1 score, precision-recall curve).
#
# Code implementation:
 	# Import necessary libraries
 	import numpy as np
 	import pandas as pd
 	import matplotlib.pyplot as plt
 	from sklearn.ensemble import IsolationForest
 	from sklearn.preprocessing import StandardScaler
 	from sklearn.metrics import classification_report
 	
 	# Load data
 	# data = pd.read_csv('path_to_dataset.csv')
 	
 	# Sample data for illustration (replace with actual loading code)
 	data = pd.DataFrame({
 	    'Amount': np.random.rand(1000),
 	    'Timestamp': pd.date_range(start='2022-01-01', periods=1000, freq='H'),
  	    'MerchantID': np.random.choice(100, 1000),
 	    'Location': np.random.choice(['LocationA', 'LocationB', 'LocationC'], 1000)
 	})
 	
 	# Process features
 	data['HourOfDay'] = data['Timestamp'].dt.hour
 	data = data.drop(columns=['Timestamp'])
 	
 	# Convert categorical features using one-hot encoding
 	data = pd.get_dummies(data, drop_first=True)
 	
 	# Standardize data
 	scaler = StandardScaler()
 	scaled_data = scaler.fit_transform(data)
 	
 	# Apply Isolation Forest
 	iso_forest = IsolationForest(contamination=0.05) # assume 5% of the data are anomalies
 	anomaly_scores = iso_forest.fit_predict(scaled_data)
 	
 	# Visualize anomaly scores
 	plt.hist(anomaly_scores)
 	plt.title('Distribution of Anomaly Scores')
  	plt.xlabel('Score')
 	plt.ylabel('Number of Transactions')
 	plt.show()
 	
 	# If you have labels indicating fraudulent transactions:
 	# true_labels = data['IsFraud'].values
 	# print(classification_report(true_labels, anomaly_scores))

# Task for the student
# •	Adjust parameters of the Isolation Forest, such as contamination or n_estimators. How do they influence the detection results?
# •	Experiment with other anomaly detection algorithms available in scikit-learn (e.g., One-Class SVM, Local Outlier Factor). Compare their performances.
# •	How can the results from the anomaly detection process be used in real-life scenarios to potentially halt or investigate suspicious transactions?
 
# LDA exercise: Topic modelling on news articles
# Objective: Use a dataset of news articles to identify the main topics they cover using the LDA algorithm.
# Sample dataset features
# •	The dataset should contain news articles as text data. The dataset could be from online sources or news websites.
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the news articles dataset.
#   c.	Pre-process the text data: tokenize, remove stop words, and stem or lemmatize the words.
# 2.	LDA implementation
#   a.	Convert the tokenized documents into a bag-of-words or TF-IDF representation.
#   b.	Use the LDA model from the gensim library to identify topics in the corpus.
# 3.	Analysis
#   a.	Display the words associated with each topic.
#   b.	For a few sample articles, display the topic distribution to see which topics are most prominent.
# Code skeleton:
 	import gensim
 	from gensim import corpora
 	from nltk.tokenize import word_tokenize
 	from nltk.corpus import stopwords
 	from nltk.stem import WordNetLemmatizer
 	
 	# Load data
 	# articles = pd.read_csv('path_to_dataset.csv')['article_text'].tolist()
 	
	# Sample data (replace with actual data loading code)
	articles = ["This is a sample news about politics.", "This article discusses the latest in technology.", "Sports news are covered in this article."]
	
	# Pre-process articles
 	lemmatizer = WordNetLemmatizer()
  	stop_words = set(stopwords.words('english'))
 	
 	processed_articles = []
 	for article in articles:
 	    tokens = word_tokenize(article)
 	    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
  	    processed_articles.append(tokens)
 	
 	# Create dictionary and corpus for LDA
 	dictionary = corpora.Dictionary(processed_articles)
 	corpus = [dictionary.doc2bow(article) for article in processed_articles]
 	
 	# Apply LDA
 	num_topics = 3
 	lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)
 	
 	# Display topics
 	for idx, topic in lda_model.print_topics(-1):
 	    print(f"Topic: {idx} \nWords: {topic}\n")
 	
 	# For a sample article, display topic distribution
 	print(lda_model[corpus[0]])

# Task for the students
# •	Adjust the number of topics (num_topics). How does the quality and interpretability of the topics change?
# •	Analyse a larger set of articles and try to interpret the topics generated by the LDA. Do they make intuitive sense?
# •	How might these topics be useful for organizing or categorizing news articles on a news website?
# •	Experiment with different pre-processing steps or hyperparameters in the LDA model to see if they lead to more coherent topics.
