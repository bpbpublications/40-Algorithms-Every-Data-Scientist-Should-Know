# Classic examples of AI and ML
# Here are a few classic examples of projects in the fields of AI and ML, along with simplified source code snippets. Let us dive in:
#	Sentiment analysis: Sentiment analysis is the process of determining the sentiment or emotion behind a given text. Here is a simple example using Python and the Natural Language Toolkit (NLTK) library:
    from nltk.sentiment import SentimentIntensityAnalyzer
	
	def analyze_sentiment(text):
	    analyzer = SentimentIntensityAnalyzer()
	    sentiment_scores = analyzer.polarity_scores(text)
	    sentiment = sentiment_scores['compound']
	    
	    if sentiment >= 0.05:
	        return 'Positive'
	    elif sentiment <= -0.05:
            return 'Negative'
	    else:
	        return 'Neutral'
	
	# Example usage
	text = "I love this movie!"
	sentiment = analyze_sentiment(text)
	print(sentiment)

#	Image classification: Image classification involves training a model to classify images into predefined categories. Here is an example using Python, TensorFlow, and the Modified National Institute of Standards and Technology (MNIST) dataset:
	import tensorflow as tf
	from tensorflow.keras.datasets import mnist
	
	# Load the MNIST dataset
	(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
	
	# Preprocess the data
	train_images = train_images / 255.0
	test_images = test_images / 255.0
	
	# Build and train the model
	model = tf.keras.Sequential([
	    tf.keras.layers.Flatten(input_shape=(28, 28)),
	    tf.keras.layers.Dense(128, activation='relu'),
	    tf.keras.layers.Dense(10, activation='softmax')
	])
	
	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])
	
	model.fit(train_images, train_labels, epochs=5)
	
	# Evaluate the model
	test_loss, test_accuracy = model.evaluate(test_images, test_labels)
	print(f'Test accuracy: {test_accuracy}')

#	Spam detection: Spam detection aims to classify emails or messages as spam or not spam. Here is a basic example using Python and scikit-learn:
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.model_selection import train_test_split
	from sklearn.svm import SVC
	
	# Sample email dataset
	emails = [
	    ("Hello, this is a legitimate email."),
	    ("Get a free vacation now! Limited time offer!"),
	    ("Your bank account has been compromised. Please update your information."),
	    # ... more emails
	]
	labels = [0, 1, 1]  # 0 for legitimate, 1 for spam
	
	# Vectorize the emails
	vectorizer = CountVectorizer()
	features = vectorizer.fit_transform(emails)
	
	# Split the dataset
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
	
	# Train the classifier
	classifier = SVC()
	classifier.fit(X_train, y_train)
	
	# Predict labels for new emails
	new_emails = [
	    ("Hello, just wanted to say hi!"),
	    ("You've won a million dollars! Claim your prize now!")
	]
	new_features = vectorizer.transform(new_emails)
	predictions = classifier.predict(new_features)
	print(predictions)

# Please note that these code snippets are simplified for illustrative purposes, and real-world projects may require more complex code, data pre-processing, and additional steps such as hyperparameter tuning, model evaluation, and deployment considerations.
# Examples of AI/ML algorithms
# Please note that these code snippets are simplified for illustrative purposes, and real-world projects may require more complex code, data pre-processing, and additional steps such as hyperparameter tuning, model evaluation, and deployment considerations.
# Here are some code examples for popular AI and ML algorithms using Python:

# Decision tree classifier (Supervised learning):
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score

	# Load the Iris dataset
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	# Split the data into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Create a decision tree classifier
	clf = DecisionTreeClassifier()

	# Train the classifier
	clf.fit(X_train, y_train)

	# Make predictions on the test set
	y_pred = clf.predict(X_test)

	# Evaluate the accuracy of the model
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

# K-Means clustering (Unsupervised learning):
	from sklearn import datasets
	from sklearn.cluster import KMeans
	
	# Load the Iris dataset
	iris = datasets.load_iris()
	X = iris.data
	
	# Create a K-Means clustering model
	kmeans = KMeans(n_clusters=3)
	
	# Fit the model to the data
	kmeans.fit(X)
	
	# Get the cluster labels for each data point
	labels = kmeans.labels_
	
	# Print the cluster labels
	print("Cluster Labels:", labels)

# Q-learning (Reinforcement learning):
	import numpy as np
	
	# Create a simple grid world environment
	env = np.array([
	    [0, 0, 0, 0],
	    [0, 1, 0, -1],
	    [0, 0, 0, 0]
	])
	
	# Initialize the Q-table
	q_table = np.zeros((3, 4))
	
	# Set hyperparameters
	learning_rate = 0.1
	discount_factor = 0.9
	num_episodes = 1000
	
	# Q-learning algorithm
	for episode in range(num_episodes):
	    state = (2, 0)  # Starting state
	    done = False
	
	    while not done:
	        # Choose an action (e.g., up, down, left, right) based on the current state
	        action = np.argmax(q_table[state])
	
	        # Perform the action and observe the next state and reward
	        next_state = ...  # Code to get the next state based on the action
	        reward = ...  # Code to get the reward based on the next state
	
	        # Update the Q-table using the Q-learning update rule
	        q_table[state][action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state][action])
	
	        state = next_state
	
	        if env[state] == -1:  # Reached the goal state
	            done = True
	
	# Print the learned Q-table
	print("Q-table:")
	print(q_table)

# Deep Learning – CNN:
	import tensorflow as tf
	from tensorflow.keras import datasets, layers, models
	
	# Load the CIFAR-10 dataset
	(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
	
	# Normalize pixel values between 0 and 1
	train_images, test_images = train_images / 255.0, test_images / 255.0
	
	# Create a convolutional neural network model
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(10))
	
	# Compile the model
	model.compile(optimizer='adam',
	              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	              metrics=['accuracy'])
	
	# Train the model
	model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
	
	# Evaluate the model
	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	print("Test accuracy:", test_acc)

# Evolutionary algorithms - Genetic algorithm:
	import numpy as np
	
	# Define the fitness function
	def fitness_function(x):
	    return np.sum(x)
	
	# Define the genetic algorithm
	def genetic_algorithm(population_size, chromosome_length, generations):
	    population = np.random.randint(2, size=(population_size, chromosome_length))
	
	    for generation in range(generations):
	        fitness_values = np.array([fitness_function(chromosome) for chromosome in population])
	        best_chromosome = population[np.argmax(fitness_values)]
	        best_fitness = np.max(fitness_values)
	
	        # Perform selection, crossover, and mutation
	        # ...
	
	    return best_chromosome, best_fitness
	
	# Run the genetic algorithm
	population_size = 100
	chromosome_length = 10
	generations = 50
	
	best_chromosome, best_fitness = genetic_algorithm(population_size, chromosome_length, generations)
	print("Best Chromosome:", best_chromosome)
	print("Best Fitness:", best_fitness)

# Recommender system - Collaborative filtering (Matrix factorization):
	import numpy as np
	from scipy.sparse import lil_matrix
	from scipy.sparse.linalg import svds
	
	# Create a user-item matrix
	ratings = np.array([
	    [5, 3, 0, 1],
	    [4, 0, 0, 1],
	    [1, 1, 0, 5],
	    [1, 0, 0, 4],
	    [0, 1, 5, 4],
	])
	
	# Perform matrix factorization using Singular Value Decomposition (SVD)
	U, sigma, Vt = svds(ratings, k=2)  # k is the number of latent factors
	
	# Predict the ratings for missing entries in the user-item matrix
	predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt)
	
	# Example usage: Predict the rating for user 2 and item 3
	user_id = 2
	item_id = 3
	predicted_rating = predicted_ratings[user_id, item_id]
	print("Predicted Rating:", predicted_rating)