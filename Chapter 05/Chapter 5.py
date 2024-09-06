# Advanced Supervised Learning Algorithms

# Naive Bayes
# Here is a coding example in Python using Scikit-learn library for Gaussian Naive Bayes classification:
	from sklearn.datasets import load_iris
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	
	# Load the Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create an instance of the Gaussian Naive Bayes Classifier
	model = GaussianNB()
	
	# Train the model
	model.fit(X_train, y_train)
	
	# Make predictions on the testing set
	y_pred = model.predict(X_test)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

# Real-world coding example
# In this example, we have a dataset containing weather conditions with four features: Outlook, Temperature, Humidity, and Wind. The corresponding target labels indicate whether playing outside (Yes) is suitable based on those weather conditions.
	from sklearn.naive_bayes import GaussianNB
	from sklearn.metrics import accuracy_score
	
	# Weather features: Outlook, Temperature, Humidity, Wind
	X = [
	    ['Sunny', 'Hot', 'High', 'Weak'],
	    ['Sunny', 'Hot', 'High', 'Strong'],
	    ['Overcast', 'Hot', 'High', 'Weak'],
	    ['Rainy', 'Mild', 'High', 'Weak'],
	    ['Rainy', 'Cool', 'Normal', 'Weak'],
	    ['Rainy', 'Cool', 'Normal', 'Strong'],
	    ['Overcast', 'Cool', 'Normal', 'Strong'],
	    ['Sunny', 'Mild', 'High', 'Weak'],
	    ['Sunny', 'Cool', 'Normal', 'Weak'],
	    ['Rainy', 'Mild', 'Normal', 'Weak'],
	    ['Sunny', 'Mild', 'Normal', 'Strong'],
	    ['Overcast', 'Mild', 'High', 'Strong'],
	    ['Overcast', 'Hot', 'Normal', 'Weak'],
	    ['Rainy', 'Mild', 'High', 'Strong']
	]
	
	# Corresponding target labels: Play (Yes/No)
	y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']
	
	# Create an instance of the Gaussian Naive Bayes Classifier
	model = GaussianNB()
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions for new weather conditions
	X_new = [['Sunny', 'Cool', 'High', 'Strong'], ['Overcast', 'Hot', 'Normal', 'Weak']]
	y_pred = model.predict(X_new)
	
	# Print the predicted labels
	print("Predicted Labels:", y_pred)

# k-Nearest Neighbors
# Here is a coding example in Python using Scikit-learn library for k-NN classification:
	from sklearn.datasets import load_iris
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	
	# Load the Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create an instance of the k-NN Classifier
	model = KNeighborsClassifier(n_neighbors=3)
	
	# Train the model
	model.fit(X_train, y_train)
	
	# Make predictions on the testing set
	y_pred = model.predict(X_test)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	accuracy = accuracy_score(y_test, y_pred)
	print("Accuracy:", accuracy)

# Real world coding example
# In this example, we have a user-item matrix representing user ratings for different items. Each row corresponds to a user, and each column corresponds to an item. The values in the matrix represent user ratings, with 0 indicating that the user has not rated the item.
	import numpy as np
	from sklearn.neighbors import NearestNeighbors
	
	# User-item matrix
	user_item_matrix = np.array([
        [5, 3, 0, 4, 0, 0, 0],
	    [0, 0, 0, 0, 4, 0, 0],
	    [4, 0, 0, 5, 0, 1, 0],
	    [0, 0, 0, 0, 0, 0, 5],
	    [0, 0, 0, 0, 0, 0, 4],
	    [0, 0, 0, 0, 0, 0, 5]
	])
	
	# Create an instance of the NearestNeighbors model
	model = NearestNeighbors(metric='cosine', algorithm='brute')
	
	# Fit the model to the user-item matrix
	model.fit(user_item_matrix)
	
	# Define a user for whom we want to make recommendations
	user = user_item_matrix[2]  # User 3
	
	# Find the k nearest neighbors to the user
	k = 3
	distances, indices = model.kneighbors([user], n_neighbors=k+1)  # Include the user itself
	
	# Get the indices of the k nearest neighbors
	neighbor_indices = indices.flatten()[1:]
	
	# Calculate the average rating for each item by the neighbors
	item_ratings = np.mean(user_item_matrix[neighbor_indices], axis=0)
	
	# Find the indices of items that the user has not rated
	unrated_indices = np.where(user == 0)[0]
	
	# Sort the unrated items based on the predicted ratings
	sorted_indices = np.argsort(item_ratings[unrated_indices])[::-1]
	
	# Get the top recommended items for the user
	top_items = unrated_indices[sorted_indices][:k]
	
	# Print the top recommended items
	print("Top Recommended Items:", top_items)

# Neural networks
# Here is a coding example in Python using the Keras library for building a neural network for image classification:
	import numpy as np
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
	
	# Load the dataset (example: MNIST digits dataset)
	(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
	
	# Preprocess the data
	x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
	x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0
	
	# Create the neural network model
	model = keras.Sequential([
	    layers.Dense(64, activation="relu", input_shape=(28 * 28,)),
	    layers.Dense(64, activation="relu"),
	    layers.Dense(10, activation="softmax")
	])
	
	# Compile the model
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	
	# Train the model
	model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=2)
	
	# Evaluate the model
	loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
	print("Test Accuracy:", accuracy)
	
	# Make predictions
	predictions = model.predict(x_test[:5])
	predicted_labels = np.argmax(predictions, axis=1)
	print("Predicted Labels:", predicted_labels)

# Real-world coding example
# In this example, we will perform sentiment analysis using a neural network for text classification.
# Here is a real-world example of NLP using neural networks in Python:
	import numpy as np
	import tensorflow as tf
	from tensorflow import keras
	from tensorflow.keras import layers
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import LabelEncoder
	from tensorflow.keras.preprocessing.text import Tokenizer
	from tensorflow.keras.preprocessing.sequence import pad_sequences
	
	# Example dataset - Sentiment analysis
	texts = [
	    "I love this movie",
	    "This restaurant has great food",
	    "I'm feeling happy today",
	    "The customer service was terrible",
	    "I didn't like the product"
	]
	labels = ["positive", "positive", "positive", "negative", "negative"]
	
	# Preprocess the data
	tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
	tokenizer.fit_on_texts(texts)
	sequences = tokenizer.texts_to_sequences(texts)
	padded_sequences = pad_sequences(sequences, padding="post", truncating="post")
	
	label_encoder = LabelEncoder()
	encoded_labels = label_encoder.fit_transform(labels)
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)
	
	# Create the neural network model
	model = keras.Sequential([
	    layers.Embedding(input_dim=1000, output_dim=16, input_length=len(padded_sequences[0])),
	    layers.Flatten(),
	    layers.Dense(10, activation="relu"),
	    layers.Dense(1, activation="sigmoid")
	])
	
	# Compile the model
	model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
	
	# Train the model
	model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))
	
	# Make predictions
	sample_texts = [
	    "This movie was amazing",
	    "I had a terrible experience at the restaurant"
	]
	sample_sequences = tokenizer.texts_to_sequences(sample_texts)
	sample_padded_sequences = pad_sequences(sample_sequences, padding="post", truncating="post")
	
	predictions = model.predict(sample_padded_sequences)
	predicted_labels = ["positive" if pred > 0.5 else "negative" for pred in predictions]
	
	# Print the predicted labels
	print("Predicted Labels:", predicted_labels)

# Gradient Boosting Machines
# Coding example using XGBoost: Here is a coding example in Python using the XGBoost library for a regression problem:
	import numpy as np
	import xgboost as xgb
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import mean_squared_error
	
	# Load the dataset (example: Boston Housing dataset)
	from sklearn.datasets import load_boston
	boston = load_boston()
	X, y = boston.data, boston.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Convert the data to DMatrix format
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	
	# Set the hyperparameters
	params = {
	    'objective': 'reg:squarederror',
	    'learning_rate': 0.1,
	    'max_depth': 3,
	    'n_estimators': 100
	}
	
	# Train the model
	model = xgb.train(params, dtrain)
	
	# Make predictions
	y_pred = model.predict(dtest)
	
	# Calculate root mean squared error
	rmse = np.sqrt(mean_squared_error(y_test, y_pred))
	print("Root Mean Squared Error:", rmse)

# Real-world GBM coding example
# In this example, we assume that you have a dataset of energy load information that includes features like temperature, day of the week, and the corresponding energy load. Here is a real-world example of energy load forecasting using GBM in Python:
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.metrics import mean_absolute_error, mean_squared_error
	
	# Load the dataset (example: Energy load dataset)
	energy_data = pd.read_csv(“energy_data.csv”)
	
	# Preprocess the data
	# Assume the dataset has columns ‘datetime’, ‘temperature’, ‘dayofweek’, ‘load’ (target variable)
	features = [‘temperature’, ‘dayofweek’]
	target = ‘load’
	
	X = energy_data[features]
	y = energy_data[target]
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create and train the GBM model
	model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
	model.fit(X_train, y_train)
	
	# Make predictions on the testing set
	y_pred = model.predict(X_test)
	
	# Calculate evaluation metrics
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	rmse = np.sqrt(mse)
	
	print(“Mean Absolute Error:”, mae)
	print(“Mean Squared Error:”, mse)
	print(“Root Mean Squared Error:”, rmse)

# XGBoost
# Coding example using XGBoost: Here is a coding example in Python using the XGBoost library for a classification problem:
	import numpy as np
	import xgboost as xgb
	from sklearn.datasets import load_breast_cancer
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	
	# Load the dataset (example: Breast Cancer dataset)
	data = load_breast_cancer()
	X, y = data.data, data.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Convert the data to DMatrix format
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test, label=y_test)
	
	# Set the hyperparameters
	params = {
	    'objective': 'binary:logistic',
	    'learning_rate': 0.1,
	    'max_depth': 3,
	    'n_estimators': 100
	}
	
	# Train the model
	model = xgb.train(params, dtrain)
	
	# Make predictions
	y_pred = model.predict(dtest)
	y_pred_labels = np.round(y_pred)
	
	# Calculate accuracy
	accuracy = accuracy_score(y_test, y_pred_labels)
	print("Accuracy:", accuracy)

# Real-world XGBoost coding example
# In this example, we assume you have a fraud detection dataset with features and a target variable 'fraud' indicating whether a transaction is fraudulent (1) or not (0). Here is a real-world example of fraud detection and cybersecurity using XGBoost in Python:
	import pandas as pd
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report
	import xgboost as xgb
	
	# Load the dataset (example: Fraud detection dataset)
	data = pd.read_csv("fraud_dataset.csv")
	
	# Preprocess the data
	# Assume the dataset has features and a target variable 'fraud' (0: non-fraud, 1: fraud)
	X = data.drop('fraud', axis=1)
	y = data['fraud']
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create the XGBoost classifier
	model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
	
	# Train the model
	model.fit(X_train, y_train)
	
	# Make predictions on the testing set
	y_pred = model.predict(X_test)
	
	# Evaluate the model
	report = classification_report(y_test, y_pred)
	print(report)

# Exercises and solutions
# Naive Bayes exercise: Classifying email messages as spam or not spam
# Objective: Using a dataset of email messages labeled as either spam or not spam (ham), build a Naive Bayes model to classify these messages.
# Dataset features:
# •	text: The content of the email.
# •	label: The classification of the email (either "spam" or "ham").
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the email dataset.
#   c.	Transform email text into a matrix of token counts using CountVectorizer.
# 2.	Model building:
#   a.	Train a Naive Bayes classifier on the training set.
#   b.	Predict the label on the testing set.
# 3.	Evaluation:
#   a.	Calculate the accuracy, precision, recall, and F1-score of the model on the testing set.
#
# Code implementation:
	# Import necessary libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.feature_extraction.text import CountVectorizer
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.metrics import classification_report
	
    # Load the dataset
	data = pd.read_csv('spam_emails.csv')
	
	# Transform the email text into a matrix of token counts
	vectorizer = CountVectorizer(stop_words='english')
	X = vectorizer.fit_transform(data['text'])
	y = data['label']
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the Naive Bayes model
	classifier = MultinomialNB()
	classifier.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = classifier.predict(X_test)
	
	# Evaluate the model
	print(classification_report(y_test, y_pred))

# Task for the student:
# •	Experiment with different preprocessing techniques on the text data, such as stemming, lemmatization, or using a TF-IDF vectorizer instead of a simple count vectorizer.
# •	Try using a different variant of Naive Bayes available in scikit-learn, like BernoulliNB or GaussianNB, and observe any changes in performance.
# •	Analyze false positives and false negatives to identify patterns or potential areas of improvement.

# k-NN exercise: Classifying types of flowers based on measurements
# Objective: Using the famous Iris dataset, which contains measurements of iris flowers of three species (Setosa, Versicolour, and Virginica), build a k-NN model to classify these flowers based on their measurements.
# Dataset features
# •	sepal_length: Length of the sepal in cm.
# •	sepal_width: Width of the sepal in cm.
# •	petal_length: Length of the petal in cm.
# •	petal_width: Width of the petal in cm.
# •	species: The species of the flower (Setosa, Versicolour, Virginica).
# Steps
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the Iris dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building
#   a.	Train a k-NN classifier with different values of k.
#   b.	Predict the flower species on the testing set.
# 3.	Evaluation
#   a.	Calculate the accuracy of the model on the testing set for each k value.
#   b.	Plot the results to visualize the best k.
#
# Code implementation
	# Import necessary libraries
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.metrics import accuracy_score
	
	# Load the Iris dataset
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Choose different values of k
	k_values = np.arange(1, 50)
	accuracies = []
	
	for k in k_values:
	    # Build the k-NN model
	    classifier = KNeighborsClassifier(n_neighbors=k)
	    classifier.fit(X_train, y_train)
	
	    # Predict on the testing set
	    y_pred = classifier.predict(X_test)
	
	    # Evaluate the model
	    accuracies.append(accuracy_score(y_test, y_pred))
	
	# Plot the results
	plt.plot(k_values, accuracies, marker='o')
	plt.xlabel('Value of k')
	plt.ylabel('Accuracy')
	plt.title('Accuracy vs. Value of k')
	plt.grid(True)
	plt.show()

# Task for the student:
# •	Analyze the effect of distance metrics (for example, Euclidean, Manhattan, Minkowski) on the model's performance.
# •	Explore the influence of feature scaling on the k-NN algorithm. Try using StandardScaler or MinMaxScaler from scikit-learn.
# •	Examine the boundaries created by the k-NN classifier for each class by visualizing them on a 2D plane for any two features.

# Neural Network exercise: Handwritten digit classification with the MNIST dataset
# Objective: Use the MNIST dataset, a collection of 28x28 grayscale images of handwritten digits (0 through 9), to train a neural network model that can classify these images based on the digit they represent.
# Dataset features:
# ●	28x28 pixel values for each image (flattened, this becomes a 784-dimensional vector).
# ●	Label: The actual digit (0-9) the image represents.
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the MNIST dataset.
#   c.	Normalize the image data to the range [0, 1].
#   d.	Convert the labels to one-hot encoded vectors.
# 2.	Model building
#   a.	Define a neural network architecture using sequential API.
#   b.	Compile the model specifying a loss function, optimizer, and evaluation metric.
#   c.	Train the model using the training data.
# 3.	Evaluation
#   a.	Evaluate the model's accuracy on the testing set.
#   b.	Optionally, visualize the predictions on a sample of test images.
#
# Code implementation
	# Import necessary libraries
	import numpy as np
	import tensorflow as tf
	from tensorflow.keras import Sequential
	from tensorflow.keras.layers import Dense, Flatten
	from tensorflow.keras.utils import to_categorical
	
	# Load the MNIST dataset
	mnist = tf.keras.datasets.mnist
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	
	# Normalize the data
	X_train, X_test = X_train / 255.0, X_test / 255.0
	
	# Convert labels to one-hot encoded vectors
	y_train_one_hot = to_categorical(y_train)
	y_test_one_hot = to_categorical(y_test)
	
	# Define the neural network architecture
	model = Sequential([
	    Flatten(input_shape=(28, 28)),
	    Dense(128, activation='relu'),
	    Dense(10, activation='softmax')
	])
	
	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	
	# Train the model
	model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_data=(X_test, y_test_one_hot))
	
	# Evaluate the model
	test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, verbose=2)
	print("\nTest accuracy:", test_acc)

# Task for the student:
# •	Experiment with different neural network architectures by adding more layers or changing the number of neurons.
# •	Try using different activation functions like tanh or sigmoid.
# •	Implement dropout regularization using Dropout layer to prevent overfitting.
# •	Visualize the model's predictions on a subset of test images and compare them to the true labels.

# GBM exercise: Predicting house prices with the Boston Housing dataset
# Objective: Use the Boston Housing dataset, which contains data about various houses in Boston and their respective prices, to build a GBM model that predicts the median value of owner-occupied homes.
# Dataset features:
# •	CRIM: Per capita crime rate by town.
# •	ZN: Proportion of residential land zoned for large lots.
# •	INDUS: Proportion of non-retail business acres per town.
# •	CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
# •	NOX: Nitrogen oxides concentration (parts per 10 million).
# •	RM: Average number of rooms per dwelling.
# •	AGE: Proportion of owner-occupied units built prior to 1940.
# •	DIS: Weighted distances to five Boston employment centers.
# •	RAD: Index of accessibility to radial highways.
# •	TAX: Full-value property tax rate per $10,000.
# •	PTRATIO: Pupil-teacher ratio by town.
# •	B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town.
# •	LSTAT: Percentage lower status of the population.
# •	MEDV: Median value of owner-occupied homes in $1000s. (This is the target variable)
#
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the Boston Housing dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building
#   a.	Train a GBM regressor on the training set.
#   b.	Predict house prices on the testing set.
# 3.	Evaluation
#   a.	Calculate the Mean Absolute Error (MAE), MSE, and R^2 score for the predictions.
#
# Code implementation
    # Import necessary libraries
	import numpy as np
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import GradientBoostingRegressor
	from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
	
	# Load the Boston Housing dataset
	boston = datasets.load_boston()
	X = boston.data
	y = boston.target
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the GBM regressor
	gbm = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
	gbm.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = gbm.predict(X_test)
	
	# Evaluate the model
	mae = mean_absolute_error(y_test, y_pred)
	mse = mean_squared_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)
	
	print(f"Mean Absolute Error: {mae:.2f}")
	print(f"Mean Squared Error: {mse:.2f}")
	print(f"R^2 Score: {r2:.2f}")

# Task for the student
# •	Experiment with different hyperparameters of the GBM model such as n_estimators, learning_rate, and max_depth.
# •	Use feature importance provided by the GBM model to identify the most significant predictors for the house prices.
# •	Compare the GBM model's performance with a basic linear regression model to gauge the improvements.

# XGBoost exercise: Predicting iabetes Ouotcomes with the Pima Indians Diabetes dataset
# Objective: Utilize the Pima Indians Diabetes dataset, which contains health measurements and outcomes for female patients, to train an XGBoost model to predict whether a patient has diabetes.
# Dataset features:
# •	Pregnancies: Number of times pregnant.
# •	Glucose: Plasma glucose concentration.
# •	BloodPressure: Diastolic blood pressure (mm Hg).
# •	SkinThickness: Triceps skin fold thickness (mm).
# •	Insulin: 2-Hour serum insulin (mu U/ml).
# •	BMI: Body mass index.
# •	DiabetesPedigreeFunction: Diabetes pedigree function.
# •	Age: Age (years).
# •	Outcome: Class variable (0 or 1) indicating whether the patient has diabetes (1) or not (0).
#
# Steps:
# 1.	Data preparation
#   a.	Import necessary libraries.
#   b.	Load the Pima Indians Diabetes dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building
#   a.	Train an XGBoost classifier on the training set.
#   b.	Predict diabetes outcomes on the testing set.
# 3.	Evaluation
#   a.	Calculate accuracy, precision, recall, and F1-score for the predictions.
#
# Code implementation
	# Import necessary libraries
	import numpy as np
	import xgboost as xgb
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report
	
	# Load the Pima Indians Diabetes dataset
	data = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")
	X = data[:, 0:8]
	y = data[:, 8]
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the XGBoost classifier
	model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=100, seed=42)
	model.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = model.predict(X_test)
	
	# Evaluate the model
	print(classification_report(y_test, y_pred))

# Task for the student
# •	Experiment with different hyperparameters of the XGBoost model such as n_estimators, learning_rate, and max_depth.
# •	Utilize the feature_importances_ attribute of the trained model to determine which features are most influential in predicting diabetes outcomes.
# •	Compare the performance of the XGBoost model with other classifiers (for example, RandomForest, Logistic Regression) to assess its relative strength.

