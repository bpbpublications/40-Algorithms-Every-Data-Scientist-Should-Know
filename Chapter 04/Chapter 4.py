# Basic Supervised Learning Algorithms
# Linear regression
# Here is a coding example in Python using Scikit-learn library for linear regression:
	
# Node class definition
	class Node:
	    def __init__(self, data):
	        self.data = data
	        self.left = None
	        self.right = None
	
	# Create the family tree
	john = Node("John")
	alice = Node("Alice")
	bob = Node("Bob")
	emily = Node("Emily")
	mark = Node("Mark")
	sarah = Node("Sarah")
	
	john.left = alice
	john.right = bob
	alice.left = emily
	alice.right = mark
	bob.right = sarah
	
	# Display the family tree
	def display_tree(node, level=0):
	    if node:
	        print("  " * level + str(node.data))
	        display_tree(node.left, level + 1)
	        display_tree(node.right, level + 1)
	
	display_tree(john)
	
	# Output:
	# John
	#   Alice
	#     Emily
	#     Mark
	#   Bob
	#     Sarah
	
# Real-world coding example
# Let us assume we have a dataset containing information about houses, including the size of the house (in square feet) and the corresponding sale prices (in dollars). We will use linear regression to build a model that can predict the sale price based on the size of the house.
	import numpy as np
	import matplotlib.pyplot as plt
	from sklearn.linear_model import LinearRegression
	
	# Sample data (house size in square feet and sale prices)
	house_sizes = np.array([1000, 1500, 1200, 1800, 1350, 2000, 1600, 2200, 1850])
	sale_prices = np.array([300000, 450000, 350000, 500000, 400000, 550000, 420000, 600000, 480000])
	
	# Reshape the data to 2D arrays
	X = house_sizes.reshape((-1, 1))
	y = sale_prices.reshape((-1, 1))
	
	# Create a Linear Regression model
	model = LinearRegression()
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions
	X_new = np.array([[1700], [1900], [2100]])  # New house sizes for prediction
	y_pred = model.predict(X_new)
	
	# Print the coefficient and intercept of the linear regression line
	print("Coefficient:", model.coef_[0][0])
	print("Intercept:", model.intercept_[0])
	
	# Plot the data points and the linear regression line
	plt.scatter(X, y, color='blue', label='Actual Prices')
	plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
	plt.scatter(X_new, y_pred, color='green', label='Predicted Prices')
	plt.xlabel('House Size (sq. ft)')
	plt.ylabel('Sale Price ($)')
	plt.title('Linear Regression - House Prices')
	plt.legend()
	plt.show() 

# Logistic regression
# Here is a coding example in Python using the Scikit-learn library for logistic regression:
	# Import the required libraries
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	import numpy as np
	
	# Prepare the dataset
	X = np.array([[2, 1], [4, 5], [6, 3], [8, 7], [10, 9]])
	y = np.array([0, 0, 0, 1, 1])
	
	# Create an instance of the Logistic Regression model
	model = LogisticRegression()
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions on new data
	X_new = np.array([[3, 2], [9, 8]])
	y_pred = model.predict(X_new)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	y_pred_train = model.predict(X)
	accuracy = accuracy_score(y, y_pred_train)
	print("Accuracy:", accuracy)

# Real-world coding example
# Let us assume we have a dataset containing information about bank customers, including their age and income, and whether they have subscribed to a term deposit or not. We will use logistic regression to build a model that can predict whether a customer will subscribe to the term deposit based on their age and income.
	import numpy as np
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score
	
	# Prepare the dataset
	age = np.array([30, 45, 55, 25, 35, 52, 41, 28, 39, 57])
	income = np.array([50000, 75000, 100000, 35000, 60000, 90000, 80000, 40000, 58000, 120000])
	subscribed = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 1])
	
	# Create an instance of the Logistic Regression model
	model = LogisticRegression()
	
	# Prepare the data for training
	X = np.column_stack((age, income))
	y = subscribed
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions on new data
	X_new = np.array([[33, 55000], [47, 85000]])
	y_pred = model.predict(X_new)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	y_pred_train = model.predict(X)
	accuracy = accuracy_score(y, y_pred_train)
	print("Accuracy:", accuracy)

# Decision trees
# Here is a coding example in Python using Scikit-learn library for decision tree classification:
	from sklearn.datasets import load_iris
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	
	# Load the Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create an instance of the Decision Tree Classifier
	model = DecisionTreeClassifier()
	
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
# Let us assume we have a dataset containing information about bank customers, including their age, income, and whether they have defaulted on a loan or not. We will use a decision tree classifier to build a model that can predict whether a customer will default on a loan based on their age and income.
	import numpy as np
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.metrics import accuracy_score
	
	# Prepare the dataset
	age = np.array([30, 45, 55, 25, 35, 52, 41, 28, 39, 57])
	income = np.array([50000, 75000, 100000, 35000, 60000, 90000, 80000, 40000, 58000, 120000])
	defaulted = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
	
	# Create an instance of the Decision Tree Classifier
	model = DecisionTreeClassifier()
	
	# Prepare the data for training
	X = np.column_stack((age, income))
	y = defaulted
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions on new data
	X_new = np.array([[33, 55000], [47, 85000]])
	y_pred = model.predict(X_new)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	y_pred_train = model.predict(X)
	accuracy = accuracy_score(y, y_pred_train)
	print("Accuracy:", accuracy)

# Random forests
# Here is a coding example in Python using Scikit-learn library for random forest classification:
	from sklearn.datasets import load_iris
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	
	# Load the Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create an instance of the Random Forest Classifier
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	
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
# Let us assume we have a dataset containing information about bank customers, including their age, income, and whether they have defaulted on a loan or not. We will use a random forest classifier to build a model that can predict whether a customer will default on a loan based on their age and income.
	import numpy as np
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.metrics import accuracy_score
	
	# Prepare the dataset
	age = np.array([30, 45, 55, 25, 35, 52, 41, 28, 39, 57])
	income = np.array([50000, 75000, 100000, 35000, 60000, 90000, 80000, 40000, 58000, 120000])
	defaulted = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
	
	# Create an instance of the Random Forest Classifier
	model = RandomForestClassifier(n_estimators=100, random_state=42)
	
	# Prepare the data for training
	X = np.column_stack((age, income))
	y = defaulted
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions on new data
	X_new = np.array([[33, 55000], [47, 85000]])
	y_pred = model.predict(X_new)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	y_pred_train = model.predict(X)
	accuracy = accuracy_score(y, y_pred_train)
	print("Accuracy:", accuracy)

# Support Vector Machines
# Here is a coding example in Python using Scikit-learn library for SVM classification:
	from sklearn.datasets import load_iris
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	from sklearn.model_selection import train_test_split
	
	# Load the Iris dataset
	iris = load_iris()
	X = iris.data
	y = iris.target
	
	# Split the dataset into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Create an instance of the SVM Classifier
	model = SVC(kernel='linear')
	
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
# Let us assume we have a dataset containing information about bank customers, including their age, income, and whether they have defaulted on a loan or not. We will use an SVM classifier to build a model that can predict whether a customer will default on a loan based on their age and income.
	import numpy as np
	from sklearn.svm import SVC
	from sklearn.metrics import accuracy_score
	
	# Prepare the dataset
	age = np.array([30, 45, 55, 25, 35, 52, 41, 28, 39, 57])
	income = np.array([50000, 75000, 100000, 35000, 60000, 90000, 80000, 40000, 58000, 120000])
	defaulted = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
	
	# Create an instance of the SVM Classifier
	model = SVC(kernel='linear')
	
	# Prepare the data for training
	X = np.column_stack((age, income))
	y = defaulted
	
	# Train the model
	model.fit(X, y)
	
	# Make predictions on new data
	X_new = np.array([[33, 55000], [47, 85000]])
	y_pred = model.predict(X_new)
	
	# Print the predicted values
	print("Predicted Classes:", y_pred)
	
	# Calculate the accuracy of the model
	y_pred_train = model.predict(X)
	accuracy = accuracy_score(y, y_pred_train)
	print("Accuracy:", accuracy)

# Exercises
# Linear regression exercise: Predict house prices
# Objective: Given a dataset that contains information about houses and their respective prices, build a linear regression model to predict the price of a house based on its attributes.
# Dataset features:
#	area: Area of the house (in square feet).
#	bedrooms: Number of bedrooms.
#	bathrooms: Number of bathrooms.
#	age: Age of the house (in years).
#
# Target variable:
#	price: Price of the house.
#
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the dataset.
#   c.	Split the data into training and testing sets.
#
# 2.	Model building:
#   a.	Use the training set to build a linear regression model.
#   b.	Predict house prices on the testing set.
# 3.	Evaluation:
#   a.	Calculate the MSE between predicted and actual prices on the testing set.
#
# Code implementation:
	# Import necessary libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LinearRegression
	from sklearn.metrics import mean_squared_error
	
	# Load the dataset
	data = pd.read_csv('house_prices.csv')
	
    # Split the data
	X = data[['area', 'bedrooms', 'bathrooms', 'age']]
	y = data['price']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the linear regression model
	regressor = LinearRegression()
	regressor.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = regressor.predict(X_test)
	
	# Evaluate the model
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse}")

# Task for the student:
# •	Modify the features to include only 'area' and 'bedrooms'. How does this impact the performance?
# •	Explore other evaluation metrics such as mean absolute error (MAE) or R^2 score. How does the model fare on these metrics?

# Logistic regression exercise: Predicting customer churn
# Objective: Given a dataset of a telecommunications company's customers, build a logistic regression model to predict whether a customer will churn (leave the company) based on certain features.
# Dataset features:
# •	monthly_charge: The monthly charge the customer pays.
# •	contract_length: Length of the customer's contract (in months).
# •	support_calls: Number of customer support calls made by the customer in the last month.
# •	data_usage: Amount of data (in GB) used by the customer in the last month.
#
# Target variable:
# •	churn: Whether the customer churned or not. (1 for 'Yes', 0 for 'No').
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building:
#   a.	Use the training set to build a logistic regression model.
#   b.	Predict churn likelihood on the testing set.
# 3.	Evaluation:
#   a.	Calculate the accuracy of the model on the testing set.
#   b.	Generate a confusion matrix for further insights.
#
# Code implementation:
	# Import necessary libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.linear_model import LogisticRegression
	from sklearn.metrics import accuracy_score, confusion_matrix
	
	# Load the dataset
	data = pd.read_csv('customer_churn.csv')
	
	# Split the data
	X = data[['monthly_charge', 'contract_length', 'support_calls', 'data_usage']]
	y = data['churn']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the logistic regression model
	classifier = LogisticRegression(max_iter=1000)
	classifier.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = classifier.predict(X_test)
	
	# Evaluate the model
	accuracy = accuracy_score(y_test, y_pred)
	conf_matrix = confusion_matrix(y_test, y_pred)
	print(f"Accuracy: {accuracy * 100:.2f}%")
	print("Confusion Matrix:")
	print(conf_matrix)

# Task for the student:
# •	Explore the significance of each feature using the coef_ attribute of the trained model. Which feature(s) have the greatest impact on churn?
# •	Experiment with different features or combinations. How do they affect model performance?
# •	Compute other evaluation metrics like precision, recall, and the F1-score to get a holistic view of the model's performance.

# Decision tree exercise: Diagnosing plant diseases
# Objective: Given a dataset that contains information about different plant leaf characteristics, build a decision tree classifier to diagnose the type of disease a plant might have.
# Dataset features:
# ●	leaf_length: Length of the leaf (in cm).
# ●	leaf_width: Width of the leaf (in cm).
# ●	leaf_color_score: A numerical score representing the color of the leaf (1 for healthy green, 10 for brown).
# ●	spot_count: Number of spots or blemishes on the leaf.
#
# Target variable:
disease: Type of plant disease (for example, 'Healthy', 'Fungus', 'Bacteria', 'Virus').
#
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building:
#   a.	Use the training set to build a decision tree classifier.
#   b.	Predict the disease type on the testing set.
# 3.	Evaluation:
#   a.	Calculate the accuracy of the model on the testing set.
#   b.	Visualize the decision tree for insights.
#
# Code implementation:
	# Import necessary libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.tree import DecisionTreeClassifier, plot_tree
	from sklearn.metrics import accuracy_score
	import matplotlib.pyplot as plt
	
	# Load the dataset
	data = pd.read_csv('plant_diseases.csv')
	
	# Split the data
	X = data[['leaf_length', 'leaf_width', 'leaf_color_score', 'spot_count']]
	y = data['disease']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the decision tree classifier
	tree_classifier = DecisionTreeClassifier(max_depth=3)
	tree_classifier.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = tree_classifier.predict(X_test)
	
	# Evaluate the model
	accuracy = accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy * 100:.2f}%")
	
	# Visualize the decision tree
	plt.figure(figsize=(15,10))
	plot_tree(tree_classifier, filled=True, feature_names=X.columns, class_names=tree_classifier.classes_)
	plt.show()

# Task for the student:
# ●	Adjust the max_depth parameter of the decision tree. How does it impact the model's accuracy and interpretability?
# ●	Use other tree-specific metrics like Gini impurity or entropy to split the tree nodes. Observe any changes in tree structure or performance.
# ●	Consider pruning techniques or use a random forest to improve the model's generalization capability on unseen data.

# Random forest exercise: Predicting wine quality
# Objective: Given a dataset with various physicochemical properties of wines, build a random forest model to predict the quality of the wine.
# Dataset features:
# ●	fixed_acidity: Amount of tartaric acid in wine (in g/dm³).
# ●	volatile_acidity: Amount of acetic acid in wine (in g/dm³).
# ●	citric_acid: Amount of citric acid in wine (in g/dm³).
# ●	residual_sugar: Amount of sugar remaining after fermentation (in g/dm³).
# ●	chlorides: Amount of salt in the wine (in g/dm³).
# ●	alcohol: Alcohol content (% volume).
# ... (There could be more features in a typical dataset.)
#
# Target variable:
# ●	quality: Quality of the wine (scale from 1 to 10).
#
# Steps
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the dataset.
#   c.	Split the data into training and testing sets.
# 2.	Model building:
#   a.	Use the training set to build a random forest model.
#   b.	Predict the wine quality on the testing set.
# 3.	Evaluation:
#   a.	Calculate the MSE of the model on the testing set.
#   b.	Determine feature importance to understand which features contribute most to wine quality.
#
# Code implementation:
	# Import necessary libraries
	import pandas as pd
	from sklearn.model_selection import train_test_split
	from sklearn.ensemble import RandomForestRegressor
	from sklearn.metrics import mean_squared_error
	import matplotlib.pyplot as plt
	
	# Load the dataset
	data = pd.read_csv('wine_quality.csv')
	
	# Split the data
	X = data.drop('quality', axis=1)
	y = data['quality']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	
	# Build the random forest model
	rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
	rf_regressor.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = rf_regressor.predict(X_test)
	
	# Evaluate the model
	mse = mean_squared_error(y_test, y_pred)
	print(f"Mean Squared Error: {mse:.2f}")
	
	# Display feature importance
	features = X.columns
	importances = rf_regressor.feature_importances_
	indices = np.argsort(importances)
	
	plt.title('Feature Importances')
	plt.barh(range(len(indices)), importances[indices], align='center')
	plt.yticks(range(len(indices)), [features[i] for i in indices])
	plt.xlabel('Relative Importance')
	plt.show()

# Task for the student:
# ●	Adjust the n_estimators parameter in the random forest model. Observe how changing the number of trees influences performance.
# ●	Explore the impact of other hyperparameters like max_depth, min_samples_split, and min_samples_leaf on the model's performance.
# ●	Use a Random Forest classifier instead of a regressor if you categorize wine quality into classes (for example, "Low", "Medium", "High").

# SVM exercise: Classifying handwritten digits
# Objective: Use the famous MNIST dataset, which contains grayscale images of handwritten digits (from 0 to 9), to build an SVM model that classifies these digits.
# Dataset:
# ●	28x28 pixel images flattened to a 1D array of 784 values.
# ●	Target variable: the actual digit the image represents (from 0 to 9).
# Steps:
# 1.	Data preparation:
#   a.	Import necessary libraries.
#   b.	Load the MNIST dataset.
#   c.	Normalize the pixel values (typically to a range of 0-1 or -1 to 1).
#   d.	Split the data into training and testing sets.
# 2.	Model building:
#   a.	Train an SVM model using the training set.
#   b.	Predict the digit class on the testing set.
# 3.	Evaluation:
#   a.	Calculate the accuracy of the model on the testing set.
#   b.	Optionally, visualize the confusion matrix for detailed insights.
# Code implementation:
	# Import necessary libraries
    import numpy as np
	import matplotlib.pyplot as plt
	from sklearn import datasets, svm, metrics
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import MinMaxScaler
	
	# Load the MNIST dataset
	digits = datasets.load_digits()
	
	# Flatten the images and normalize the data
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))
	scaler = MinMaxScaler().fit(data)
	data = scaler.transform(data)
	
	# Split the data
	X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, random_state=42)
	
	# Build the SVM model
	classifier = svm.SVC(gamma=0.001)
	classifier.fit(X_train, y_train)
	
	# Predict on the testing set
	y_pred = classifier.predict(X_test)
	
	# Evaluate the model
	accuracy = metrics.accuracy_score(y_test, y_pred)
	print(f"Accuracy: {accuracy * 100:.2f}%")
	
	# Display a confusion matrix (optional)
	disp = metrics.plot_confusion_matrix(classifier, X_test, y_test)
	disp.figure_.suptitle("Confusion Matrix")
	plt.show()

# Task for the student:
# ●	Modify the SVM kernel (for example, linear, poly, rbf, sigmoid) and observe how it impacts the classification accuracy.
# ●	Play around with other SVM hyperparameters such as C, degree, and coef0 to see their influence on performance.
# Implement a multi-class classification strategy (for example, One-vs-One or One-vs-Rest) and determine if it affects the results.
