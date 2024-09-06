# Examples of typical data structures
# Some examples of the data structures are explained below: 

# Arrays
# Let us consider an example where we want to store the temperatures of a week in an array:
	# Define an array to store temperatures
	temperatures = [25.5, 27.3, 23.8, 26.1, 24.9, 22.7, 28.5]
	
	# Accessing elements in the array
	print(temperatures[0])  # Access the first temperature (25.5)
	print(temperatures[3])  # Access the fourth temperature (26.1)
	
	# Updating an element in the array
	temperatures[2] = 24.2  # Update the third temperature to 24.2
	
	# Length of the array
	print(len(temperatures))  # Output: 7
	
	# Looping through the array
	for temp in temperatures:
	    print(temp)
	
	# Output:
	# 25.5
	# 27.3
	# 24.2
	# 26.1
	# 24.9
	# 22.7
	# 28.5

# Matrices
# Let us consider an example where we want to represent a 3x3 matrix that stores the grayscale pixel values of an image:
	+---+---+---+
	| 150 | 200 | 180 |
	+---+---+---+
	| 100 | 120 | 140 |
	+---+---+---+
	| 90  | 80  | 110 |
	+---+---+---+
# In this example, we have a 3x3 matrix representing an image's grayscale pixel values. Each element in the matrix corresponds to the intensity value of a specific pixel in the image. The matrix has three rows and three columns, forming a rectangular grid.
# To represent this matrix in code, we can use various programming languages. Here is an example using Python:
	# Define the matrix
	image_matrix = [
	    [150, 200, 180],
	    [100, 120, 140],
	    [90, 80, 110]
	]
	
	# Accessing elements in the matrix
	print(image_matrix[0][0])  # Access the first element (150)
	print(image_matrix[1][2])  # Access the element at the second row and third column (140)
	
	# Updating an element in the matrix
	image_matrix[2][1] = 85  # Update the element at the third row and second column to 85
	
	# Dimensions of the matrix
	rows = len(image_matrix)
	columns = len(image_matrix[0])
	print("Rows:", rows)  # Output: 3
	print("Columns:", columns)  # Output: 3
	
	# Looping through the matrix
	for row in image_matrix:
	    for element in row:
	        print(element)
	
	# Output:
	# 150
	# 200
	# 180
	# 100
	# 120
	# 140
	# 90
	# 85
	# 110

# Tensors
# Let us consider an example where we want to represent a tensor to store RGB color images with dimensions of height, width, and channels:
	Tensor Shape: (3, 224, 224, 3)
# In this example, we have a tensor representing RGB color images. The tensor shape is defined as (3, 224, 224, 3), indicating that the tensor has four dimensions. The first dimension (3) represents the number of images, the second and third dimensions (224, 224) represent the height and width of each image, and the fourth dimension (3) represents the RGB channels.
# To represent this tensor in code, we can use various programming languages. Here is an example using Python and the NumPy library:
	import numpy as np
	
	# Create a tensor representing RGB color images
	image_tensor = np.zeros((3, 224, 224, 3), dtype=np.uint8)
	
	# Accessing elements in the tensor
	print(image_tensor[0, 0, 0, 0])  # Access the first element of the first image's first pixel's red channel
	
	# Updating an element in the tensor
	image_tensor[1, 100, 100, 1] = 128  # Update the second image's pixel at coordinates (100, 100) in the green channel
	
	# Dimensions of the tensor
	tensor_shape = image_tensor.shape
	print("Tensor Shape:", tensor_shape)  # Output: (3, 224, 224, 3)
	
	# Looping through the tensor
	for image_index in range(tensor_shape[0]):
	    for row_index in range(tensor_shape[1]):
	        for col_index in range(tensor_shape[2]):
	            for channel_index in range(tensor_shape[3]):
	                print(image_tensor[image_index, row_index, col_index, channel_index])
	
	# Output: (prints all elements of the tensor)

# Linked lists
# Let us consider an example of a linked list that stores a sequence of integers:
	Node 1        Node 2        Node 3        Node 4
	+-------+     +-------+     +-------+     +-------+
	| Data: 10|    | Data: 20|    | Data: 30|    | Data: 40|
	| Next: -----→| Next: -----→| Next: -----→| Next: Null|
	+-------+     +-------+     +-------+     +-------+
# In this example, we have a linked list with four nodes, each containing an integer value and a reference to the next node. The last node's reference is set to Null to indicate the end of the list.
# To represent this linked list in code, we can define a node structure and create the necessary functions to manipulate the list. Here is an example using Python:
	# Node class definition
	class Node:
	    def __init__(self, data):
	        self.data = data
	        self.next = None
	
	# Linked list class definition
	class LinkedList:
	    def __init__(self):
	        self.head = None
	
	    def insert(self, data):
	        new_node = Node(data)
	        if self.head is None:
	            self.head = new_node
	        else:
	            current = self.head
	            while current.next:
	                current = current.next
	            current.next = new_node
	
	    def display(self):
	        current = self.head
	        while current:
	            print(current.data)
	            current = current.next
	
    # Create an instance of the linked list
	my_list = LinkedList()
	
	# Insert elements into the linked list
	my_list.insert(10)
	my_list.insert(20)
	my_list.insert(30)
	my_list.insert(40)
	
	# Display the linked list
	my_list.display()
	
	# Output:
	# 10
	# 20
	# 30
	# 40

# Graphs
# Let us consider an example of a simple undirected graph that represents a social network:
	Graph Example: Social Network
	       A
	      / \
	     B---C
	    / \ / \
	   D---E---F
# In this example, we have a social network represented by a graph. The graph comprises six nodes (A, B, C, D, E, F) and seven edges (A-B, A-C, B-D, B-E, C-E, C-F, E-F). The connections between nodes represent relationships between individuals in the social network.
# To represent this graph in code, we can use various data structures and algorithms. One common approach is using an adjacency list representation. Here is an example using Python:
	# Graph class definition
	class Graph:
	    def __init__(self):
	        self.adjacency_list = {}
	
	    def add_vertex(self, vertex):
	        self.adjacency_list[vertex] = []
	
	    def add_edge(self, vertex1, vertex2):
	        self.adjacency_list[vertex1].append(vertex2)
	        self.adjacency_list[vertex2].append(vertex1)
	
	    def display(self):
	        for vertex, neighbors in self.adjacency_list.items():
	            print(vertex, "->", neighbors)
	
	# Create an instance of the graph
	social_network = Graph()
	
	# Add vertices (nodes)
	social_network.add_vertex("A")
	social_network.add_vertex("B")
	social_network.add_vertex("C")
	social_network.add_vertex("D")
	social_network.add_vertex("E")
	social_network.add_vertex("F")
	
	# Add edges (relationships)
	social_network.add_edge("A", "B")
	social_network.add_edge("A", "C")
	social_network.add_edge("B", "D")
	social_network.add_edge("B", "E")
	social_network.add_edge("C", "E")
	social_network.add_edge("C", "F")
	social_network.add_edge("E", "F")
	
	# Display the graph
	social_network.display()
	
	# Output:
	# A -> ['B', 'C']
	# B -> ['A', 'D', 'E']
	# C -> ['A', 'E', 'F']
	# D -> ['B']
	# E -> ['B', 'C', 'F']
	# F -> ['C', 'E']

# Hash tables
# Let us consider an example where we want to store a collection of student names and their corresponding ages using a hash table:
    Hash Table Example: Student Ages
	Bucket 0: Empty
	Bucket 1: ("John", 20) -> ("Sarah", 18) -> ("Emily", 19)
	Bucket 2: ("Michael", 21)
	Bucket 3: Empty
# In this example, we have a hash table that stores student names and their ages. The hash table consists of four buckets, each represented by an index. Each bucket can hold multiple key-value pairs, with keys (student names) mapped to specific buckets using a hash function.
# To represent this hash table in code, we can use various programming languages. Here is an example using Python:
	# Hash table class definition
	class HashTable:
	    def __init__(self, size):
	        self.size = size
	        self.buckets = [[] for _ in range(size)]
	
	    def hash_function(self, key):
	        return hash(key) % self.size
	
	    def insert(self, key, value):
	        index = self.hash_function(key)
	        bucket = self.buckets[index]
	        for item in bucket:
	            if item[0] == key:
	                item[1] = value
	                return
	        bucket.append((key, value))
	
	    def get(self, key):
	        index = self.hash_function(key)
	        bucket = self.buckets[index]
	        for item in bucket:
	            if item[0] == key:
	                return item[1]
	        return None
	
	    def display(self):
	        for index, bucket in enumerate(self.buckets):
	            print("Bucket", index)
	            for item in bucket:
	                print(item)
	
	# Create an instance of the hash table
	student_ages = HashTable(size=4)
	
	# Insert key-value pairs into the hash table
	student_ages.insert("John", 20)
	student_ages.insert("Sarah", 18)
	student_ages.insert("Emily", 19)
	student_ages.insert("Michael", 21)
	
	# Retrieve values based on keys from the hash table
	age = student_ages.get("Sarah")
	print("Sarah's age:", age)  # Output: 18
	
	# Display the hash table
	student_ages.display()
	
	# Output:
	# Sarah's age: 18
	# Bucket 0
	# Bucket 1
	# ('John', 20)
	# ('Sarah', 18)
	# ('Emily', 19)
	# Bucket 2
	# ('Michael', 21)
	# Bucket 3

# Queues
# Let us consider an example where we want to simulate a ticket queue at a movie theatre using a queue data structure:
	Queue Example: Movie Ticket Queue
	Front -> ["John", "Alice", "Emily", "Michael"] <- Rear
# In this example, we have a queue representing the ticket queue at a movie theatre. The front of the queue is where new people join the queue, and the rear of the queue is where people are served and removed from the queue.
# To represent this queue in code, we can use various programming languages. Here is an example using Python:
	# Queue class definition
	class Queue:
	    def __init__(self):
            self.queue = []
	
	    def enqueue(self, item):
	        self.queue.append(item)
	
	    def dequeue(self):
	        if self.is_empty():
	            return None
	        return self.queue.pop(0)
	
	    def is_empty(self):
	        return len(self.queue) == 0
	
	    def size(self):
	        return len(self.queue)
	
	    def display(self):
	        print("Front ->", self.queue, "<- Rear")
	
	# Create an instance of the queue
	ticket_queue = Queue()
	
	# Enqueue people into the queue
	ticket_queue.enqueue("John")
	ticket_queue.enqueue("Alice")
	ticket_queue.enqueue("Emily")
	ticket_queue.enqueue("Michael")
	
	# Display the queue
	ticket_queue.display()
	
	# Dequeue people from the queue
	served_person = ticket_queue.dequeue()
	print("Served Person:", served_person)  # Output: Served Person: John
	
	# Display the updated queue
	ticket_queue.display()
	
	# Output:
	# Front -> ['John', 'Alice', 'Emily', 'Michael'] <- Rear
	# Served Person: John
	# Front -> ['Alice', 'Emily', 'Michael'] <- Rear

# Trees
# Example: Let us consider an example of a binary tree that represents a family tree:
	Tree Example: Family Tree
	                John
	               /    \
	            Alice   Bob
	           /   \      \
	        Emily  Mark   Sarah
# In this example, we have a binary tree representing a family tree. The root node is John, and each node represents a family member. The relationships are established through parent-child connections. For instance, John has two children, Alice and Bob, while Alice has two children, Emily and Mark. Bob has one child, Sarah.
# To represent this tree in code, we can use various data structures and algorithms. Here is an example using Python:
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

# Knowledge graph
# Here's an example of how you can create and work with knowledge graphs using Python and the rdflib library:
	from rdflib import Graph, Namespace, Literal, URIRef
	
	# Create a new graph
	graph = Graph()
	
	# Define namespace prefixes
	ex = Namespace("http://example.com/")
	
	# Add triples to the graph
	graph.add((ex.alice, ex.knows, ex.bob))
	graph.add((ex.bob, ex.knows, ex.charlie))
	graph.add((ex.alice, ex.age, Literal(25)))
	graph.add((ex.bob, ex.age, Literal(30)))
	graph.add((ex.charlie, ex.age, Literal(35)))
	
	# Query the graph
	print("People Alice knows:")
	for person in graph.objects(ex.alice, ex.knows):
	    print(person)
	
	print("\nPeople and their ages:")
	for person, age in graph.subject_objects(ex.age):
	    print(f"{person}: {age}")
	
	# Serialize the graph to a file (in Turtle format)
	with open("knowledge_graph.ttl", "w") as file:
	    file.write(graph.serialize(format="turtle"))
# When you run this code, it will output:
	People Alice knows:
	http://example.com/bob
	
	People and their ages:
	http://example.com/alice: 25
	http://example.com/bob: 30
	http://example.com/charlie: 35