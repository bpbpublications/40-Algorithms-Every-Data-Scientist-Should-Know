# Natural Language Processing 

# Natural Language Understanding
# Python coding example for NLU 
# Here is a Python coding example using the spaCy library, which is a popular NLP library, to demonstrate some of the NLU tasks mentioned above:
	import spacy
	
	# Load the spaCy English model
	nlp = spacy.load("en_core_web_sm")
	
	# Define a sample text for NLU processing
	text = "I want to book a flight from New York to London next Monday."
	
	# Perform NLU tasks on the text
	doc = nlp(text)
	
	# Tokenization
	tokens = [token.text for token in doc]
	print("Tokens:", tokens)
	
	# Part-of-Speech (POS) Tagging
	pos_tags = [(token.text, token.pos_) for token in doc]
	print("POS Tags:", pos_tags)
	
	# Named Entity Recognition (NER)
	ner_entities = [(entity.text, entity.label_) for entity in doc.ents]
	print("Named Entities:", ner_entities)
	
	# Sentiment Analysis (using the built-in polarity scoring)
	sentiment = doc._.polarity
	print("Sentiment:", sentiment)
	
	# Syntax and Dependency Parsing
	syntax_tree = [(token.text, token.dep_, token.head.text) for token in doc]
	print("Syntax Tree:", syntax_tree)
	
	# Topic Modeling (using simple noun phrases as topics)
	topics = [chunk.text for chunk in doc.noun_chunks]
	print("Topics:", topics)

# Natural Language Generation
# Python coding example for NLG
# Here is a Python coding example using the Natural Language ToolKit (NLTK) library, which is a popular NLP library, to demonstrate text generation using NLG techniques:
	import nltk
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from nltk.probability import FreqDist
	
	# Sample text for text generation
	text = "Natural language generation is the process of converting structured data into human-like text."
	
	# Tokenize the text into words
	tokens = word_tokenize(text)
	
	# Remove stop words
	stop_words = set(stopwords.words('english'))
	tokens = [token for token in tokens if token.lower() not in stop_words]
	
	# Calculate the frequency distribution of the remaining words
	frequency_dist = FreqDist(tokens)
	
	# Generate text by sampling words based on their frequencies
	generated_text = ' '.join(frequency_dist.keys())
	
	print("Generated Text:", generated_text)

# Large Language Models
# Python coding example for LLMs
# Here is a Python coding example using the Hugging Face Transformers library, which provides easy access to pre-trained LLMs, to demonstrate how to generate text using a large language model:
	from transformers import GPT2LMHeadModel, GPT2Tokenizer
	
	# Load the pretrained GPT-2 model and tokenizer
	model_name = 'gpt2'
	model = GPT2LMHeadModel.from_pretrained(model_name)
	tokenizer = GPT2Tokenizer.from_pretrained(model_name)
	
	# Input text for text generation
	input_text = "Once upon a time"
	
	# Tokenize the input text
	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	
	# Generate text using the model
	output = model.generate(input_ids, max_length=50, num_return_sequences=3, temperature=0.7)
	
	# Decode and print the generated text
	generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
	for text in generated_text:
	    print("Generated Text:", text)

# Generative AI
# Python coding example for Generative AI
# Here is a Python coding example using the TensorFlow library to demonstrate text generation using a pre-trained generative AI model called GPT-2:
	import tensorflow as tf
	import gpt_2_simple as gpt2
	
	# Download and load the pre-trained GPT-2 model
	gpt2.download_gpt2(model_name='124M')
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, model_name='124M')
	
	# Generate text using the GPT-2 model
	generated_text = gpt2.generate(sess, model_name='124M', length=100, prefix="Once upon a time")
	
	print("Generated Text:")
	print(generated_text)

# Coding example
# Here is a coding example for NLP using Python and the NLTK:
	import nltk
	from nltk.tokenize import word_tokenize
	from nltk.corpus import stopwords
	from nltk.stem import WordNetLemmatizer
	
	# Sample text for NLP processing
	text = "Natural Language Processing (NLP) is a subfield of artificial intelligence that focuses on the interaction between computers and humans through natural language."
	
	# Tokenization
	tokens = word_tokenize(text)
	
	# Removing stop words
	stop_words = set(stopwords.words('english'))
	filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
	
	# Lemmatization
	lemmatizer = WordNetLemmatizer()
	lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
	
	# POS Tagging
	pos_tags = nltk.pos_tag(lemmatized_tokens)
	
	# Named Entity Recognition (NER)
	ner_tags = nltk.ne_chunk(pos_tags)
	
	# Print the processed tokens, POS tags, and NER tags
	print("Tokenized Text:", tokens)
	print("Filtered Tokens:", filtered_tokens)
	print("Lemmatized Tokens:", lemmatized_tokens)
	print("POS Tags:", pos_tags)
	print("NER Tags:", ner_tags)