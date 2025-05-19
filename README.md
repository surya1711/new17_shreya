1) 1. Text Classification using TensorFlow
This code implements a sentiment analysis classifier for airline tweets using TensorFlow.
###################################################################
1. Text Classification using TensorFlow
This code implements a sentiment analysis classifier for airline tweets using TensorFlow. It's designed to categorize tweets as negative, neutral, or positive.

Data Loading and Exploration:

The code starts by importing necessary libraries including pandas, numpy, matplotlib, seaborn, and TensorFlow components.

It loads a dataset of airline tweets from a GitHub repository.

The dataset is explored using head() and value_counts() to understand its structure and distribution of sentiment labels.

Data Preprocessing:

The code extracts only the 'text' and 'airline_sentiment' columns.

It maps sentiment labels to numerical values: negative=0, neutral=1, positive=2.

The data is split into training and testing sets using train_test_split with an 80/20 ratio.

Text Tokenization:

The Tokenizer class converts text to sequences of integers.

Parameters include:

max_words=10000: Limits vocabulary to 10,000 most frequent words

oov_token='<OOV>': Handles out-of-vocabulary words

The tokenizer is fitted on the training data.

Text sequences are padded to a uniform length of 200 words using pad_sequences.

Model Building:

A sequential neural network is created with the following layers:

Embedding layer: Converts word indices to dense vectors of dimension 16

Dropout layer (0.2): Prevents overfitting

1D Convolutional layer: Extracts n-gram features

Global Max Pooling: Reduces dimensionality

Dense layer with ReLU activation

Another Dropout layer

Output layer with softmax activation for 3-class classification

The model is compiled with Adam optimizer and categorical crossentropy loss.

Training and Evaluation:

The model is trained for 10 epochs with a batch size of 32.

10% of training data is used for validation.

After training, the model is evaluated on the test set.

Predictions are converted from probabilities to class labels.

A classification report shows precision, recall, and F1-score for each sentiment class.

Visualization:

A confusion matrix is plotted to visualize prediction accuracy.

Training history (accuracy and loss) is plotted to show learning progress over epochs.
###################################################################
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
url = "https://raw.githubusercontent.com/ruchitgandhi/Twitter-Airline-Sentiment-Analysis/master/Tweets.csv"
df = pd.read_csv(url)

# Basic exploration
print(df.head())
print(df['airline_sentiment'].value_counts())

# Preprocessing
# Keep only the text and sentiment columns
df = df[['text', 'airline_sentiment']]

# Map sentiments to numbers
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['airline_sentiment'].map(sentiment_mapping)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['sentiment_label'], test_size=0.2, random_state=42
)

# Tokenize the text
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

# Convert labels to categorical
y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes=3)
y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes=3)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 16, input_length=max_len),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X_train_pad, y_train_cat,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test_cat)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
y_pred = model.predict(X_test_pad)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test_cat, axis=1)

# Print classification report
print(classification_report(y_test_classes, y_pred_classes, 
                           target_names=['negative', 'neutral', 'positive']))

# Plot confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

''' 
******************************************************************************************************************************************************************************************************

2) . Text Summarization
This implementation creates an extractive text summarization system for BBC news articles.
##############################################################
2. Text Summarization
This code implements an extractive text summarization system for BBC news articles, which identifies and extracts the most important sentences from a text.

Data Loading and Preprocessing:

The code imports necessary libraries, including NLTK for natural language processing.

It downloads required NLTK data: 'punkt' for tokenization and 'stopwords'.

BBC news articles are loaded from a CSV file.

Text preprocessing removes special characters and extra whitespace.

Sentence Similarity Functions:

sentence_similarity: Calculates similarity between two sentences using cosine distance.

Tokenizes sentences into words and removes stopwords

Creates vector representations of sentences

Returns a similarity score (1 - cosine distance)

build_similarity_matrix: Creates a matrix of similarity scores between all pairs of sentences.

Summarization Function:

generate_summary: The main function that produces an extractive summary.

Tokenizes the text into sentences

Creates a similarity matrix between sentences

Uses PageRank algorithm to rank sentences by importance

Selects the top N sentences based on their scores

Reorders selected sentences to match their original order

Joins sentences to form the summary

Application and Evaluation:

The summarization function is applied to a sample of articles.

Statistics are calculated for each summary:

Original length

Summary length

Compression ratio (summary length / original length)

Sample summaries are printed with their statistics.

Visualization:

A box plot shows compression ratios by article category.

A scatter plot compares original text length vs. summary length.
############################################################

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.cluster.util import cosine_distance
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the dataset
url = "https://raw.githubusercontent.com/codehax41/BBC-Text-Classification/master/bbc-text.csv"
df = pd.read_csv(url)

print(df.head())
print(df['category'].value_counts())

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Function to create a similarity matrix between sentences
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
                
    return similarity_matrix

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []
    
    sent1 = [w.lower() for w in word_tokenize(sent1) if w.lower() not in stop_words]
    sent2 = [w.lower() for w in word_tokenize(sent2) if w.lower() not in stop_words]
    
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    # Build the vector for the first sentence
    for w in sent1:
        vector1[all_words.index(w)] += 1
    
    # Build the vector for the second sentence
    for w in sent2:
        vector2[all_words.index(w)] += 1
    
    return 1 - cosine_distance(vector1, vector2)

# Function to generate summary
def generate_summary(text, num_sentences=5):
    stop_words = set(stopwords.words('english'))
    sentences = sent_tokenize(text)
    
    # If there are fewer sentences than requested, return all sentences
    if len(sentences) <= num_sentences:
        return ' '.join(sentences)
    
    # Generate similarity matrix
    sentence_similarity_matrix = build_similarity_matrix(sentences, stop_words)
    
    # Rank sentences using PageRank algorithm
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Sort sentences by score and select top sentences
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Get the top n sentences as the summary
    summary_sentences = [ranked_sentences[i][1] for i in range(min(num_sentences, len(ranked_sentences)))]
    
    # Sort the selected sentences based on their original order in the text
    summary_sentences = [s for s in sentences if s in summary_sentences]
    
    # Join the sentences to form the summary
    summary = ' '.join(summary_sentences)
    
    return summary

# Apply the summarization function to a sample of articles
sample_size = min(10, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Generate summaries
df_sample['summary'] = df_sample['text'].apply(lambda x: generate_summary(x, num_sentences=3))

# Calculate summary length statistics
df_sample['original_length'] = df_sample['text'].apply(len)
df_sample['summary_length'] = df_sample['summary'].apply(len)
df_sample['compression_ratio'] = df_sample['summary_length'] / df_sample['original_length']

# Print sample summaries
for i, row in df_sample.iterrows():
    print(f"Category: {row['category']}")
    print(f"Original Length: {row['original_length']} characters")
    print(f"Summary Length: {row['summary_length']} characters")
    print(f"Compression Ratio: {row['compression_ratio']:.2f}")
    print("Summary:")
    print(row['summary'])
    print("-" * 80)

# Visualize compression ratio by category
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='compression_ratio', data=df_sample)
plt.title('Compression Ratio by Category')
plt.xlabel('Category')
plt.ylabel('Compression Ratio (Summary Length / Original Length)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Visualize original vs summary length
plt.figure(figsize=(12, 6))
plt.scatter(df_sample['original_length'], df_sample['summary_length'], alpha=0.7)
plt.title('Original Text Length vs Summary Length')
plt.xlabel('Original Length (characters)')
plt.ylabel('Summary Length (characters)')
for i, row in df_sample.iterrows():
    plt.annotate(row['category'], (row['original_length'], row['summary_length']))
plt.tight_layout()
plt.show()

'''
***************************************************************************************************************************************************************

3). Spam Detection
This implementation creates an email spam detection system.
######################################################
3. Spam Detection
This code creates an email spam detection system using multiple machine learning algorithms.

Data Loading and Preprocessing:

The code loads an email spam dataset with 'v1' (label) and 'v2' (text) columns.

Labels are mapped from 'ham'/'spam' to binary values (0/1).

Text preprocessing includes:

Converting to lowercase

Removing URLs, HTML tags, punctuation, numbers

Removing stopwords

Stemming words to their root form using Porter Stemmer

Model Building:

Three different classification models are created using scikit-learn pipelines:

Naive Bayes: A probabilistic classifier based on Bayes' theorem

Support Vector Machine (SVM): A linear classifier that finds the optimal hyperplane

Random Forest: An ensemble of decision trees

Each pipeline includes:

TF-IDF Vectorizer: Converts text to numerical features

The respective classifier

Training and Evaluation:

All three models are trained on the same training data.

Predictions are made on the test set.

Performance metrics are calculated:

Accuracy score

Classification report (precision, recall, F1-score)

Visualization:

A bar chart compares accuracy across the three models.

Confusion matrices visualize true vs. predicted labels for each model.

For the best-performing model, feature importance is analyzed:

Top features indicating spam

Top features indicating ham (non-spam)
#########################################################

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('stopwords')

# Load the dataset
url = "https://raw.githubusercontent.com/gaurayushi/Email-Spam-Detection-/master/spam.csv"
df = pd.read_csv(url, encoding='latin-1')

# Keep only relevant columns and rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# Convert spam/ham to binary values
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Basic exploration
print(df.head())
print(df['label'].value_counts())

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42
)

# Create a pipeline with TF-IDF and Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', MultinomialNB())
])

# Create a pipeline with TF-IDF and SVM
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', LinearSVC())
])

# Create a pipeline with TF-IDF and Random Forest
rf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the models
models = {
    'Naive Bayes': nb_pipeline,
    'SVM': svm_pipeline,
    'Random Forest': rf_pipeline
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {
        'accuracy': accuracy,
        'predictions': y_pred
    }
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 80)

# Visualize results
# Accuracy comparison
plt.figure(figsize=(10, 6))
plt.bar(results.keys(), [r['accuracy'] for r in results.values()])
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.0)  # Adjust as needed
for i, (name, result) in enumerate(results.items()):
    plt.text(i, result['accuracy'] - 0.02, f"{result['accuracy']:.4f}", 
             ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.show()

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name} Confusion Matrix')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('True')
    axes[i].set_xticklabels(['Ham', 'Spam'])
    axes[i].set_yticklabels(['Ham', 'Spam'])
plt.tight_layout()
plt.show()

# Feature importance analysis (for the best model)
best_model_name = max(results, key=lambda k: results[k]['accuracy'])
best_model = models[best_model_name]

# Get feature names and coefficients (works for NB and SVM)
if best_model_name in ['Naive Bayes', 'SVM']:
    tfidf = best_model.named_steps['tfidf']
    classifier = best_model.named_steps['classifier']
    
    feature_names = tfidf.get_feature_names_out()
    
    if best_model_name == 'Naive Bayes':
        # For Naive Bayes, use log probabilities
        coefficients = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
    else:  # SVM
        coefficients = classifier.coef_[0]
    
    # Get top features for spam and ham
    top_n = 20
    top_spam_indices = np.argsort(coefficients)[-top_n:]
    top_ham_indices = np.argsort(coefficients)[:top_n]
    
    top_spam_features = [(feature_names[i], coefficients[i]) for i in top_spam_indices]
    top_ham_features = [(feature_names[i], coefficients[i]) for i in top_ham_indices]
    
    # Visualize top features
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.barh([f[0] for f in top_spam_features], [f[1] for f in top_spam_features])
    plt.title('Top Features Indicating Spam')
    plt.xlabel('Coefficient / Log Probability Ratio')
    
    plt.subplot(1, 2, 2)
    plt.barh([f[0] for f in top_ham_features], [f[1] for f in top_ham_features])
    plt.title('Top Features Indicating Ham')
    plt.xlabel('Coefficient / Log Probability Ratio')
    
    plt.tight_layout()
    plt.show()
'''

******************************************************************************************************************************************************************************************************************

4) . Genetic Cipher
This implementation creates a genetic algorithm to decrypt substitution ciphers.

#######################################################
4. Genetic Cipher
This code implements a genetic algorithm to decrypt substitution ciphers, which are ciphers where each letter is replaced by another letter.

Text Processing and Frequency Analysis:

The code loads text from Moby Dick to establish English language patterns.

Letter and bigram (two-letter sequence) frequencies are calculated.

A substitution cipher is created by randomly mapping each letter to another.

A sample text is encrypted using this cipher.

Genetic Algorithm Implementation:

The GeneticCipher class implements the genetic algorithm:

initialize_population: Creates random key mappings

decrypt: Uses a key to decrypt ciphertext

fitness: Evaluates how well a key decrypts the text by comparing:

Letter frequencies with English language frequencies

Bigram frequencies with English language frequencies

select_parents: Uses tournament selection to choose parents

crossover: Combines two parent keys to create a child key

mutate: Randomly swaps letters in a key

evolve: Runs the genetic algorithm for multiple generations

Evaluation and Visualization:

The best key from the genetic algorithm is used to decrypt the ciphertext.

Accuracy is calculated by comparing the decrypted text with the original.

Fitness history is plotted to show improvement over generations.

Letter mappings are visualized to show the decryption key.
######################################################
'''
import numpy as np
import string
import random
import matplotlib.pyplot as plt
from collections import Counter

# Load the Moby Dick text file
with open('moby_dict.txt', 'r', encoding='utf-8') as file:
    moby_text = file.read()

# Preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    return text

moby_text_clean = preprocess_text(moby_text)

# Calculate letter frequency in English text
def get_letter_frequency(text):
    # Count only alphabetic characters
    letters = [c for c in text.lower() if c.isalpha()]
    letter_count = Counter(letters)
    total = sum(letter_count.values())
    return {letter: count / total for letter, count in letter_count.items()}

english_letter_freq = get_letter_frequency(moby_text_clean)

# Calculate bigram frequency in English text
def get_bigram_frequency(text):
    text = text.lower()
    # Remove spaces and non-alphabetic characters
    text = ''.join(c for c in text if c.isalpha())
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    bigram_count = Counter(bigrams)
    total = sum(bigram_count.values())
    return {bigram: count / total for bigram, count in bigram_count.items()}

english_bigram_freq = get_bigram_frequency(moby_text_clean)

# Create a simple substitution cipher
def create_cipher():
    alphabet = string.ascii_lowercase
    shuffled = list(alphabet)
    random.shuffle(shuffled)
    cipher_dict = {alphabet[i]: shuffled[i] for i in range(len(alphabet))}
    return cipher_dict

# Encrypt a message using a substitution cipher
def encrypt(text, cipher_dict):
    text = text.lower()
    encrypted = ''
    for char in text:
        if char in cipher_dict:
            encrypted += cipher_dict[char]
        else:
            encrypted += char
    return encrypted

# Create a sample ciphertext from Moby Dick
sample_text = moby_text_clean[:1000]  # Use first 1000 characters
cipher = create_cipher()
ciphertext = encrypt(sample_text, cipher)

# Genetic Algorithm for decryption
class GeneticCipher:
    def __init__(self, ciphertext, population_size=100, mutation_rate=0.1):
        self.ciphertext = ciphertext.lower()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.alphabet = string.ascii_lowercase
        self.population = self.initialize_population()
        
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            # Create a random mapping (key)
            key = list(self.alphabet)
            random.shuffle(key)
            population.append(key)
        return population
    
    def decrypt(self, ciphertext, key):
        # Create a mapping from cipher alphabet to key
        mapping = {self.alphabet[i]: key[i] for i in range(len(self.alphabet))}
        decrypted = ''
        for char in ciphertext:
            if char in mapping:
                decrypted += mapping[char]
            else:
                decrypted += char
        return decrypted
    
    def fitness(self, key):
        # Decrypt the ciphertext using the key
        decrypted = self.decrypt(self.ciphertext, key)
        
        # Calculate letter frequency in the decrypted text
        decrypted_freq = get_letter_frequency(decrypted)
        
        # Calculate bigram frequency in the decrypted text
        decrypted_bigram_freq = get_bigram_frequency(decrypted)
        
        # Calculate fitness based on letter frequency difference
        letter_fitness = 0
        for letter in self.alphabet:
            english_freq = english_letter_freq.get(letter, 0)
            decrypted_freq_val = decrypted_freq.get(letter, 0)
            letter_fitness += abs(english_freq - decrypted_freq_val)
        
        # Calculate fitness based on bigram frequency difference
        bigram_fitness = 0
        for bigram, freq in english_bigram_freq.items():
            if bigram in decrypted_bigram_freq:
                bigram_fitness += abs(freq - decrypted_bigram_freq[bigram])
            else:
                bigram_fitness += freq
        
        # Combine the two fitness measures (lower is better)
        total_fitness = letter_fitness + 2 * bigram_fitness
        
        return 1 / (1 + total_fitness)  # Convert to a maximization problem
    
    def select_parents(self, fitnesses):
        # Tournament selection
        tournament_size = 3
        selected = []
        
        for _ in range(2):  # Select 2 parents
            tournament_indices = random.sample(range(len(fitnesses)), tournament_size)
            tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitnesses.index(max(tournament_fitnesses))]
            selected.append(self.population[winner_idx])
        
        return selected
    
    def crossover(self, parent1, parent2):
        # Order-1 Crossover for permutation
        child = [None] * len(self.alphabet)
        
        # Select a random segment from parent1
        start = random.randint(0, len(self.alphabet) - 2)
        end = random.randint(start + 1, len(self.alphabet) - 1)
        
        # Copy the segment from parent1 to child
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Fill the remaining positions with values from parent2
        parent2_idx = 0
        for i in range(len(child)):
            if child[i] is None:
                while parent2[parent2_idx] in child:
                    parent2_idx += 1
                child[i] = parent2[parent2_idx]
                parent2_idx += 1
        
        return child
    
    def mutate(self, key):
        if random.random() < self.mutation_rate:
            # Swap mutation - swap two positions
            pos1, pos2 = random.sample(range(len(key)), 2)
            key[pos1], key[pos2] = key[pos2], key[pos1]
        return key
    
    def evolve(self, generations=100):
        best_fitness_history = []
        best_key = None
        best_fitness = 0
        
        for generation in range(generations):
            # Calculate fitness for each key in the population
            fitnesses = [self.fitness(key) for key in self.population]
            
            # Track the best key
            max_fitness_idx = fitnesses.index(max(fitnesses))
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_key = self.population[max_fitness_idx]
            
            best_fitness_history.append(best_fitness)
            
            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}, Best Fitness: {best_fitness:.4f}")
                best_decryption = self.decrypt(self.ciphertext[:100], best_key)
                print(f"Sample decryption: {best_decryption}")
            
            # Create a new population
            new_population = []
            
            # Elitism - keep the best key
            new_population.append(self.population[max_fitness_idx])
            
            # Create the rest of the new population
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self.select_parents(fitnesses)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                # Add to new population
                new_population.append(child)
            
            # Replace the old population
            self.population = new_population
        
        # Return the best key and its fitness history
        return best_key, best_fitness_history
    
    def decrypt_with_best_key(self, best_key):
        return self.decrypt(self.ciphertext, best_key)

# Run the genetic algorithm
ga = GeneticCipher(ciphertext, population_size=100, mutation_rate=0.1)
best_key, fitness_history = ga.evolve(generations=100)

# Decrypt the ciphertext with the best key
decrypted_text = ga.decrypt_with_best_key(best_key)

# Print the results
print("\nOriginal text (first 100 chars):")
print(sample_text[:100])
print("\nEncrypted text (first 100 chars):")
print(ciphertext[:100])
print("\nDecrypted text (first 100 chars):")
print(decrypted_text[:100])

# Calculate accuracy
correct_chars = sum(1 for a, b in zip(sample_text, decrypted_text) if a == b)
accuracy = correct_chars / len(sample_text)
print(f"\nDecryption accuracy: {accuracy:.2%}")

# Visualize the fitness history
plt.figure(figsize=(10, 6))
plt.plot(fitness_history)
plt.title('Fitness History')
plt.xlabel('Generation')
plt.ylabel('Fitness (higher is better)')
plt.grid(True)
plt.show()

# Visualize letter mappings
original_mapping = {k: v for k, v in zip(self.alphabet, best_key)}
plt.figure(figsize=(12, 6))
plt.bar(original_mapping.keys(), [self.alphabet.index(v) for v in original_mapping.values()])
plt.title('Letter Mappings')
plt.xlabel('Ciphertext Letter')
plt.ylabel('Plaintext Letter Index')
plt.xticks(list(original_mapping.keys()))
plt.grid(True)
plt.show()
'''
*********************************************************************************************************************************************************************************************

5. Simple Substitution Cipher
This implementation creates a simple substitution cipher with frequency analysis for decryption.
####################################################
5. Simple Substitution Cipher
This code implements a simple substitution cipher with frequency analysis for decryption, which is a more straightforward approach compared to the genetic algorithm.

Cipher Implementation:

Similar to the previous example, it loads Moby Dick text and preprocesses it.

Functions are defined to:

Create a random substitution cipher

Encrypt text using the cipher

Decrypt text using the reverse mapping

Frequency Analysis Decryption:

The frequency_analysis_decrypt function:

Calculates letter frequencies in the ciphertext

Sorts both English and ciphertext frequencies

Maps ciphertext letters to English letters based on frequency rank

Decrypts the text using this mapping

Evaluation and Visualization:

The frequency analysis decryption is applied to the ciphertext.

Accuracy is calculated by comparing with the original text.

Letter frequencies are visualized:

English letter frequencies

Ciphertext letter frequencies

Cipher mappings are visualized:

Original cipher mapping

Frequency analysis mapping
###################################################
'''
import numpy as np
import string
import random
import matplotlib.pyplot as plt
from collections import Counter

# Load the Moby Dick text file
with open('moby_dict.txt', 'r', encoding='utf-8') as file:
    moby_text = file.read()

# Preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic characters
    text = ''.join(c for c in text if c.isalpha() or c.isspace())
    return text

moby_text_clean = preprocess_text(moby_text)

# Create a substitution cipher
def create_cipher():
    alphabet = string.ascii_lowercase
    shuffled = list(alphabet)
    random.shuffle(shuffled)
    cipher_dict = {alphabet[i]: shuffled[i] for i in range(len(alphabet))}
    return cipher_dict

# Encrypt a message using a substitution cipher
def encrypt(text, cipher_dict):
    text = text.lower()
    encrypted = ''
    for char in text:
        if char in cipher_dict:
            encrypted += cipher_dict[char]
        else:
            encrypted += char
    return encrypted

# Decrypt a message using a cipher key
def decrypt(text, cipher_dict):
    # Create a reverse mapping
    reverse_dict = {v: k for k, v in cipher_dict.items()}
    decrypted = ''
    for char in text:
        if char in reverse_dict:
            decrypted += reverse_dict[char]
        else:
            decrypted += char
    return decrypted

# Calculate letter frequency
def get_letter_frequency(text):
    # Count only alphabetic characters
    letters = [c for c in text.lower() if c.isalpha()]
    letter_count = Counter(letters)
    total = sum(letter_count.values())
    return {letter: count / total for letter, count in letter_count.items()}

# English letter frequency (from Moby Dick)
english_letter_freq = get_letter_frequency(moby_text_clean)

# Create a sample ciphertext from Moby Dick
sample_text = moby_text_clean[:1000]  # Use first 1000 characters
cipher = create_cipher()
ciphertext = encrypt(sample_text, cipher)

# Frequency analysis for decryption
def frequency_analysis_decrypt(ciphertext):
    # Calculate letter frequency in the ciphertext
    cipher_freq = get_letter_frequency(ciphertext)
    
    # Sort both frequency dictionaries by frequency (descending)
    sorted_english_freq = sorted(english_letter_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_cipher_freq = sorted(cipher_freq.items(), key=lambda x: x[1], reverse=True)
    
    # Create a mapping based on frequency
    mapping = {}
    for (cipher_letter, _), (english_letter, _) in zip(sorted_cipher_freq, sorted_english_freq):
        mapping[cipher_letter] = english_letter
    
    # Decrypt using this mapping
    decrypted = ''
    for char in ciphertext:
        if char in mapping:
            decrypted += mapping[char]
        else:
            decrypted += char
    
    return decrypted, mapping

# Perform frequency analysis decryption
freq_decrypted, freq_mapping = frequency_analysis_decrypt(ciphertext)

# Print the results
print("Original text (first 100 chars):")
print(sample_text[:100])
print("\nEncrypted text (first 100 chars):")
print(ciphertext[:100])
print("\nFrequency analysis decryption (first 100 chars):")
print(freq_decrypted[:100])

# Calculate accuracy of frequency analysis
correct_chars = sum(1 for a, b in zip(sample_text, freq_decrypted) if a == b)
accuracy = correct_chars / len(sample_text)
print(f"\nFrequency analysis decryption accuracy: {accuracy:.2%}")

# Visualize letter frequencies
plt.figure(figsize=(12, 6))

# English letter frequencies
plt.subplot(1, 2, 1)
english_letters = sorted(english_letter_freq.keys())
english_freqs = [english_letter_freq[letter] for letter in english_letters]
plt.bar(english_letters, english_freqs)
plt.title('English Letter Frequencies')
plt.xlabel('Letter')
plt.ylabel('Frequency')

# Ciphertext letter frequencies
plt.subplot(1, 2, 2)
cipher_freq = get_letter_frequency(ciphertext)
cipher_letters = sorted(cipher_freq.keys())
cipher_freqs = [cipher_freq[letter] for letter in cipher_letters]
plt.bar(cipher_letters, cipher_freqs)
plt.title('Ciphertext Letter Frequencies')
plt.xlabel('Letter')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Visualize the mapping
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# Original cipher mapping
original_mapping = {v: k for k, v in cipher.items()}  # Reverse to show cipher -> plain
plt.bar(original_mapping.keys(), [ord(v) - ord('a') for v in original_mapping.values()])
plt.title('Original Cipher Mapping')
plt.xlabel('Ciphertext Letter')
plt.ylabel('Plaintext Letter Index (a=0, b=1, ...)')
plt.xticks(list(original_mapping.keys()))

plt.subplot(1, 2, 2)
# Frequency analysis mapping
plt.bar(freq_mapping.keys(), [ord(v) - ord('a') for v in freq_mapping.values()])
plt.title('Frequency Analysis Mapping')
plt.xlabel('Ciphertext Letter')
plt.ylabel('Plaintext Letter Index (a=0, b=1, ...)')
plt.xticks(list(freq_mapping.keys()))

plt.tight_layout()
plt.show()

'''

*********************************************************************************************************************************************************************************************
6. Article Spinner
This implementation creates an article spinner that generates variations of text by replacing words with synonyms.

###########################################################################
6. Article Spinner
This code creates an article spinner that generates variations of text by replacing words with synonyms, which is useful for creating alternative versions of content.

Data Loading and Preprocessing:

BBC news articles are loaded and preprocessed.

Text preprocessing removes special characters, digits, and extra whitespace.

Synonym Generation:

get_synonyms: Finds synonyms for a word based on its part-of-speech tag.

Maps NLTK POS tags to WordNet POS tags

Retrieves synsets (sets of synonyms) from WordNet

Extracts lemma names as synonyms

Article Spinning:

spin_article: Generates a variation of the input text.

Tokenizes text into sentences and words

Tags words with part-of-speech

Replaces content words (nouns, adjectives, verbs, adverbs) with synonyms

The replacement probability controls how many words are replaced

Reconstructs sentences and the full article

Similarity Calculation:

calculate_similarity: Measures how similar the spun text is to the original.

Uses TF-IDF vectorization and cosine similarity

Evaluation and Visualization:

Articles are spun with different replacement probabilities (0.1, 0.3, 0.5).

Similarity scores are calculated between original and spun texts.

Visualizations include:

Box plot of similarity by replacement probability

Box plot of similarity by article category

Scatter plot of word replacement statistics
###########################################################################
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import random
import re
import string
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Load the dataset
url = "https://raw.githubusercontent.com/codehax41/BBC-Text-Classification/master/bbc-text.csv"
df = pd.read_csv(url)

print(df.head())
print(df['category'].value_counts())

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Function to get synonyms for a word based on its POS tag
def get_synonyms(word, pos=None):
    synonyms = []
    
    # Map NLTK POS tags to WordNet POS tags
    pos_map = {
        'NN': wordnet.NOUN,
        'NNS': wordnet.NOUN,
        'NNP': wordnet.NOUN,
        'NNPS': wordnet.NOUN,
        'JJ': wordnet.ADJ,
        'JJR': wordnet.ADJ,
        'JJS': wordnet.ADJ,
        'RB': wordnet.ADV,
        'RBR': wordnet.ADV,
        'RBS': wordnet.ADV,
        'VB': wordnet.VERB,
        'VBD': wordnet.VERB,
        'VBG': wordnet.VERB,
        'VBN': wordnet.VERB,
        'VBP': wordnet.VERB,
        'VBZ': wordnet.VERB
    }
    
    wordnet_pos = pos_map.get(pos) if pos else None
    
    # Get synsets for the word
    if wordnet_pos:
        synsets = wordnet.synsets(word, pos=wordnet_pos)
    else:
        synsets = wordnet.synsets(word)
    
    # Get lemma names from synsets
    for synset in synsets:
        for lemma in synset.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)
    
    return synonyms

# Function to spin an article
def spin_article(text, replacement_probability=0.3):
    sentences = sent_tokenize(text)
    spun_sentences = []
    
    for sentence in sentences:
        # Tokenize and tag words with POS
        words = word_tokenize(sentence)
        tagged_words = pos_tag(words)
        
        spun_words = []
        for word, pos in tagged_words:
            # Only try to replace content words with some probability
            if (pos.startswith('NN') or pos.startswith('JJ') or 
                pos.startswith('VB') or pos.startswith('RB')) and random.random() < replacement_probability:
                
                synonyms = get_synonyms(word, pos)
                
                # If synonyms are found, randomly select one
                if synonyms:
                    replacement = random.choice(synonyms)
                    spun_words.append(replacement)
                else:
                    spun_words.append(word)
            else:
                spun_words.append(word)
        
        # Reconstruct the sentence
        spun_sentence = ' '.join(spun_words)
        spun_sentences.append(spun_sentence)
    
    # Reconstruct the article
    spun_article = ' '.join(spun_sentences)
    
    return spun_article

# Function to calculate similarity between original and spun text
def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors)[0][1]

# Apply the spinner to a sample of articles
sample_size = min(10, len(df))
df_sample = df.sample(sample_size, random_state=42)

# Generate spun versions with different probabilities
replacement_probs = [0.1, 0.3, 0.5]
results = []

for idx, row in df_sample.iterrows():
    original_text = row['text']
    category = row['category']
    
    for prob in replacement_probs:
        spun_text = spin_article(original_text, replacement_probability=prob)
        similarity = calculate_similarity(original_text, spun_text)
        
        results.append({
            'category': category,
            'replacement_prob': prob,
            'original_length': len(original_text),
            'spun_length': len(spun_text),
            'similarity': similarity,
            'original_text': original_text[:200] + '...',  # First 200 chars
            'spun_text': spun_text[:200] + '...'  # First 200 chars
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print sample results
for i, row in results_df.sample(3).iterrows():
    print(f"Category: {row['category']}")
    print(f"Replacement Probability: {row['replacement_prob']}")
    print(f"Similarity: {row['similarity']:.4f}")
    print("Original Text:")
    print(row['original_text'])
    print("Spun Text:")
    print(row['spun_text'])
    print("-" * 80)

# Visualize similarity by replacement probability
plt.figure(figsize=(10, 6))
sns.boxplot(x='replacement_prob', y='similarity', data=results_df)
plt.title('Text Similarity by Replacement Probability')
plt.xlabel('Replacement Probability')
plt.ylabel('Cosine Similarity')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Visualize similarity by category
plt.figure(figsize=(12, 6))
sns.boxplot(x='category', y='similarity', hue='replacement_prob', data=results_df)
plt.title('Text Similarity by Category and Replacement Probability')
plt.xlabel('Category')
plt.ylabel('Cosine Similarity')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Replacement Probability')
plt.tight_layout()
plt.show()

# Visualize word replacement statistics
# Count replaced words
def count_replaced_words(original, spun):
    original_words = set(word_tokenize(original))
    spun_words = set(word_tokenize(spun))
    
    # Words in original but not in spun (potentially replaced)
    replaced = original_words - spun_words
    # Words in spun but not in original (replacements)
    new_words = spun_words - original_words
    
    return len(replaced), len(new_words)

# Calculate replacement statistics for each article
replacement_stats = []
for i, row in results_df.iterrows():
    replaced, new = count_replaced_words(row['original_text'], row['spun_text'])
    replacement_stats.append({
        'category': row['category'],
        'replacement_prob': row['replacement_prob'],
        'replaced_words': replaced,
        'new_words': new
    })

replacement_stats_df = pd.DataFrame(replacement_stats)

# Visualize replacement statistics
plt.figure(figsize=(10, 6))
sns.scatterplot(x='replaced_words', y='new_words', 
                hue='replacement_prob', style='category', 
                data=replacement_stats_df, s=100, alpha=0.7)
plt.title('Word Replacement Statistics')
plt.xlabel('Number of Words Replaced')
plt.ylabel('Number of New Words Added')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Replacement Probability')
plt.tight_layout()
plt.show()

'''

************************************************************************************************************************************************************************************************************

7. Poetry Generator
This implementation creates a poetry generator based on Robert Frost's poems.

#####################################################################
7. Poetry Generator
This code creates a poetry generator based on Robert Frost's poems using a recurrent neural network.

Data Preparation:

Robert Frost's poetry is loaded and preprocessed.

Sequences of words are created for training:

Each sequence consists of 5 consecutive words as input

The next word is the target output

Model Building:

A sequential neural network is created with:

Embedding layer: Converts word indices to dense vectors

LSTM layers: Capture sequential patterns in text

Dropout layers: Prevent overfitting

Dense output layer: Predicts the next word

Poetry Generation:

generate_poem: Creates new poetry based on a seed text.

Takes a starting phrase and predicts the next word

Adds the predicted word to the poem

Updates the seed text with the new word

Repeats the process for a specified number of words

Temperature parameter controls randomness in word selection

Evaluation and Visualization:

Poems are generated with different temperature settings and seed texts.

Training history (accuracy and loss) is visualized.

Word frequencies and line lengths are analyzed:

Comparison of top words in original vs. generated poems

Distribution of line lengths in original vs. generated poems
######################################################################
'''
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import re
import random
import string

# Load Robert Frost's poetry
with open('robert_frost.txt', 'r', encoding='utf-8') as file:
    frost_text = file.read()

# Preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace newlines with spaces
    text = re.sub(r'\n', ' \n ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

frost_text_clean = preprocess_text(frost_text)

# Create sequences of words
def create_sequences(text, seq_length):
    words = text.split()
    sequences = []
    for i in range(len(words) - seq_length):
        seq = words[i:i + seq_length]
        target = words[i + seq_length]
        sequences.append((seq, target))
    return sequences

# Parameters
seq_length = 5
sequences = create_sequences(frost_text_clean, seq_length)

# Prepare the data
input_sequences = [seq[0] for seq in sequences]
targets = [seq[1] for seq in sequences]

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([frost_text_clean])
total_words = len(tokenizer.word_index) + 1

# Convert sequences to numeric form
X = tokenizer.texts_to_sequences(input_sequences)
X = np.array(X)
y = tokenizer.texts_to_sequences(targets)
y = np.array(y).reshape(-1, 1)

# One-hot encode the target words
y_one_hot = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Build the model
model = Sequential([
    Embedding(total_words, 100, input_length=seq_length),
    LSTM(150, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.01),
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(
    X, y_one_hot,
    epochs=100,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Function to generate poetry
def generate_poem(seed_text, next_words, model, tokenizer, seq_length, temperature=1.0):
    poem = seed_text
    seed_text = seed_text.lower()
    
    for _ in range(next_words):
        # Tokenize the seed text
        token_list = tokenizer.texts_to_sequences([seed_text.split()[-seq_length:]])
        
        # If the sequence is shorter than seq_length, pad it
        if len(token_list[0]) < seq_length:
            token_list = pad_sequences(token_list, maxlen=seq_length)
        
        # Predict the next word
        predicted = model.predict(token_list, verbose=0)[0]
        
        # Apply temperature to control randomness
        predicted = np.log(predicted) / temperature
        exp_predicted = np.exp(predicted)
        predicted = exp_predicted / np.sum(exp_predicted)
        
        # Sample from the distribution
        probabilities = np.random.multinomial(1, predicted, 1)[0]
        next_index = np.argmax(probabilities)
        
        # Convert the index to a word
        next_word = ""
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                next_word = word
                break
        
        # Add the next word to the poem
        if next_word == '\n':
            poem += '\n'
        else:
            poem += ' ' + next_word
        
        # Update the seed text
        seed_text = ' '.join(seed_text.split()[-seq_length+1:] + [next_word])
    
    return poem

# Generate poems with different temperatures
temperatures = [0.5, 0.7, 1.0, 1.2]
seed_texts = [
    "the road not taken",
    "stopping by woods on",
    "two roads diverged in",
    "the woods are lovely"
]

generated_poems = {}

for temp in temperatures:
    poems_at_temp = []
    for seed in seed_texts:
        poem = generate_poem(seed, 50, model, tokenizer, seq_length, temperature=temp)
        poems_at_temp.append(poem)
    generated_poems[temp] = poems_at_temp

# Print sample poems
for temp, poems in generated_poems.items():
    print(f"Temperature: {temp}")
    print("-" * 40)
    print(poems[0])  # Print the first poem at this temperature
    print("=" * 80)

# Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# Analyze word frequencies in original and generated poems
def get_word_frequencies(text):
    words = text.lower().split()
    word_freq = {}
    for word in words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    return word_freq

# Get word frequencies from original text
original_freq = get_word_frequencies(frost_text_clean)

# Get word frequencies from generated poems
generated_text = ' '.join([poem for poems in generated_poems.values() for poem in poems])
generated_freq = get_word_frequencies(generated_text)

# Get top words
def get_top_words(word_freq, n=20):
    return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:n]

top_original = get_top_words(original_freq)
top_generated = get_top_words(generated_freq)

# Visualize top words comparison
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.bar([word for word, freq in top_original], [freq for word, freq in top_original])
plt.title('Top Words in Original Poems')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.subplot(1, 2, 2)
plt.bar([word for word, freq in top_generated], [freq for word, freq in top_generated])
plt.title('Top Words in Generated Poems')
plt.xlabel('Word')
plt.ylabel('Frequency')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Analyze line lengths
def get_line_lengths(text):
    lines = text.split('\n')
    return [len(line.split()) for line in lines if line.strip()]

original_line_lengths = get_line_lengths(frost_text_clean)
generated_line_lengths = get_line_lengths(generated_text)

# Visualize line length distributions
plt.figure(figsize=(12, 6))
plt.hist(original_line_lengths, bins=15, alpha=0.5, label='Original Poems')
plt.hist(generated_line_lengths, bins=15, alpha=0.5, label='Generated Poems')
plt.title('Line Length Distribution')
plt.xlabel('Words per Line')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
'''

*****************************************************************************************************************************************************************************************************

8. Markov Classifier
This implementation creates a Hidden Markov Model to classify poetry by author.

#########################################################################
8. Markov Classifier
This code creates a Hidden Markov Model to classify poetry by author, specifically distinguishing between Robert Frost and Edgar Allan Poe.

Data Loading and Preprocessing:

Poetry from Robert Frost and Edgar Allan Poe is loaded.

Text is preprocessed by converting to lowercase, removing punctuation, and extra whitespace.

Poems are split into individual poems based on blank lines.

Feature Extraction:

extract_features: Extracts various features from each poem:

Average word length

Percentage of common words

Percentage of unique words

Average line length

Percentage of lines ending with common words

Vowel-to-consonant ratio

Percentage of words longer than 6 characters

These features are used to train a Hidden Markov Model that can classify poems by their author based on these statistical patterns.
#########################################################################
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from hmmlearn import hmm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import string
from collections import Counter

# Load the poetry files
def load_poetry(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Load Robert Frost and Edgar Allan Poe poems
frost_text = load_poetry('robert_frost.txt')
poe_text = load_poetry('edgar_allan_poem.txt')

# Preprocess the text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

frost_text_clean = preprocess_text(frost_text)
poe_text_clean = preprocess_text(poe_text)

# Split poems into individual poems
def split_into_poems(text):
    # This is a simple approach - assuming poems are separated by blank lines
    poems = re.split(r'\n\s*\n', text)
    return [poem.strip() for poem in poems if poem.strip()]

frost_poems = split_into_poems(frost_text)
poe_poems = split_into_poems(poe_text)

print(f"Number of Frost poems: {len(frost_poems)}")
print(f"Number of Poe poems: {len(poe_poems)}")

# Feature extraction functions
def extract_features(poem):
    # Preprocess the poem
    poem = preprocess_text(poem)
    
    # Extract various features
    features = []
    
    # 1. Average word length
    words = poem.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    features.append(avg_word_length)
    
    # 2. Percentage of common words (the, and, of, etc.)
    common_words = {'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'he', 'for', 'it', 'with', 'as', 'his', 'on', 'be', 'at', 'by', 'i', 'this', 'had', 'not', 'are', 'but', 'from', 'or', 'have', 'an', 'they', 'which', 'one', 'you', 'were', 'her', 'all', 'she', 'there', 'would', 'their', 'we', 'him', 'been', 'has', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'said', 'what', 'up', 'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'my', 'now', 'such', 'like', 'our', 'over', 'man', 'me', 'even', 'most', 'made', 'after', 'also', 'did', 'many', 'before', 'must', 'through', 'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down', 'should', 'because', 'each', 'just', 'those', 'people', 'mr', 'how', 'too', 'little', 'state', 'good', 'very', 'make', 'world', 'still', 'own', 'see', 'men', 'work', 'long', 'get', 'here', 'between', 'both', 'life', 'being', 'under', 'never', 'day', 'same', 'another', 'know', 'while', 'last', 'might', 'us', 'great', 'old', 'year', 'off', 'come', 'since', 'against', 'go', 'came', 'right', 'used', 'take', 'three'}
    common_word_count = sum(1 for word in words if word in common_words)
    common_word_pct = common_word_count / len(words) if words else 0
    features.append(common_word_pct)
    
    # 3. Percentage of unique words
    unique_words = set(words)
    unique_word_pct = len(unique_words) / len(words) if words else 0
    features.append(unique_word_pct)
    
    # 4. Average line length
    lines = poem.split('\n')
    line_lengths = [len(line.split()) for line in lines if line.strip()]
    avg_line_length = sum(line_lengths) / len(line_lengths) if line_lengths else 0
    features.append(avg_line_length)
    
    # 5. Percentage of lines that end with common ending words
    common_endings = {'the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'with', 'as', 'on', 'at', 'by', 'this', 'but', 'from', 'or', 'an', 'they', 'which', 'one', 'you', 'all', 'there', 'their', 'we', 'when', 'who', 'will', 'more', 'no', 'if', 'out', 'so', 'what', 'up', 'its', 'about', 'into', 'than', 'them', 'can', 'only', 'other', 'new', 'some', 'could', 'time', 'these', 'two', 'may', 'then', 'do', 'first', 'any', 'my', 'now', 'such', 'like', 'our', 'over', 'man', 'me', 'even', 'most', 'after', 'also', 'many', 'before', 'must', 'through', 'back', 'years', 'where', 'much', 'your', 'way', 'well', 'down', 'should', 'because', 'each', 'just', 'those', 'people', 'how', 'too', 'little', 'state', 'good', 'very', 'make', 'world', 'still', 'own', 'see', 'men', 'work', 'long', 'get', 'here', 'between', 'both', 'life', 'being', 'under', 'never', 'day', 'same', 'another', 'know', 'while', 'last', 'might', 'us', 'great', 'old', 'year', 'off', 'come', 'since', 'against', 'go', 'came', 'right', 'used', 'take', 'three'}
    ending_words = [line.split()[-1] if line.split() else '' for line in lines]
    common_ending_count = sum(1 for word in ending_words if word in common_endings)
    common_ending_pct = common_ending_count / len(ending_words) if ending_words else 0
    features.append(common_ending_pct)
    
    # 6. Vowel-to-consonant ratio
    vowels = 'aeiou'
    vowel_count = sum(1 for char in poem if char.lower() in vowels)
    consonant_count = sum(1 for char in poem if char.lower() in string.ascii_lowercase and char.lower() not in vowels)
    vowel_consonant_ratio = vowel_count / consonant_count if consonant_count else 0
    features.append(vowel_consonant_ratio)
    
    # 7. Percentage of words that are longer than 6 characters
    long_word_count = sum(1 for word in words if len(word) > 6)
    long_word_pct = long_word_count / len(words) if words else
'''
