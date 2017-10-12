
# coding: utf-8

# # Sentiment Prediction RNN

# In[1]:


import numpy as np
import tensorflow as tf

from string import punctuation
from collections import Counter


# ## DataPreprocessor

# In[2]:


class DataPreprocessor:
    
    def load_and_preprocess_data(self, 
                                 reviews_file='reviews.txt', 
                                 labels_file='labels.txt', 
                                 seq_len=200):
        data = Data()
        
        dataLoader = DataLoader()
        data.reviews, data.labels = dataLoader.load_data(reviews_file, labels_file)
        
        dataExtractor = DataExtractor()
        data.reviews = dataExtractor.remove_punctuation(data.reviews)
        data.words = dataExtractor.extract_words(data.reviews)
        
        dataEncoder = DataEncoder()
        data.words_to_ints = dataEncoder.map_words_to_ints(data.words)
        data.reviews_to_ints = dataEncoder.map_reviews_to_ints(data.reviews, data.words_to_ints)
        data.labels_to_ints = dataEncoder.map_labels_to_ints(data.labels)
        
        dataFilterer = DataFilterer()
        data.reviews_to_ints, data.labels_to_ints = dataFilterer.filter_out_zero_len_entries(
            data.reviews_to_ints, data.labels_to_ints)
        
        dataFormatter = DataFormatter()
        data.features = dataFormatter.format_features(data.reviews_to_ints, seq_len)
        data.targets = data.labels_to_ints
        
        return data


# ### Data

# In[3]:


class Data:
    
    def __init__(self):
        self.reviews = None
        self.labels = None
        self.words = None
        self.words_to_ints = None
        self.reviews_to_ints = None
        self.labels_to_ints = None
        self.features = None
        self.targets = None


# ### DataLoader

# In[4]:


class DataLoader:
    
    def load_data(self, reviews_file, labels_file):
        with open(reviews_file, 'r') as f:
            reviews = f.read()
        with open(labels_file, 'r') as f:
            labels = f.read()
        self.log_data(reviews, labels)
        return reviews, labels
    
    def log_data(self, reviews, labels):
        print("Loaded data\n")
        print("reviews[:100]:\n{}\n".format(reviews[:100]))
        print("labels[:100]:\n{}\n\n".format(labels[:100]))


# ### DataExtractor

# In[5]:


class DataExtractor:

    def remove_punctuation(self, review_text):
        all_text = ''.join([c for c in review_text if c not in punctuation])
        reviews = all_text.split('\n')
        self.log_reviews(reviews)
        return reviews
    
    def extract_words(self, reviews):
        all_text = ' '.join(reviews)
        words = all_text.split()
        self.log_words(words)
        return words
    
    def log_reviews(self, reviews):
        print("Removed punctuation from review text\n")
        print("reviews[:1]:\n{}\n\n".format(reviews[:1]))
    
    def log_words(self, words):
        print("Extracted words\n")
        print("words[:100]:\n{}\n\n".format(words[:100]))


# ### DataEncoder

# In[6]:


class DataEncoder:
    
    def map_words_to_ints(self, words):
        counts = Counter(words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        words_to_ints = {word: ii for ii, word in enumerate(vocab, 1)}
        self.log_words_to_ints(words, words_to_ints)
        return words_to_ints
    
    def map_reviews_to_ints(self, reviews, words_to_ints):
        reviews_to_ints = []
        for review in reviews:
            reviews_to_ints.append([words_to_ints[word] for word in review.split()])
        self.log_reviews_to_ints(reviews_to_ints)
        return reviews_to_ints
    
    def map_labels_to_ints(self, labels):
        labels = labels.split('\n')
        labels_to_ints = np.array([1 if each == 'positive' else 0 for each in labels])
        self.log_labels_to_ints(labels_to_ints)
        return labels_to_ints
  
    def log_words_to_ints(self, words, words_to_ints):
        print("Mapped words to ints\n")
        for word in words[:10]:
            print('{}: {}'.format(word, words_to_ints[word]))
        print("\n")    
            
    def log_reviews_to_ints(self, reviews_to_ints):
        print("Mapped reviews to ints\n")
        print("reviews_to_ints[0]:\n{}\n\n".format(reviews_to_ints[0]))
        
    def log_labels_to_ints(self, labels_to_ints):
        print("Mapped labels to ints\n")
        print("labels_to_ints[:10]:\n{}\n\n".format(labels_to_ints[:10]))


# ### DataFilterer

# In[8]:


class DataFilterer:
    
    def filter_out_zero_len_entries(self, reviews_to_ints, labels_to_ints):
        non_zero_len_indices =             [ii for ii, review in enumerate(reviews_to_ints) if len(review) > 0]
        reviews_to_ints = [reviews_to_ints[ii] for ii in non_zero_len_indices]
        labels_to_ints = np.array([labels_to_ints[ii] for ii in non_zero_len_indices])
        self.log()
        return reviews_to_ints, labels_to_ints
    
    def log(self):
        print("Filtered out zero-len entries\n\n")


# ### DataFormatter

# In[9]:


class DataFormatter:
    
    def format_features(self, reviews_to_ints, seq_len):
        """
        truncate each review to seq_len
        add zero-padding to the left if review_len < seq_len
        """
        
        features = np.zeros((len(reviews_to_ints), seq_len), dtype=int)
        
        #for i, row in enumerate(reviews_ints):
        #    features[i, -len(row):] = np.array(row)[:seq_len]
        
        for i, review in enumerate(reviews_to_ints):
            review_len = len(review)
            
            if review_len >= seq_len:
                features[i] = review[:seq_len]
            else:
                offset = seq_len - review_len
                features[i,:offset] = 0
                features[i,offset:] = review
        
        self.log_features(features)
        
        return features
    
    def log_features(self, features):
        print("Formatted features\n")
        print("features[:5,:100]:\n{}\n\n".format(features[:5,:100]))


# ## DataSetCreator

# In[10]:


class DataSetCreator:
    
    def create_training_validation_and_testing_sets(self, features, targets, split_frac=0.8):
        split_idx = int(len(features)*split_frac)
        
        train_x, val_x = features[:split_idx], features[split_idx:]
        train_y, val_y = targets[:split_idx], targets[split_idx:]
        
        test_idx = int(len(val_x)*0.5)
        
        val_x, test_x = val_x[:test_idx], val_x[test_idx:]
        val_y, test_y = val_y[:test_idx], val_y[test_idx:]
        
        self.log_feature_shapes(train_x, val_x, test_x)
        
        return DataSets(train_x, train_y, val_x, val_y, test_x, test_y)

    def log_feature_shapes(self, train_x, val_x, test_x):
        print("Created training, validation, and testing sets\n")
        print("Training set features shape: \t{}".format(train_x.shape))
        print("Validation set features shape: \t{}".format(val_x.shape))
        print("Testing set features shape: \t{}\n".format(test_x.shape))


# ### DataSets

# In[11]:


class DataSets:
    
    def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y


# ## RNNetwork

# In[12]:


class RNNetwork:
    
    def __init__(self):
        self.graph = tf.Graph()
        self.log_graph()
    
    def create_placeholders(self):
        with self.graph.as_default():
            self.inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
            self.labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.log_placeholders()
    
    def add_embedding_layer(self, n_words, embed_size=300):
        with self.graph.as_default():
            self.embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
            self.embed = tf.nn.embedding_lookup(self.embedding, self.inputs_)
            self.log_embedding_layer()

    def add_lstm_layers(self, lstm_size=256, lstm_layers=1, batch_size=500):
        with self.graph.as_default():
            self.lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
            self.drop = tf.contrib.rnn.DropoutWrapper(self.lstm, output_keep_prob=self.keep_prob)
            self.cell = tf.contrib.rnn.MultiRNNCell([self.drop]*lstm_layers)
            self.init_state = self.cell.zero_state(batch_size, tf.float32)
            self.log_lstm_layers()
    
    def add_forward_pass(self):
        with self.graph.as_default():
            self.outputs, self.final_state = tf.nn.dynamic_rnn(self.cell, 
                                                               self.embed, 
                                                               initial_state=self.init_state)
        self.log_forward_pass()
        
    def add_train_loss_computation(self, learning_rate=0.001):
        with self.graph.as_default():
            self.predictions = tf.contrib.layers.fully_connected(self.outputs[:, -1], 
                                                                 1, 
                                                                 activation_fn=tf.sigmoid)
            self.cost = tf.losses.mean_squared_error(self.labels_, self.predictions)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        self.log_train_loss_computation() 
    
    def add_validation_accuracy_computation(self):
        with self.graph.as_default():
            self.correct_pred = tf.equal(tf.cast(tf.round(self.predictions), tf.int32), 
                                         self.labels_)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        self.log_validation_accuracy_computation()    
    
    def log_graph(self):
        print("Created graph object\n")
    
    def log_placeholders(self):
        print("Created placeholders\n")
        print("inputs_: {}".format(self.inputs_))
        print("labels_: {}".format(self.labels_))
        print("keep_prob_: {}\n".format(self.keep_prob))
        
    def log_embedding_layer(self):
        print("Added embedding layer\n")
        
    def log_lstm_layers(self):
        print("Added LSTM layers\n")
        
    def log_forward_pass(self):
        print("Added forward pass\n")
        
    def log_train_loss_computation(self):
        print("Added training loss computation\n")

    def log_validation_accuracy_computation(self):
        print("Added validation accuracy computation\n")


# ## NetworkTrainer

# In[13]:


class NetworkTrainer:
    
    def __init__(self, network, datasets, epochs=10, batch_size=500):
        self.network = network
        self.datasets = datasets
        self.epochs = epochs
        self.batch_size = batch_size
        
    def train_network(self):
        with self.network.graph.as_default():
            saver = tf.train.Saver()
            
        with tf.Session(graph=self.network.graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            
            for e in range(self.epochs):
                state = sess.run(self.network.init_state)
                
                for ii, (x, y) in enumerate(self.get_batches(
                    self.datasets.train_x, self.datasets.train_y, self.batch_size), 1):
                    
                    feed = {self.network.inputs_: x,
                            self.network.labels_: y[:, None],
                            self.network.keep_prob: 0.5,
                            self.network.init_state: state}
                    
                    loss, state, _ = sess.run([self.network.cost, 
                                               self.network.final_state, 
                                               self.network.optimizer],
                                               feed_dict=feed)
                    
                    if iteration%5==0:
                        print("Epoch: {}/{}".format(e+1, self.epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration%25==0:
                        val_acc = []
                        val_state = sess.run(self.network.cell.zero_state(self.batch_size, 
                                                                          tf.float32))
                        
                        for x, y in self.get_batches(self.datasets.val_x, 
                                                     self.datasets.val_y, 
                                                     self.batch_size):
                            feed = {self.network.inputs_: x,
                                    self.network.labels_: y[:, None],
                                    self.network.keep_prob: 1,
                                    self.network.init_state: val_state}
                            
                            batch_acc, val_state = sess.run([self.network.accuracy, 
                                                             self.network.final_state], 
                                                            feed_dict=feed)
                            val_acc.append(batch_acc)
                            
                        print("Val acc: {:.3f}".format(np.mean(val_acc)))

                    iteration +=1
            
            saver.save(sess, "checkpoints/sentiment.ckpt")
            
        return saver
            
    def get_batches(self, x, y, batch_size=100):
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]        


# ## NetworkTester

# In[14]:


class NetworkTester:
    
    def __init__(self, network, datasets, batch_size=500):
        self.network = network
        self.datasets = datasets
        self.batch_size = batch_size
        self.test_acc = []
        
    def test_network(self, saver):
        with tf.Session(graph=self.network.graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            test_state = sess.run(self.network.cell.zero_state(self.batch_size, tf.float32))
            
            for ii, (x, y) in enumerate(self.get_batches(
                self.datasets.test_x, self.datasets.test_y, self.batch_size), 1):
                
                feed = {self.network.inputs_: x,
                        self.network.labels_: y[:, None],
                        self.network.keep_prob: 1,
                        self.network.init_state: test_state}
                
                batch_acc, test_state = sess.run([self.network.accuracy, 
                                                  self.network.final_state], 
                                                 feed_dict=feed)
                self.test_acc.append(batch_acc)
                
            print("Test accuracy: {:.3f}".format(np.mean(self.test_acc)))

    def get_batches(self, x, y, batch_size=100):
        n_batches = len(x)//batch_size
        x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]

        for ii in range(0, len(x), batch_size):
            yield x[ii:ii+batch_size], y[ii:ii+batch_size]


# # SentimentRNN

# In[15]:


class SentimentRNN:
    
    def load_and_prepare_data(self):
        print('DATA PREPROCESSING STEP\n')
        dataPreprocessor = DataPreprocessor()
        self.data = dataPreprocessor.load_and_preprocess_data()
        print('DATA PROCESSING COMPLETE\n')
        
    def create_train_val_test_sets(self):
        print('DATA SET CREATION STEP\n')
        dataSetCreator = DataSetCreator()
        self.datasets = dataSetCreator.create_training_validation_and_testing_sets(
            self.data.features, self.data.targets)
        print('DATA SET CREATION COMPLETE\n')
    
    def build_network(self):
        print('NETWORK BUILDING STEP\n')
        self.network = RNNetwork()
        self.network.create_placeholders()
        self.network.add_embedding_layer(len(self.data.words_to_ints)+1)
        self.network.add_lstm_layers()
        self.network.add_forward_pass()
        self.network.add_train_loss_computation()
        self.network.add_validation_accuracy_computation()
        print('NETWORK BUILDING COMPLETE\n')
        
    def train_network(self):
        print('TRAINING STEP\n')
        networkTrainer = NetworkTrainer(self.network, self.datasets)
        self.saver = networkTrainer.train_network()
        print('\nTRAINING STEP COMPLETE\n')
        
    def test_network(self):
        print('TESTING STEP\n')
        networkTester = NetworkTester(self.network, self.datasets)
        networkTester.test_network(self.saver)
        print('\nTESTING COMPLETE\n')


# In[16]:


rnn = SentimentRNN()


# In[17]:


rnn.load_and_prepare_data()


# In[18]:


rnn.create_train_val_test_sets()


# In[19]:


rnn.build_network()


# In[24]:


rnn.train_network()


# In[25]:


rnn.test_network()

