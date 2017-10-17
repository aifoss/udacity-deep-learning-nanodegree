
# coding: utf-8

# # TV Script Generation RNN

# In[1]:


import os
import pickle

from collections import Counter

from distutils.version import LooseVersion
import warnings

import numpy as np
import tensorflow as tf

from tensorflow.contrib import seq2seq


# ## Loading Data

# In[2]:


class DataLoader:
    
    def load_data(self, path):
        """
        Load Dataset from File
        """
 
        input_file = os.path.join(path)
        with open(input_file, "r") as f:
            data = f.read()
            
        return data


# In[3]:


data_dir = './data/simpsons/moes_tavern_lines.txt'


# In[4]:


dataLoader = DataLoader()

text = dataLoader.load_data(data_dir)

# Ignore notice, since we don't use it for analysing the data
text = text[81:]


# ## Exploring Data

# In[5]:


class DataExplorer:
    
    def explore_data(self, text, view_sentence_range):
        """
        Explore input data text.
        
        :param text: Input text to explore
        :param view_sentence_range: Range of sentences to display
        """
        
        print('Dataset Stats')
        print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
        scenes = text.split('\n\n')
        print('Number of scenes: {}'.format(len(scenes)))
        sentence_count_scene = [scene.count('\n') for scene in scenes]
        print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

        sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
        print('Number of lines: {}'.format(len(sentences)))
        word_count_sentence = [len(sentence.split()) for sentence in sentences]
        print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

        print()
        print('The sentences {} to {}:'.format(*view_sentence_range))
        print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


# In[6]:


view_sentence_range = (0, 10)


# In[7]:


dataExplorer = DataExplorer()

dataExplorer.explore_data(text, view_sentence_range)


# ## Preprocessing Data

# In[8]:


class DataPreprocessor:
    
    def preprocess_and_save_data(self, text):
        """
        Preprocess and save text data
        
        :param text: The text of tv scripts split into words
        """

        punc_dict = self.create_punc_lookup_table()
        
        for key, token in punc_dict.items():
            text = text.replace(key, ' {} '.format(token))

        text = text.lower()
        text = text.split()

        vocab_to_int, int_to_vocab = self.create_lookup_tables(text)
        int_text = [vocab_to_int[word] for word in text]
        
        PickleHelper().save_preprocessed_data((int_text, vocab_to_int, int_to_vocab, punc_dict))
        
    
    def create_lookup_tables(self, text):
        """
        Create lookup tables for vocabulary
        
        :param text: The text of tv scripts split into words
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        
        word_counts = Counter(text)
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
        int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
        vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
        
        return (vocab_to_int, int_to_vocab)
    
    
    def create_punc_lookup_table(self):
        """
        Generate a dict to turn punctuation into a token.
        
        :return: Tokenize dictionary where the key is the punctuation and the value is the token
        """
        
        punc_dict = {}
        punc_dict['.']  = "||Period||"
        punc_dict[',']  = "||Comma||"
        punc_dict['"']  = "||Quotation_Mark||"
        punc_dict[';']  = "||Semicolon||"
        punc_dict['!']  = "||Exclamation_Mark||"
        punc_dict['?']  = "||Question_Mark"
        punc_dict['(']  = "||Left_Parenthesis||"
        punc_dict[')']  = "||Right_Parenthesis||"
        punc_dict['--'] = "||Dash||"
        punc_dict['\n'] = "||Return||"

        return punc_dict


# In[9]:


class PickleHelper:
    
    def save_preprocessed_data(self, data):
        """
        Save preprocessed training data.
        """
        pickle.dump(data, open('preprocess.p', 'wb'))
        
    def load_preprocessed_data(self):
        """
        Load the Preprocessed training data and return them in batches of <batch_size> or less
        """
        return pickle.load(open('preprocess.p', mode='rb'))
    
    def save_params(self, params):
        """
        Save parameters to file
        """
        pickle.dump(params, open('params.p', 'wb'))
    
    def load_params(self):
        """
        Load parameters from file
        """
        return pickle.load(open('params.p', mode='rb'))


# In[10]:


dataPreprocessor = DataPreprocessor()

dataPreprocessor.preprocess_and_save_data(text)


# ## Checkpoint

# In[11]:


int_text, vocab_to_int, int_to_vocab, token_dict = PickleHelper().load_preprocessed_data()


# ## Checking Tensorflow Version

# In[12]:


class TensorflowVersionChecker:
    
    def check_version(self):
        # Check tensorflow version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'),             'Please use TensorFlow version 1.0 or newer'
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[13]:


tfVersionChecker = TensorflowVersionChecker()

tfVersionChecker.check_version()


# ## Building the Recurrent Neural Network

# In[14]:


class RNN:
    
    def __init__(self):
        self.input_text = None
        self.targets = None
        self.lr = None
        self.initial_state = None
        self.final_state = None
        self.cost = None
        self.train_op = None


# In[15]:


class RNNBuilder:
    
    def create_placeholders(self):
        """
        Create TF Placeholders for input, targets, and learning rate.
        
        :return: Tuple (input, targets, learning rate)
        """
        inputs = tf.placeholder(tf.int32, [None, None], name='input')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        return (inputs, targets, learning_rate)
    
    
    def build_init_cell(self, batch_size, rnn_size, num_rnn_layers, keep_prob):
        """
        Create an RNN Cell and initialize it.
        
        :param batch_size: Size of batches
        :param rnn_size: Size of RNNs
        :param num_rnn_layers: Number of RNN (LSTM) layers
        :param keep_prob: Keep probability value
        :return: Tuple (cell, initialize state)
        """
        
        cell = tf.contrib.rnn.MultiRNNCell(
                [self.build_lstm_cell(rnn_size, keep_prob) for _ in range(num_rnn_layers)])

        initial_state = cell.zero_state(batch_size, tf.float32)
        initial_state = tf.identity(initial_state, name='initial_state')

        return (cell, initial_state)
    
    
    def build_lstm_cell(self, rnn_size, keep_prob):
        """ 
        Build LSTM cell and apply dropout.
        :param rnn_size: Size of RNNs
        :param keep_prob: Keep probability value
        """
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop
    
    
    def get_embed(self, input_data, vocab_size, embed_dim):
        """
        Create embedding for <input_data>.
        
        :param input_data: TF placeholder for text input
        :param vocab_size: Number of words in vocabulary
        :param embed_dim: Number of embedding dimensions
        :return: Embedded input.
        """        
        embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_data)
        return embed
    
    
    def build_rnn(self, cell, inputs):
        """
        Create a RNN using a RNN Cell
        
        :param cell: RNN Cell
        :param inputs: Input text data
        :return: Tuple (Outputs, Final State)
        """
        outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
        final_state = tf.identity(final_state, name='final_state')
        return (outputs, final_state)
    
    
    def build_nn(self, cell, rnn_size, input_data, vocab_size, embed_dim):
        """
        Build part of the neural network
        
        :param cell: RNN cell
        :param rnn_size: Size of rnns
        :param input_data: Input data
        :param vocab_size: Vocabulary size
        :param embed_dim: Number of embedding dimensions
        :return: Tuple (Logits, FinalState)
        """
        
        embed = self.get_embed(input_data, vocab_size, embed_dim)
        outputs, final_state = self.build_rnn(cell, embed)
        
        logits = tf.contrib.layers.fully_connected(inputs=outputs,
                                                   num_outputs=vocab_size,
                                                   activation_fn=None)
        
        return (logits, final_state)


# In[16]:


class RNNGraphBuilder:
    
    def build_rnn_graph(self, 
                        int_to_vocab, 
                        rnn_size,
                        num_rnn_layers,
                        keep_prob,
                        embed_dim):
        
        """
        Build RNN graph.
        
        :param int_to_vocab: Mapping of input words
        :param rnn_size: Size of rnns
        :param num_rnn_layers: Number of RNN (LSTM) layers
        :param keep_prob: Keep probability value
        :param embed_dim: Number of embedding dimensions
        :return: Tuple (RNN, TrainGraph, Probs) 
        """
        
        train_graph = tf.Graph()
        rnn = RNN()
        rnnBuilder = RNNBuilder()

        with train_graph.as_default():
            vocab_size = len(int_to_vocab)
            
            input_text, targets, lr = rnnBuilder.create_placeholders()
            input_data_shape = tf.shape(input_text)
            
            rnn.input_text = input_text
            rnn.targets = targets
            rnn.lr = lr
            
            cell, initial_state = rnnBuilder.build_init_cell(input_data_shape[0], 
                                                             rnn_size, 
                                                             num_rnn_layers, 
                                                             keep_prob)
            
            rnn.initial_state = initial_state
 
            logits, final_state = rnnBuilder.build_nn(cell, 
                                                      rnn_size, 
                                                      input_text, 
                                                      vocab_size, 
                                                      embed_dim)
    
            rnn.final_state = final_state

            # Probabilities for generating words
            probs = tf.nn.softmax(logits, name='probs')

            # Loss function
            cost = seq2seq.sequence_loss(logits,
                                         targets, 
                                         tf.ones([input_data_shape[0], input_data_shape[1]]))

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)                                 for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            
            rnn.cost = cost
            rnn.train_op = train_op
            
            return (rnn, train_graph, probs)


# In[17]:


rnn_size = 256
num_rnn_layers = 2
keep_prob = 0.75
embed_dim = 200


# In[18]:


rnnGraphBuilder = RNNGraphBuilder()

rnn, train_graph, probs = rnnGraphBuilder.build_rnn_graph(int_to_vocab, 
                                                          rnn_size,
                                                          num_rnn_layers,
                                                          keep_prob,
                                                          embed_dim)


# ## Training the Network

# In[19]:


class RNNTrainer:
    
    def train_rnn(self, rnn, train_graph):
        """
        Train and save RNN model.
        
        :param rnn: RNN to train and save
        :train_graph: RNN graph
        """
            
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            dataBatchGenerator = DataBatchGenerator()
        
            batches = dataBatchGenerator.get_batches(int_text, batch_size, seq_length)

            for epoch_i in range(num_epochs):
                feed = {rnn.input_text: batches[0][0]}
                
                state = sess.run(rnn.initial_state, feed_dict=feed)

                for batch_i, (x, y) in enumerate(batches):
                    feed = {
                        rnn.input_text: x,
                        rnn.targets: y,
                        rnn.initial_state: state,
                        rnn.lr: learning_rate}
                    
                    train_loss, state, _ = sess.run([rnn.cost, rnn.final_state, rnn.train_op],
                                                     feed_dict=feed)

                    # Show every <show_every_n_batches> batches
                    if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                        print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                                epoch_i+1,
                                batch_i+1,
                                len(batches),
                                train_loss))

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, save_dir)
            
            print('\nModel Trained and Saved')


# In[20]:


class DataBatchGenerator:

    def get_batches(self, int_text, batch_size, seq_len):
        """
        Return batches of input and target
        
        :param int_text: Text with the words replaced by their ids
        :param batch_size: The size of batch
        :param seq_len: The length of sequence
        :return: Batches as a Numpy array
        """
        
        """
        Example Input:
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)
        
        Example Output:
        [
          # First Batch
          [
            # Batch of Input
            [[ 1  2], [ 7  8], [13 14]]
            # Batch of targets
            [[ 2  3], [ 8  9], [14 15]]
          ]

          # Second Batch
          [
            # Batch of Input
            [[ 3  4], [ 9 10], [15 16]]
            # Batch of targets
            [[ 4  5], [10 11], [16 17]]
          ]

          # Third Batch
          [
            # Batch of Input
            [[ 5  6], [11 12], [17 18]]
            # Batch of targets
            [[ 6  7], [12 13], [18  1]]
          ]
        ]
        """
        
        n_chars_per_batch = batch_size * seq_len
        n_batches = int(len(int_text) / n_chars_per_batch)
        full_batch_size = n_batches * n_chars_per_batch

        # Drop the last few characters to make only full batches
        x_data = np.array(int_text[: full_batch_size])
        y_data = np.array(int_text[1: full_batch_size + 1])

        x_batches = np.split(x_data.reshape(batch_size, -1), n_batches, 1)
        y_batches = np.split(y_data.reshape(batch_size, -1), n_batches, 1)

        first_x_char = x_batches[0][0][0]
        y_batches[n_batches-1][batch_size-1][-1] = first_x_char

        return np.array(list(zip(x_batches, y_batches)))


# In[21]:


learning_rate = 0.001
num_epochs = 50
batch_size = 128
seq_length = 2
show_every_n_batches = 10
save_dir = './save'


# In[22]:


rnnTrainer = RNNTrainer()

rnnTrainer.train_rnn(rnn, train_graph)


# ## Saving Parameters

# In[23]:


pickleHelper = PickleHelper()


# In[24]:


pickleHelper.save_params((seq_length, save_dir))


# ## Checkpoint

# In[25]:


_, vocab_to_int, int_to_vocab, punc_dict = pickleHelper.load_preprocessed_data()
seq_length, load_dir = pickleHelper.load_params()


# ## Generating TV Script

# In[26]:


class TVScriptGenerator:
    
    def generate_tv_script(self, gen_length, prime_word):
        """
        Generate TV script using the trainde RNN model.
        
        :param gen_length: Generation length
        :param prime_word: Prime word to use 
        :return: Generated TV script
        """
        
        loaded_graph = tf.Graph()

        tensorLoader = TensorLoader()
        wordSelector = WordSelector()
        
        with tf.Session(graph=loaded_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            # Load saved model
            loader = tf.train.import_meta_graph(load_dir + '.meta')
            loader.restore(sess, load_dir)

            # Get Tensors from loaded model
            input_text, initial_state, final_state, probs =                 tensorLoader.get_tensors(loaded_graph)

            # Sentence generation setup
            gen_sentences = [prime_word + ':']
            feed = {input_text: np.array([[1]])}
            prev_state = sess.run(initial_state, feed_dict=feed)

            # Generate sentences
            for n in range(gen_length):
                # Dynamic Input
                dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
                dyn_seq_length = len(dyn_input[0])

                # Get Prediction
                feed = {input_text: dyn_input, initial_state: prev_state}
                probabilities, prev_state = sess.run([probs, final_state],
                                                      feed_dict=feed)

                pred_word = wordSelector.pick_word(probabilities[dyn_seq_length-1], 
                                                   int_to_vocab)
                gen_sentences.append(pred_word)

            # Remove punctuation tokens
            tv_script = ' '.join(gen_sentences)
            for key, token in punc_dict.items():
                ending = ' ' if key in ['\n', '(', '"'] else ''
                tv_script = tv_script.replace(' ' + token.lower(), key)
            tv_script = tv_script.replace('\n ', '\n')
            tv_script = tv_script.replace('( ', '(')

            return tv_script


# In[27]:


class TensorLoader:

    def get_tensors(self, loaded_graph):
        """
        Get input, initial state, final state, and probabilities tensor from <loaded_graph>
        
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
        """

        with tf.Session(graph=loaded_graph) as sess:
            inputs = tf.get_default_graph().get_tensor_by_name("input:0")
            initial_state = tf.get_default_graph().get_tensor_by_name("initial_state:0")
            final_state = tf.get_default_graph().get_tensor_by_name("final_state:0")
            probs = tf.get_default_graph().get_tensor_by_name("probs:0")

        return (inputs, initial_state, final_state, probs)


# In[28]:


class WordSelector:
    
    def pick_word(self, probabilities, int_to_vocab):
        """
        Pick the next word in the generated text.
        
        :param probabilities: Probabilites of the next word
        :param int_to_vocab: Dictionary of word ids as the keys and words as the values
        :return: String of the predicted word
        """
    #     max_idx = np.argmax(probabilities)
    #     return int_to_vocab[max_idx]

        top_n = 5
        p = np.squeeze(probabilities)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        i = np.random.choice(len(int_to_vocab), 1, p=p)[0]
        
        return int_to_vocab[i]


# In[29]:


gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'


# In[30]:


tvScriptGenerator = TVScriptGenerator()

tv_script = tvScriptGenerator.generate_tv_script(gen_length, prime_word)

print(tv_script)

