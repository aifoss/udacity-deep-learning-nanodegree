
# coding: utf-8

# # Character-wise Recurrent Neural Network

# In[1]:


import time

import numpy as np
import tensorflow as tf


# # Getting and Preprocessing Data

# ### Data

# In[2]:


class Data:
    
    def __init__(self):
        text = None
        chars = None
        chars_to_ints = None
        ints_to_chars = None
        encoded = None


# ## DataPreprocessor

# In[3]:


class DataPreprocessor:
    
    def load_and_preprocess_data(self, input_file):
        """
        Load and preprocess data.
        
        Arguments
        ---------
        : input_file: Input file name
        """
        
        print("\nLoading and preprocesing data ...\n")
        
        data = Data()
        
        with open(input_file, 'r') as f:
            data.text = f.read()
                
        data.chars = sorted(set(data.text))
        data.chars_to_ints = {c: i for i, c in enumerate(data.chars)}
        data.ints_to_chars = dict(enumerate(data.chars))
        data.encoded = np.array([data.chars_to_ints[c] for c in data.text], dtype=np.int32)
        self.log_data(data)
        
        print("Loaded and preprocessed data\n")
        
        return data
    
    
    def log_data(self, data):
        txt = ""
        for ii in range(0, 100):
            ch = data.text[ii]
            ch = '(NEWLINE)' if ch == '\n' else ch
            txt += ch
        print("text[:100]:\n{}\n".format(txt))    
        
        print("len(chars):\n{}\n".format(len(data.chars)))
        print("chars[:50]:\n{}\n".format(data.chars[:50]))
        print("chars_to_ints:\n")
        for ii in range(0, 10):
            ch = data.chars[ii]
            ch = 'NEWLINE' if ch == '\n' else 'SPACE' if ch == ' ' else ch
            print("chars_to_ints[{}]: {}".format(ch, ii))
        print("")
        print("ints_to_chars:\n")
        for ii in range(0, 10):
            ch = data.ints_to_chars[ii]
            ch = 'NEWLINE' if ch == '\n' else 'SPACE' if ch == ' ' else ch
            print("ints_to_chars[{}]: {}".format(ii, ch))
        print("")
        print("encoded.shape:\n{}\n".format(data.encoded.shape))
        print("encoded[:100]:\n{}\n".format(data.encoded[:100]))


# In[4]:


input_file = 'anna.txt'


# In[5]:


dataPreprocessor = DataPreprocessor()
data = dataPreprocessor.load_and_preprocess_data(input_file)


# # Building Character-wise RNN Model

# ## RNNetwork

# In[6]:


class RNNetwork:
    
    def create_placeholders(self, batch_size, num_steps):
        """ 
        Define placeholders for inputs, targets, and dropout 
    
        Arguments
        ---------
        : batch_size: Batch size, number of sequences per batch
        : num_steps: Number of sequence steps in a batch
        """
        
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        
        print("Created placeholders\n")
        
        return inputs, targets, keep_prob
    
    
    def build_lstm_layers(self, keep_prob, lstm_size, num_layers, batch_size):
        """
        Build LSTM layers.
    
        Arguments
        ---------
        : keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        : lstm_size: Size of the hidden layers in the LSTM cells
        : num_layers: Number of LSTM layers
        : batch_size: Batch size
        """
        
        cell = tf.contrib.rnn.MultiRNNCell(
            [self.build_lstm_cell(lstm_size, keep_prob) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, tf.float32)
        
        print("Built LSTM layers\n")
        
        return cell, initial_state
    
        
    def build_lstm_cell(self, lstm_size, keep_prob):
        """
        Build LSTM cell.
    
        Arguments
        ---------
        : lstm_size: Size of the hidden layers in the LSTM cells
        : keep_prob: Scalar tensor (tf.placeholder) for the dropout keep probability
        """
        
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        
        print("Built LSTM cell")
        
        return drop
    
    
    def build_output_layer(self, lstm_output, in_size, out_size):
        """
        Build a softmax layer, return the softmax output and logits.
    
        Arguments
        ---------
        : lstm_output: List of output tensors from the LSTM layer
        : in_size: Size of the input tensor, for example, size of the LSTM cells
        : out_size: Size of this softmax layer
        """
        
        # Reshape output so it's a bunch of rows, one row for each step for each sequence.
        # That is, the shape should be batch_size*num_steps rows by lstm_size columns
        
        # Concatenate lstm_output over axis 1 (the columns)
        seq_output = tf.concat(lstm_output, axis=1)
        
        # Reshape seq_output to a 2D tensor with lstm_size columns
        x = tf.reshape(seq_output, [-1, in_size])
        
        # Connect the RNN outputs to a softmax layer
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal((in_size, out_size), stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(out_size))
            
        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and sequence    
        logits = tf.nn.bias_add(tf.matmul(x, softmax_w), softmax_b)
        
        # Use softmax to get the probabilities for predicted characters
        out = tf.nn.softmax(logits, name='predictions')
        
        print("Built output layer\n")
        
        return out, logits
    

    def add_training_loss_computation(self, logits, targets, lstm_size, num_classes):
        """
        Calculate the loss from the logits and the targets.
    
        Arguments
        ---------
        : logits: Logits from final fully connected layer
        : targets: Targets for supervised learning
        : lstm_size: Number of LSTM hidden units
        : num_classes: Number of classes in targets
        """
        
        # One-hot encode targets and reshape to match logits, one row per sequence per step
        y_one_hot = tf.one_hot(targets, num_classes)
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        
        # Softmax cross entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
        loss = tf.reduce_mean(cross_entropy)
        
        print("Added training loss computation\n")
        
        return loss
    
    
    def build_optimizer(self, loss, learning_rate, grad_clip):
        """
        Build optmizer for training, using gradient clipping.
    
        Arguments:
        ---------
        : loss: Network loss
        : learning_rate: Learning rate for optimizer
        : grad_clip: For gradient clipping 
        """
        
        # Optimizer for training, using gradient clipping to control exploding gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        
        print("Built optimizer\n")
        
        return optimizer


# ## CharRNNModel

# In[7]:


class CharRNNModel:
    
    def __init__(self, 
                 num_classes, 
                 batch_size, 
                 num_steps,
                 lstm_size, 
                 num_layers, 
                 learning_rate,
                 grad_clip, 
                 sampling=False):

        """
        Build CharRNN model.
        
        Arguments
        ---------
        : num_classes: Number of classes in targets
        : batch_size: Batch size, number of sequences per batch
        : num_steps: Number of sequence steps in a batch
        : lstm_size: Number of LSTM hidden units
        : num_layers: Number of LSTM layers
        : learning_rate: Learning rate
        : grad_clip: For gradient clipping
        : sampling: Whether or not the model is used for sampling
        """
        
        print("\nBuilding CharRNN model ...\n")
        
        # When we're using this network for sampling later, we'll be passing in
        # one character at a time, so providing an option for that
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        
        self.lstm_size = lstm_size
        self.batch_size, self.num_steps = batch_size, num_steps
        
        tf.reset_default_graph()
        
        # Create RNNetwork object
        network = RNNetwork()
        
        # Build the placeholder tensors
        self.inputs, self.targets, self.keep_prob = network.create_placeholders(self.batch_size, 
                                                                                self.num_steps)
        
        # Build the LSTM layers
        cell, self.initial_state = network.build_lstm_layers(self.keep_prob, 
                                                             self.lstm_size, 
                                                             num_layers,
                                                             batch_size)
        
        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # Run each sequence step through the RNN with tf.nn.dynamic_rnn
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = state
        
        # Get softmax predictions and logits
        self.prediction, self.logits = network.build_output_layer(outputs, 
                                                                  self.lstm_size,
                                                                  num_classes)
        
        # Loss and optimizer (with gradient clipping)
        self.loss = network.add_training_loss_computation(self.logits, 
                                                          self.targets, 
                                                          self.lstm_size, 
                                                          num_classes)
        
        self.optimizer = network.build_optimizer(self.loss, learning_rate, grad_clip)
        
        print("Built CharRNN model\n")


# In[8]:


num_classes = len(data.chars)
batch_size = 64
num_steps = 50
lstm_size = 128
num_layers = 2
learning_rate = 0.001
grad_clip = 5


# In[9]:


model = CharRNNModel(num_classes,
                     batch_size,
                     num_steps,
                     lstm_size,
                     num_layers,
                     learning_rate,
                     grad_clip,
                     False)


# # Training Character-wise RNN Model

# ### DataBatchGenerator

# In[10]:


class DataBatchGenerator:
    
    def get_batches(self, arr, n_seqs, n_steps):
        """
        Create a generator that returns batches of size n_seqs x n_steps from arr.
        
        Arguments
        ---------
        : arr: Array you want to make batches from
        : n_seqs: Number of sequences per batch
        : n_steps: Number of sequence steps per batch
        """
        
        # Get the number of characters per batch and number of batches we can make
        chars_per_batch = n_seqs * n_steps # batch size
        n_batches = len(arr)//chars_per_batch
        
        # Keep only enough characters to make full batches
        arr = arr[:n_batches * chars_per_batch]
        
        # Reshape into n_seqs rows
        arr = arr.reshape((n_seqs, -1))
        
        # Generate each batch
        for n in range(0, arr.shape[1], n_steps):
            # features
            x = arr[:, n:n+n_steps]
            # targets
            y = np.zeros_like(x)
            
            # Targets are inputs shifted by one character
            # First input character is last target character
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
            
            yield x, y


# ## RNNModelTrainer

# In[11]:


class RNNModelTrainer:
        
    def train_model(self, 
                    model, 
                    data, 
                    epochs, 
                    keep_prob, 
                    save_every_n,
                    max_to_keep):
        """
        Train RNN model.
        
        Arguments
        ---------
        : model: Model to train
        : data: Data to train model on
        : epochs: Number of epochs to train
        : keep_prob: Keep proability to pass to model
        : save_every_n: Interval to save session
        : max_to_keep: Param to pass to session saver
        """
        
        print("\nTraining CharRNN model ...\n")
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver(max_to_keep=max_to_keep)
            
            # Load a checkpoint and resume training
            #saver.restore(sess, 'checkpoints/_____.ckpt')
            
            counter = 0
            
            for e in range(epochs):
                new_state = sess.run(model.initial_state)
                loss = 0
                
                dataBatchGenerator = DataBatchGenerator()
                batches = dataBatchGenerator.get_batches(data.encoded, model.batch_size, model.num_steps)

                for x, y in batches:
                    counter += 1
                    start = time.time()
                    
                    feed = {model.inputs: x,
                            model.targets: y,
                            model.keep_prob: keep_prob,
                            model.initial_state: new_state}
                    
                    batch_loss, new_state, _ = sess.run([model.loss,
                                                         model.final_state,
                                                         model.optimizer],
                                                         feed_dict=feed)
                    
                    end = time.time()
                    
                    print('Epoch: {}/{}... '.format(e+1, epochs),
                          'Training Step: {}... '.format(counter),
                          'Training loss: {:.4f}... '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    
                    if (counter % save_every_n == 0):
                        saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, 
                                                                           model.lstm_size))
        
            saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, model.lstm_size))
            
        print("\nTraining complete\n")


# In[12]:


epochs = 20
keep_prob = 0.5
save_every_n = 200
max_to_keep = 100


# In[13]:


mkdir "./checkpoints"


# In[14]:


modelTrainer = RNNModelTrainer()

modelTrainer.train_model(model,
                         data,
                         epochs,
                         keep_prob,
                         save_every_n,
                         max_to_keep)


# In[15]:


tf.train.get_checkpoint_state('checkpoints')


# # Sampling

# ## Sampler

# In[16]:


class Sampler:
    
    def sample(self, model, data, checkpoint, n_samples, prime="The "):
        """
        Get sample model outputs from checkpoint.
        
        Arguments
        ---------
        : model: CharRNNModel object
        : data: Dataset
        : checkpoint: Checkpoint from which to get samples
        : n_samples: Number of samples
        : prime: Word to prime sampling
        """
        
        samples = [c for c in prime]
        num_chars = len(data.chars)
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            new_state = sess.run(model.initial_state)
            
            for c in prime:
                x = np.zeros((1, 1))
                x[0,0] = data.chars_to_ints[c]
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state], 
                                             feed_dict=feed)
                
            c = self.pick_top_n(preds, num_chars)
            samples.append(data.ints_to_chars[c])
            
            for i in range(n_samples):
                x[0,0] = c
                feed = {model.inputs: x,
                        model.keep_prob: 1.,
                        model.initial_state: new_state}
                preds, new_state = sess.run([model.prediction, model.final_state],
                                             feed_dict=feed)
                
                c = self.pick_top_n(preds, num_chars)
                samples.append(data.ints_to_chars[c])
                
        return ''.join(samples)
    
    
    def pick_top_n(self, preds, num_chars, top_n=5):
        """
        Pick random char among top_n chars.
        """
        p = np.squeeze(preds)
        p[np.argsort(p)[:-top_n]] = 0
        p = p / np.sum(p)
        c = np.random.choice(num_chars, 1, p=p)[0]
        return c


# In[17]:


model = CharRNNModel(num_classes,
                     batch_size,
                     num_steps,
                     lstm_size,
                     num_layers,
                     learning_rate,
                     grad_clip,
                     True)


# In[18]:


sampler = Sampler()


# In[19]:


n_samples = 1000
prime = "Far"


# In[20]:


checkpoint = "checkpoints/i1000_l128.ckpt"

samp = sampler.sample(model, data, checkpoint, n_samples, prime)
print(samp)


# In[21]:


checkpoint = "checkpoints/i10000_l128.ckpt"

samp = sampler.sample(model, data, checkpoint, n_samples, prime)
print(samp)


# In[22]:


checkpoint = tf.train.latest_checkpoint('checkpoints')

samp = sampler.sample(model, data, checkpoint, n_samples, prime)
print(samp)

