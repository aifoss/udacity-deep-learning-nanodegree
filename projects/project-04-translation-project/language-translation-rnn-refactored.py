
# coding: utf-8

# # Language Translation RNN

# In[1]:


import os
import copy
import pickle
import warnings

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from tensorflow.python.layers.core import Dense


# ## Loading Data

# In[2]:


class DataLoader:
    
    def load_data(self, path):
        """
        Load dataset from file.
        """
        input_file = os.path.join(path)
        with open(input_file, 'r', encoding='utf-8') as f:
            return f.read()


# In[3]:


source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'


# In[4]:


dataLoader = DataLoader()

source_text = dataLoader.load_data(source_path)
target_text = dataLoader.load_data(target_path)


# ## Exploring Data

# In[5]:


class DataExplorer:
    
    def explore_data(self, source_text, target_text, sent_range):
        """
        Explore sample sentences from dataset.
        """
        print('Dataset Stats')
        print('Roughly the number of unique words: {}'              .format(len({word: None for word in source_text.split()})))

        sentences = source_text.split('\n')
        word_counts = [len(sentence.split()) for sentence in sentences]
        
        print('Number of sentences: {}'.format(len(sentences)))
        print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

        print()
        print('English sentences {} to {}:'.format(*sent_range))
        print('\n'.join(source_text.split('\n')[sent_range[0]:sent_range[1]]))
        print()
        print('French sentences {} to {}:'.format(*sent_range))
        print('\n'.join(target_text.split('\n')[sent_range[0]:sent_range[1]]))


# In[6]:


view_sentence_range = (0, 10)


# In[7]:


dataExplorer = DataExplorer()

dataExplorer.explore_data(source_text, target_text, view_sentence_range)


# ## Preprocessing Data

# In[8]:


class DataPreprocessor:
    
    def preprocess_and_save_data(self, source_text, target_text):
        """
        Preprocess text data and save to file.
        """
        source_text = source_text.lower()
        target_text = target_text.lower()

        source_vocab_to_int, source_int_to_vocab = self.create_lookup_tables(source_text)
        target_vocab_to_int, target_int_to_vocab = self.create_lookup_tables(target_text)

        source_id_text, target_id_text = self.text_to_ids(source_text, 
                                                          target_text, 
                                                          source_vocab_to_int, 
                                                          target_vocab_to_int)

        PickleHelper().save_preprocessed_data(((source_text, target_text),
                                               (source_id_text, target_id_text),
                                               (source_vocab_to_int, target_vocab_to_int),
                                               (source_int_to_vocab, target_int_to_vocab)))
                                              
                                              
    def create_lookup_tables(self, text):
        """
        Create lookup tables for vocabulary.
        """
        CODES = {'<PAD>': 0, '<EOS>': 1, '<UNK>': 2, '<GO>': 3}
        
        vocab = set(text.split())
        vocab_to_int = copy.copy(CODES)

        for v_i, v in enumerate(vocab, len(CODES)):
            vocab_to_int[v] = v_i

        int_to_vocab = {v_i: v for v, v_i in vocab_to_int.items()}

        return (vocab_to_int, int_to_vocab)

    
    def text_to_ids(self, source_text, target_text, source_vocab_to_int, target_vocab_to_int):
        """
        Convert source and target text to proper word ids.

        :param source_text: String that contains all the source text.
        :param target_text: String that contains all the target text.
        :param source_vocab_to_int: Dictionary to go from the source words to an id
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: A tuple of lists (source_id_text, target_id_text)
        """
        source_id_text = [[source_vocab_to_int[word] for word in line.split(' ') if word != '']                           for line in source_text.split('\n')]
        
        target_id_text = [[target_vocab_to_int[word] for word in line.split(' ') if word != '']                               + [target_vocab_to_int['<EOS>']]                           for line in target_text.split('\n')]

        return (source_id_text, target_id_text)


# In[9]:


class PickleHelper:
    
    def save_preprocessed_data(self, data):
        """
        Save preprocessed training data.
        """
        pickle.dump(data, open('preprocess.p', 'wb'))
        
    def load_preprocessed_data(self):
        """
        Load the Preprocessed training data and return them in batches of <batch_size> or less.
        """
        return pickle.load(open('preprocess.p', mode='rb'))
    
    def save_params(self, params):
        """
        Save parameters to file.
        """
        pickle.dump(params, open('params.p', 'wb'))
    
    def load_params(self):
        """
        Load parameters from file.
        """
        return pickle.load(open('params.p', mode='rb'))


# In[10]:


dataPreprocessor = DataPreprocessor()

dataPreprocessor.preprocess_and_save_data(source_text, target_text)


# ## Checkpoint

# In[11]:


pickleHelper = PickleHelper()

((source_text, target_text),
 (source_int_text, target_int_text), 
 (source_vocab_to_int, target_vocab_to_int),
 (source_int_to_vocab, target_int_to_vocab)) = pickleHelper.load_preprocessed_data()


# ## Checking TensorFlow Version and GPU

# In[12]:


class TensorFlowGPUChecker:
    
    def check(self):
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.1'),             'Please use TensorFlow version 1.1 or newer'
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[13]:


versionChecker = TensorFlowGPUChecker()

versionChecker.check()


# ## Building RNN Model

# In[14]:


class Seq2SeqRNN:
    
    def __init__(self):
        self.inputs = None
        self.targets = None
        
        self.learning_rate = None
        self.keep_prob = None
        
        self.source_seq_len = None
        self.target_seq_len = None
        self.max_target_len = None
        
#         self.encoder_output = None
#         self.encoder_state = None
        
#         self.decoder_input = None
#         self.training_decoder_output = None
#         self.inference_decoder_output = None
        
        self.training_logits = None
        self.inference_logits = None
        
        self.cost = None
        self.train_op = None


# In[15]:


class RNNBuilder:
    
    def create_placeholders(self):
        """
        Create TF Placeholders for input, targets, learning rate, 
        and lengths of source and target sequences.

        :return: Tuple (inputs, targets, learning_rate, keep_prob,
                        source_seq_len, target_seq_len, max_target_len)
        """
        inputs = tf.placeholder(tf.int32, (None, None), name='input')
        targets = tf.placeholder(tf.int32, (None, None), name='targets')

        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        source_seq_len = tf.placeholder(tf.float32, (None,), name='source_seq_len')
        target_seq_len = tf.placeholder(tf.int32, (None,), name='target_seq_len')
        max_target_len = tf.reduce_max(target_seq_len, name='max_target_len')
        
        return (inputs, targets, learning_rate, keep_prob,                 source_seq_len, target_seq_len, max_target_len)

    
    def build_encoding_layer(self, 
                             rnn_inputs, 
                             rnn_size, 
                             num_layers, 
                             keep_probability, 
                             source_seq_len,
                             source_vocab_size,
                             enc_embedding_size):
        """
        Create encoding layer.

        :param rnn_inputs: Inputs for the RNN
        :param rnn_size: RNN Size
        :param num_layers: Number of layers
        :param keep_prob: Dropout keep probability
        :param source_seq_len: List of the lengths of each sequence in the batch
        :param source_vocab_size: Vocabulary size of source data
        :param enc_embedding_size: Embedding size of source data
        :return: Tuple (enc_output, enc_state)
        """
        # Encodder embedding
        enc_embed = tf.contrib.layers.embed_sequence(rnn_inputs,
                                                     source_vocab_size,
                                                     enc_embedding_size)

        # Encoder cell
        def make_cell(rnn_size):
            initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2)
            cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer)
            cell = tf.contrib.rnn.DropoutWrapper(cell, keep_probability)
            return cell

        enc_cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell(rnn_size) for _ in range(num_layers)])

        enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, 
                                                  enc_embed, 
                                                  source_seq_len,
                                                  dtype=tf.float32)

        return (enc_output, enc_state)
    
    
    def process_decoder_input(self, targets, target_vocab_to_int, batch_size):
        """
        Preprocess target data for encoding.

        :param targets: Target Placehoder
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param batch_size: Batch Size
        :return: Preprocessed target data
        """
        ending = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
        dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), 
                               ending], 1)        
        return dec_input
    
    
    def build_decoding_layer(self, 
                             enc_state,
                             dec_input, 
                             target_seq_len, 
                             max_target_len,
                             rnn_size,
                             num_layers,
                             target_vocab_to_int,
                             target_vocab_size,
                             batch_size,
                             keep_probability,
                             dec_embedding_size):
        """
        Create decoding layer.

        :param enc_state: Encoder state
        :param dec_input: Decoder input
        :param target_seq_len: The lengths of each sequence in the target batch
        :param max_target_len: Maximum length of target sequences
        :param rnn_size: RNN Size
        :param num_layers: Number of layers
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :param target_vocab_size: Size of target vocabulary
        :param batch_size: The size of the batch
        :param keep_prob: Dropout keep probability
        :param dec_embedding_size: Decoding embedding size
        :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
        """
        # Decoder embedding
        dec_embedding = tf.Variable(tf.random_uniform([target_vocab_size, dec_embedding_size]))
        dec_embed = tf.nn.embedding_lookup(dec_embedding, dec_input)

        # Decoder cell
        def make_cell(rnn_size):
            initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2)
            cell = tf.contrib.rnn.LSTMCell(rnn_size, initializer)
            cell = tf.contrib.rnn.DropoutWrapper(cell, keep_probability)
            return cell

        dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])

        # Dense layer to translate the decoder's output at each time step
        # into a chocie from the target vocabulary
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=initializer)

        # Training decoder
        with tf.variable_scope("decode"):
            training_decoder_output = self.get_training_decoding(enc_state, 
                                                                 dec_cell, 
                                                                 dec_embed, 
                                                                 target_seq_len, 
                                                                 max_target_len, 
                                                                 output_layer)

        # Inference decoder
        with tf.variable_scope("decode", reuse=True):
            inference_decoder_output = self.get_inference_decoding(enc_state, 
                                                                   dec_cell, 
                                                                   dec_embedding, 
                                                                   target_vocab_to_int['<GO>'],
                                                                   target_vocab_to_int['<EOS>'],
                                                                   max_target_len,
                                                                   target_vocab_size,
                                                                   output_layer,
                                                                   batch_size)

        return (training_decoder_output, inference_decoder_output)

    
    def get_training_decoding(self,
                              enc_state, 
                              dec_cell, 
                              dec_embed_input, 
                              target_seq_len,
                              max_target_len,
                              output_layer):
        """
        Create a decoding layer for training.

        :param enc_state: Encoder state
        :param dec_cell: Decoder RNN cell
        :param dec_embed_input: Decoder embedded input
        :param target_seq_len: The lengths of each sequence in the target batch
        :param max_target_len: Maximum length of target sequences
        :param output_layer: Function to apply the output layer
        :return: BasicDecoderOutput containing training logits and sample_id
        """
        helper = tf.contrib.seq2seq.TrainingHelper(dec_embed_input,
                                                   target_seq_len,
                                                   time_major=False)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state, output_layer)

        return tf.contrib.seq2seq.dynamic_decode(decoder,
                                                 impute_finished=True,
                                                 maximum_iterations=max_target_len)[0]
    
    
    def get_inference_decoding(self,
                               enc_state, 
                               dec_cell, 
                               dec_embeddings, 
                               start_of_seq_id,
                               end_of_seq_id,
                               max_target_len,
                               vocab_size,
                               output_layer,
                               bacth_size):
        """
        Create a decoding layer for inference.

        :param enc_state: Encoder state
        :param dec_cell: Decoder RNN cell
        :param dec_embeddings: Decoder embeddings
        :param start_of_seq_id: <GO> ID
        :param end_of_seq_id: <EOS> ID
        :param max_target_len: Maximum length of target sequences
        :param vocab_size: Size of decoder/target vocabulary
        :param output_layer: Function to apply the output layer
        :param batch_size: Batch size
        :return: BasicDecoderOutput containing inference logits and sample_id
        """
        start_tokens = tf.tile(tf.constant([start_of_seq_id], dtype=tf.int32),
                               [batch_size],
                               name='start_tokens')

        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,
                                                          start_tokens,
                                                          end_of_seq_id)

        decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state, output_layer)

        return tf.contrib.seq2seq.dynamic_decode(decoder,
                                                 impute_finished=True,
                                                 maximum_iterations=max_target_len)[0]


# In[16]:


class Seq2SeqGraphBuilder:
    
    def build_train_graph(self,
                          batch_size,
                          rnn_size,
                          num_layers,
                          enc_embedding_size,
                          dec_embedding_size,
                          keep_probability,
                          source_vocab_size,
                          target_vocab_size,
                          target_vocab_to_int):
                          
        """
        Build the training graph with the Seq2Seq RNN.

        :param batch_size: Batch Size
        :param rnn_size: RNN Size
        :param num_layers: Number of layers
        :param enc_embedding_size: Decoder embedding size
        :param dec_embedding_size: Encoder embedding size
        :param keep_probability: Dropout keep probability
        :param source_vocab_size: Source vocabulary size
        :param target_vocab_size: Target vocabulary size
        :param target_vocab_to_int: Dictionary to go from the target words to an id
        :return: RNN and training graph
        """
        
        train_graph = tf.Graph()
        
        rnn = Seq2SeqRNN()
        rnnBuilder = RNNBuilder()
        optimizerTuner = OptimizerTuner()
  
        with train_graph.as_default():
        
            # Create placeholders
            inputs, targets, lr, keep_prob, source_seq_len, target_seq_len, max_target_len =                 rnnBuilder.create_placeholders()
            
            rnn.inputs, rnn.targets = inputs, targets
            rnn.learning_rate, rnn.keep_prob = lr, keep_prob
            rnn.source_seq_len, rnn.target_seq_len, rnn.max_target_len =                 source_seq_len, target_seq_len, max_target_len
        
            # Build encoding layer
            enc_output, enc_state = rnnBuilder.build_encoding_layer(tf.reverse(inputs, [-1]),
                                                                    rnn_size,
                                                                    num_layers,
                                                                    keep_probability,
                                                                    source_seq_len,
                                                                    source_vocab_size,
                                                                    enc_embedding_size)
            
            # Process decoder input
            dec_input = rnnBuilder.process_decoder_input(targets, 
                                                         target_vocab_to_int, 
                                                         batch_size)             
                                     
            # Build decoding layer
            training_decoder_output, inference_decoder_output =                 rnnBuilder.build_decoding_layer(enc_state,
                                                dec_input, 
                                                target_seq_len, 
                                                max_target_len,
                                                rnn_size,
                                                num_layers,
                                                target_vocab_to_int,
                                                target_vocab_size,
                                                batch_size,
                                                keep_probability,
                                                dec_embedding_size)
                
            training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
            inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')

            rnn.training_logits, rnn.inference_logits = training_logits, inference_logits
            
            masks = tf.sequence_mask(target_seq_len, max_target_len, 
                                     dtype=tf.float32, 
                                     name='masks')
            
            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
                
                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)
                train_op = optimizerTuner.get_gradient_clipped_optimizer(optimizer, cost)
            
                rnn.cost, rnn.train_op = cost, train_op
            
        return (rnn, train_graph)


# In[17]:


class OptimizerTuner:
    
    def get_gradient_clipped_optimizer(self, optimizer, cost):
        """
        Apply gradient clipping to optimizer.
        
        :param optimizer: Optimizer to apply gradient clipping to
        :param cost: Loss function
        :return: Optimizer with gradient clipping
        """
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var)                             for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)
        return train_op


# In[36]:


epochs = 20
batch_size = 128
rnn_size = 256
num_layers = 2
encoding_embedding_size = 20
decoding_embedding_size = 20
learning_rate = 0.001
keep_probability = 0.75
display_step = 20


# In[19]:


source_vocab_size = len(source_vocab_to_int)
target_vocab_size = len(target_vocab_to_int)


# In[20]:


tf.reset_default_graph()


# In[21]:


graphBuilder = Seq2SeqGraphBuilder()

rnn, train_graph = graphBuilder.build_train_graph(batch_size,
                                                  rnn_size,
                                                  num_layers,
                                                  encoding_embedding_size,
                                                  decoding_embedding_size,
                                                  keep_probability,
                                                  source_vocab_size,
                                                  target_vocab_size,
                                                  target_vocab_to_int)


# ## Training Seq2Seq Model

# In[34]:


class ModelTrainer:
    
    def train_seq2seq_model(self, rnn, train_graph):
        """
        Train and save the Seq2Seq model.
        
        :param rnn: Seq2Seq RNN model
        :param train_graph: TensorFlow graph
        """
        
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            batchGenerator = DataBatchGenerator()
            accuracyCalculator = AccuracyCalculator()
            
            for epoch_i in range(epochs):
                batches = batchGenerator.get_batches(batch_size,
                                                     train_source,
                                                     train_target,
                                                     source_pad_int,
                                                     target_pad_int)
                
                for batch_i, (source_batch, target_batch, source_lengths, target_lengths)                         in enumerate(batches):
                        
                    # Training step
                    feed = {rnn.inputs: source_batch,
                            rnn.targets: target_batch,
                            rnn.learning_rate: learning_rate,
                            rnn.keep_prob: keep_probability,
                            rnn.source_seq_len: source_lengths,
                            rnn.target_seq_len: target_lengths}
                    
                    loss, _ = sess.run([rnn.cost, rnn.train_op],
                                       feed_dict=feed)

                    if batch_i % display_step == 0 and batch_i > 0:
                        
                        train_feed = {rnn.inputs: source_batch,
                                      rnn.keep_prob: 1.0,
                                      rnn.source_seq_len: source_lengths,
                                      rnn.target_seq_len: target_lengths}
                        train_logits = sess.run(rnn.inference_logits,
                                                feed_dict=train_feed)
                        
                        valid_feed = {rnn.inputs: valid_source_batch,
                                      rnn.keep_prob: 1.0,
                                      rnn.source_seq_len: valid_source_lengths,
                                      rnn.target_seq_len: valid_target_lengths}
                        valid_logits = sess.run(rnn.inference_logits,
                                                feed_dict=valid_feed)
                        
                        train_acc = accuracyCalculator.get_accuracy(target_batch, 
                                                                    train_logits)
                        valid_acc = accuracyCalculator.get_accuracy(valid_target_batch,
                                                                    valid_logits)
                        
                        print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'                              .format(epoch_i+1, batch_i, len(source_int_text)//batch_size, 
                                      train_acc, valid_acc, loss))
             
            saver = tf.train.Saver()
            saver.save(sess, save_path)
        
            print('Model Trained and Saved')


# In[23]:


class TrainingValidationSetCreator:
    
    def create_train_val_sets(self, batch_size, source_letter_ids, target_letter_ids):
        """
        Create training and validation sets.
        
        :param batch_size: Batch size
        :param source_letter_ids: Mapping of source text letters to ints
        :param target_letter_ids: Mapping of target text letters to ints
        :return Tuple (train_source, train_target, valid_source, valid_target)
        """
        
        train_source = source_letter_ids[batch_size:]
        train_target = target_letter_ids[batch_size:]
        valid_source = source_letter_ids[:batch_size]
        valid_target = target_letter_ids[:batch_size]

        return (train_source, train_target, valid_source, valid_target)


# In[24]:


class ValidationSetBatchCreator:
    
    def get_val_set_batches(self, 
                            batch_size, 
                            valid_source, 
                            valid_target, 
                            source_pad_int, 
                            target_pad_int):
        """
        Get batches from validation datasets.

        :param batch_size: Batch size
        :param valid_source: Validation source dataset
        :param valid_target: Validation target dataset
        :param source_pad_int: Int ID for <PAD> in source
        :param target_pad_int: Int ID for <PAD> in target
        :return: Tuple (valid_source_batch, valid_target_batch, \
                        valid_source_lengths, valid_target_lengths)
        """

        dataBatchGenerator = DataBatchGenerator()
        
        (valid_source_batch, valid_target_batch,          valid_source_lengths, valid_target_lengths) =             next(dataBatchGenerator.get_batches(batch_size, 
                                                valid_source, 
                                                valid_target,
                                                source_pad_int,
                                                target_pad_int))
        
        return (valid_source_batch, valid_target_batch, 
                valid_source_lengths, valid_target_lengths)


# In[25]:


class DataBatchGenerator:
    
    def get_batches(self, 
                    batch_size, 
                    sources, 
                    targets, 
                    source_pad_int, 
                    target_pad_int):
        """
        Batch targets, sources, and the lengths of their sentences together.
        
        :param batch_size: Batch size
        :param sources: Source dataset
        :param targets: Target datasest
        :param source_pad_int: Int ID for <PAD> in source
        :param target_pad_int: Int ID for <PAD> in target
        :return: Batch generator to yield (pad_source_batch, pad_target_batch, \
                                           pad_source_lengths, pad_target_lengths)
        """
        
        for batch_i in range(0, len(sources)//batch_size):
            start_i = batch_i * batch_size
            
            source_batch = sources[start_i:start_i + batch_size]
            target_batch = targets[start_i:start_i + batch_size]
            
            pad_source_batch = np.array(
                self.pad_sentence_batch(source_batch, source_pad_int))
            pad_target_batch = np.array(
                self.pad_sentence_batch(target_batch, target_pad_int))

            # Need the lengths for the _lengths parameters
            pad_source_lengths = []
            for source in pad_source_batch:
                pad_source_lengths.append(len(source))
                
            pad_target_lengths = []
            for target in pad_target_batch:
                pad_target_lengths.append(len(target))

            yield pad_source_batch, pad_target_batch,                   pad_source_lengths, pad_target_lengths
  
            
    def pad_sentence_batch(self, sentence_batch, pad_int):
        """
        Pad sentences with <PAD> so that each sentence of a batch has the same length.
        
        :param sentence_batch: Batch of sentences
        :param pad_int: Int ID for <PAD>
        :return: Batch of sentences padded with <PAD>
        """
        max_sentence = max([len(sentence) for sentence in sentence_batch])
        return [sentence + [pad_int] * (max_sentence - len(sentence))                     for sentence in sentence_batch]


# In[26]:


class AccuracyCalculator:
    
    def get_accuracy(self, target, logits):
        """
        Calculate accuracy.
        """
        max_seq = max(target.shape[1], logits.shape[1])
        
        if max_seq - target.shape[1]:
            target = np.pad(
                target,
                [(0,0),(0,max_seq - target.shape[1])],
                'constant')
        if max_seq - logits.shape[1]:
            logits = np.pad(
                logits,
                [(0,0),(0,max_seq - logits.shape[1])],
                'constant')

        return np.mean(np.equal(target, logits))


# In[27]:


trainingValidationSetCreator = TrainingValidationSetCreator()

train_source, train_target, valid_source, valid_target =     trainingValidationSetCreator.create_train_val_sets(batch_size, 
                                                       source_int_text, 
                                                       target_int_text)


# In[28]:


source_pad_int = source_vocab_to_int['<PAD>']
target_pad_int = target_vocab_to_int['<PAD>']


# In[29]:


validationSetBatchCreator = ValidationSetBatchCreator()

(valid_source_batch, valid_target_batch,  valid_source_lengths, valid_target_lengths) =     validationSetBatchCreator.get_val_set_batches(
        batch_size, valid_source, valid_target, source_pad_int, target_pad_int)


# In[30]:


save_path = 'checkpoints/dev'


# In[37]:


modelTrainer = ModelTrainer()

modelTrainer.train_seq2seq_model(rnn, train_graph) 


# ## Saving Parameters

# In[38]:


pickleHelper.save_params(save_path)


# ## Checkpoint

# In[39]:


(_, _, 
(source_vocab_to_int, target_vocab_to_int), 
(source_int_to_vocab, target_int_to_vocab)) = pickleHelper.load_preprocessed_data()

load_path = pickleHelper.load_params()


# ## Checking Translation

# In[49]:


class TranslationChecker:
    
    def check_translation(self, 
                          checkpoint, 
                          sentence, 
                          source_vocab_to_int,
                          source_int_to_vocab,
                          target_int_to_vocab):
        """
        Check translation of a sample sentence.
        """
        
        # Convert input sentence into int seq
        inputSentencePreparer = InputSentencePreparer()
        input_seq = inputSentencePreparer.sentence_to_seq(sentence, source_vocab_to_int)
        
        # Get translation logits
        translation_logits = self.get_translation_logits(checkpoint, input_seq)
        
        # Print translation
        self.print_translation(input_seq, 
                               translation_logits, 
                               source_int_to_vocab, 
                               target_int_to_vocab)
        
        
    def get_translation_logits(self, checkpoint, input_seq):
        """
        Load saved model and get output logits.
        
        :param checkpoint: Checkpoint
        :param input_seq: Input sequence
        """
        loaded_graph = tf.Graph()
        
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)
            
            # Load tensors
            inputs = loaded_graph.get_tensor_by_name('input:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            source_seq_len = loaded_graph.get_tensor_by_name('source_seq_len:0')
            target_seq_len = loaded_graph.get_tensor_by_name('target_seq_len:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
            
            # Get translation logits
            feed = {inputs: [input_seq]*batch_size,
                    source_seq_len: [len(input_seq)]*batch_size,
                    target_seq_len: [len(input_seq)*2]*batch_size,
                    keep_prob: 1.0}
            
            translation_logits = sess.run(logits,
                                          feed_dict=feed)[0]
            
            return translation_logits

    
    def print_translation(self, 
                          input_seq, 
                          logits, 
                          source_int_to_vocab, 
                          target_int_to_vocab):
        
        print('Input')
        print('  Word Ids:      {}'.format([i for i in input_seq]))
        print('  English Words: {}'.format([source_int_to_vocab[i] for i in input_seq]))

        print('\nPrediction')
        print('  Word Ids:      {}'.format([i for i in logits]))
        print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in logits])))


# In[46]:


class InputSentencePreparer:
    
    def sentence_to_seq(self, sentence, vocab_to_int):
        """
        Convert a sentence to a sequence of ids.
        
        :param sentence: String
        :param vocab_to_int: Dictionary to go from the words to an id
        :return: List of word ids
        """
        words = sentence.lower().split(' ')
        return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in words]


# In[47]:


sentence = 'he saw a old yellow truck .'


# In[53]:


translationChecker = TranslationChecker()

translationChecker.check_translation(load_path, 
                                     sentence, 
                                     source_vocab_to_int, 
                                     source_int_to_vocab,
                                     target_int_to_vocab)

