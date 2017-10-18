
# coding: utf-8

# # Seq2Seq RNN

# In[1]:


import os
import time
import pickle

from distutils.version import LooseVersion

import numpy as np
import tensorflow as tf

from tensorflow.python.layers.core import Dense


# ## Loading, Preprocessing, and Exploring Data

# In[2]:


class DataLoader:
    
    def load_data(self, data_dir):
        """
        Load data from the data directory.
        """
        input_file = os.path.join(data_dir)
        
        with open(input_file, "r", encoding='utf-8', errors='ignore') as f:
            data = f.read()
            
        return data


# In[3]:


class DataPreprocessor:
 
    def extract_character_vocab(self, data):
        """
        Extract vocabulary from the data and create lookup dictionaries.
        """
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

        word_set = set([character for line in data.split('\n') for character in line])
        int_to_vocab = {word_i: word for word_i, word in enumerate(special_words + list(word_set))}
        vocab_to_int = {word: word_i for word_i, word in int_to_vocab.items()}

        return (int_to_vocab, vocab_to_int)
    
    
    def convert_characters_to_ids(self, sentences, letter_to_int, addEOS):
        """
        Convert characters in sentences to integers.
        """
        lines = sentences.split('\n')
        
        if addEOS == False:
            letter_ids =                 [[letter_to_int.get(letter, letter_to_int['<UNK>']) for letter in line]                          for line in lines]
        else:
            letter_ids =                 [[letter_to_int.get(letter, letter_to_int['<UNK>']) for letter in line]                         + [letter_to_int['<EOS>']]                     for line in lines] 
        
        return letter_ids


# In[4]:


class DataExplorer:
    
    def explore_sentences(self, sentences):
        print(sentences[:50].split('\n'))
        
    def explore_letter_ids(self, letter_ids):
        print(letter_ids[:3])


# In[5]:


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


# In[6]:


source_path = 'data/letters_source.txt'
target_path = 'data/letters_target.txt'


# In[7]:


dataLoader = DataLoader()


# In[8]:


source_sentences = dataLoader.load_data(source_path)
target_sentences = dataLoader.load_data(target_path)


# In[9]:


dataExplorer = DataExplorer()


# In[10]:


dataExplorer.explore_sentences(source_sentences)


# In[11]:


dataExplorer.explore_sentences(target_sentences)


# In[12]:


dataPreprocessor = DataPreprocessor()


# In[13]:


source_int_to_letter, source_letter_to_int =     dataPreprocessor.extract_character_vocab(source_sentences)

target_int_to_letter, target_letter_to_int =     dataPreprocessor.extract_character_vocab(target_sentences)


# In[14]:


source_letter_ids =     dataPreprocessor.convert_characters_to_ids(source_sentences, source_letter_to_int, False)

target_letter_ids =     dataPreprocessor.convert_characters_to_ids(target_sentences, target_letter_to_int, True)


# In[15]:


dataExplorer.explore_letter_ids(source_letter_ids)


# In[16]:


dataExplorer.explore_letter_ids(target_letter_ids)


# In[17]:


pickleHelper = PickleHelper()


# In[18]:


pickleHelper.save_preprocessed_data(
    (source_int_to_letter, source_letter_to_int, source_letter_ids,
     target_int_to_letter, target_letter_to_int, target_letter_ids))


# ## Checkpoint

# In[19]:


(source_int_to_letter, source_letter_to_int, source_letter_ids,
 target_int_to_letter, target_letter_to_int, target_letter_ids) = \
    pickleHelper.load_preprocessed_data()        


# ## Checking TensorFlow Version

# In[20]:


assert LooseVersion(tf.__version__) >= LooseVersion('1.1'),     'Please use TensorFlow version 1.1 or newer'

print('TensorFlow Version: {}'.format(tf.__version__))


# ## Building Seq2Seq RNN Model

# In[21]:


class RNN:
    
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.lr = None
        
        self.source_seq_len = None
        self.target_seq_len = None
        self.target_max_seq_len = None
        
        self.encoder_output = None
        self.encoder_state = None
        
        self.decoder_input = None
        self.training_decoder_output = None
        self.inference_decoder_output = None
        
        self.cost = None
        self.train_op = None


# In[22]:


class RNNBuilder:
    
    def create_placeholders(self):
        """
        Create placeholders.
        
        :return: Tuple (inputs, targets, lr, source_seq_len, target_seq_len, target_max_seq_len)
        """
        
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        
        source_seq_len = tf.placeholder(tf.int32, (None,), name='source_seq_len')
        target_seq_len = tf.placeholder(tf.int32, (None,), name='target_seq_len')
        target_max_seq_len = tf.reduce_max(target_seq_len, name='target_max_seq_len')
        
        return (inputs, targets, lr, source_seq_len, target_seq_len, target_max_seq_len)

        
    def build_encoding_layer(self, 
                             inputs, 
                             rnn_size, 
                             num_layers, 
                             source_vocab_size, 
                             source_seq_len,
                             enc_embed_size):
        """
        Build the encoding layer.
        
        :param inputs: Placeholder for inputs
        :param rnn_size: RNN size
        :param num_layers: Number of RNN layers
        :param source_vocab_size: Source vocab size
        :param source_seq_len: Source sequence length
        :param enc_embed_size: Encoding embedding dimension
        :return: Tuple (enc_output, enc_state)
        """
        
        with tf.variable_scope("encode"):
            # Encodder embedding
            enc_embed = tf.contrib.layers.embed_sequence(inputs,
                                                         source_vocab_size,
                                                         enc_embed_size)

            # Encoder cell
            def make_cell(rnn_size):
                initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2)
                dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=initializer)
                return dec_cell
            
            enc_cell = tf.contrib.rnn.MultiRNNCell(
                [make_cell(rnn_size) for _ in range(num_layers)])

            enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, 
                                                      enc_embed, 
                                                      sequence_length=source_seq_len,
                                                      dtype=tf.float32)

            return (enc_output, enc_state)
 
    
    def format_decoder_input(self, targets, target_letter_to_int, batch_size):
        """
        Process the input we'll feed to the decoder.
        Remove the last word id from each batch 
        and concatenate <GO> to the beginning of each batch.
        
        :param targets: Placeholder for targets
        :param target_letter_to_int: Mapping of target letters to ints
        :param batch_size: Batch size
        :return: Input to the decoder
        """
        ending = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
        dec_input = tf.concat([tf.fill([batch_size, 1], target_letter_to_int['<GO>']), ending], 1)        
        return dec_input
    
    
    def build_decoding_layer(self,
                             rnn_size,
                             num_layers,
                             batch_size,
                             target_vocab_size,
                             target_letter_to_int,
                             target_seq_len,
                             target_max_seq_len,
                             enc_state,
                             dec_input,
                             dec_embed_size):
        """
        Build the decoding layer.
        
        :param rnn_size: RNN size
        :param num_layers: Number of layers
        :param batch_size: Batch size
        :param target_vocab_size: Target vocabulary size
        :param target_letter_to_int: Mapping of target letters to ints
        :param target_seq_len: Target sequenge length
        :param target_max_seq_len: Max target sequence length
        :param enc_state: Encoding layer state
        :param dec_input: Input to the decoder
        :param dec_embed_size: Decoding embedding dimension
        :return Tuple (training_decoder_output, inference_decoder_output)
        """
        
        # Decoder embedding
        dec_embedding = tf.Variable(
            tf.random_uniform([target_vocab_size, dec_embed_size]))
        dec_embed = tf.nn.embedding_lookup(dec_embedding, dec_input)

        # Decoder cell
        def make_cell(rnn_size):
            initializer = tf.random_uniform_initializer(-0.1, 0.1, seed=2)
            dec_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                               initializer=initializer)
            return dec_cell

        dec_cell = tf.contrib.rnn.MultiRNNCell(
            [make_cell(rnn_size) for _ in range(num_layers)])

        # Dense layer to translate the decoder's output at each time step
        # into a chocie from the target vocabulary
        initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=initializer)

        # Training decoder
        training_decoder_output = self.build_training_decoder(target_seq_len,
                                                              target_max_seq_len,
                                                              enc_state, 
                                                              dec_embed, 
                                                              dec_cell, 
                                                              output_layer)

        # Inference decoder
        inference_decoder_output = self.build_inference_decoder(batch_size,
                                                                target_letter_to_int,
                                                                target_max_seq_len,
                                                                enc_state, 
                                                                dec_embedding,
                                                                dec_cell,
                                                                output_layer)

        return (training_decoder_output, inference_decoder_output)
 

    def build_training_decoder(self, 
                               target_seq_len, 
                               target_max_seq_len,
                               enc_state, 
                               dec_embed, 
                               dec_cell, 
                               output_layer):
        """
        Build the training decoder.
        
        :param target_seq_len: Target sequence length
        :param target_max_seq_len: Max target sequence length
        :param enc_state: Encoder state
        :param dec_embed: Decoder embed input
        :param dec_cell: Decoder cell
        :param output_layer: Output layer
        :return: Output from the training decoder
        """
        
        with tf.variable_scope("decode"):
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed,
                                                       sequence_length=target_seq_len,
                                                       time_major=False)
            
            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state, output_layer)
            
            return tf.contrib.seq2seq.dynamic_decode(decoder,
                                                     impute_finished=True,
                                                     maximum_iterations=target_max_seq_len)[0]
            
       
    def build_inference_decoder(self,
                                batch_size,
                                target_letter_to_int,
                                target_max_seq_len,
                                enc_state, 
                                dec_embedding,
                                dec_cell,
                                output_layer):
        """
        Build the inference decoder.
        Reuse the same parameters trained by the training decoder.
        
        :param batch_size: Batch size
        :param target_letter_to_int: Mapping of target letters to ints
        :param enc_state: Encoder state
        :param dec_embedding: Placeholder for decoder embdding
        :param dec_cell: Decoder cell
        :param output_layer: Output layer
        :return: Output from the inference decoder
        """
        
        with tf.variable_scope("decode", reuse=True):
            start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']], dtype=tf.int32),
                                   [batch_size],
                                   name='start_tokens')

            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embedding,
                                                              start_tokens,
                                                              target_letter_to_int['<EOS>'])

            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, helper, enc_state, output_layer)
            
            return tf.contrib.seq2seq.dynamic_decode(decoder,
                                                     impute_finished=True,
                                                     maximum_iterations=target_max_seq_len)[0]


# In[23]:


class Seq2SeqGraphBuilder:
    
    def build_train_graph(self, 
                          rnn_size, 
                          num_layers,
                          batch_size,
                          source_vocab_size,
                          target_vocab_size,
                          target_letter_to_int, 
                          encoding_embedding_size, 
                          decoding_embedding_size):
        
        """
        Build the training graph hooking up the Seq2Seq model with the optimizer.
        """
        
        train_graph = tf.Graph()
        
        rnn = RNN()
        rnnBuilder = RNNBuilder()
        optimizerTuner = OptimizerTuner()

        # Set the graph to default to ensure that it is ready for training
        with train_graph.as_default():
            
            # Create placeholders
            inputs, targets, lr, source_seq_len, target_seq_len, target_max_seq_len =                 rnnBuilder.create_placeholders()
            rnn.inputs, rnn.targets, rnn.lr = inputs, targets, lr
            rnn.source_seq_len, rnn.target_seq_len, rnn.target_max_seq_len =                 source_seq_len, target_seq_len, target_max_seq_len
            
            # Building the encoding layer
            enc_output, enc_state =                 rnnBuilder.build_encoding_layer(inputs,
                                                rnn_size, 
                                                num_layers, 
                                                source_vocab_size,
                                                source_seq_len,
                                                encoding_embedding_size)
            rnn.encoder_output, rnn.encoder_state = enc_output, enc_state
            
            # Format the decoder input
            dec_input = rnnBuilder.format_decoder_input(targets, target_letter_to_int, batch_size)
            rnn.decoder_input = dec_input
            
            # Build the decoding layer
            training_decoder_output, inference_decoder_output =                 rnnBuilder.build_decoding_layer(rnn_size,
                                                num_layers,
                                                batch_size,
                                                target_vocab_size,
                                                target_letter_to_int,
                                                target_seq_len,
                                                target_max_seq_len,
                                                enc_state,
                                                dec_input,
                                                decoding_embedding_size)
            rnn.training_decoder_output, rnn.inference_decoder_output =                 training_decoder_output, inference_decoder_output
            
            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(training_decoder_output.rnn_output, name='logits')
            inference_logits = tf.identity(inference_decoder_output.sample_id, name='predictions')
            
            # Create weights for sequence loss
            masks = tf.sequence_mask(target_seq_len,
                                     target_max_seq_len,
                                     dtype=tf.float32,
                                     name='masks')

            with tf.variable_scope("optimization"):    
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)
 
                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)
                train_op = optimizerTuner.get_gradient_clipped_optimizer(optimizer, cost)
            
                rnn.cost, rnn.train_op = cost, train_op
                
            return rnn, train_graph


# In[24]:


class OptimizerTuner:
    
    def get_gradient_clipped_optimizer(self, optimizer, cost):
        """
        Apply gradient clipping to optimizer.
        
        :param optimizer: Optimizer to apply gradient clipping to
        :param cost: Loss function
        :return: Optimizer with gradient clipping
        """
        
        gradients = optimizer.compute_gradients(cost)
        
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)                             for grad, var in gradients if grad is not None]
        
        train_op = optimizer.apply_gradients(capped_gradients)
        
        return train_op


# In[25]:


rnn_size = 50
num_layers = 2
batch_size = 128
source_vocab_size = len(source_letter_to_int)
target_vocab_size = len(target_letter_to_int)
encoding_embedding_size = 15
decoding_embedding_size = 15


# In[26]:


tf.reset_default_graph()


# In[27]:


graphBuilder = Seq2SeqGraphBuilder()


# In[28]:


rnn, train_graph = graphBuilder.build_train_graph(rnn_size, 
                                                  num_layers,
                                                  batch_size,
                                                  source_vocab_size,
                                                  target_vocab_size,
                                                  target_letter_to_int, 
                                                  encoding_embedding_size, 
                                                  decoding_embedding_size)


# ## Training Seq2Seq Model

# In[29]:


class ModelTrainer:
    
    def train_seq2seq_model(self,
                            rnn,
                            train_graph, 
                            epochs, 
                            learning_rate, 
                            display_step, 
                            checkpoint):
        """
        Train the Seq2Seq model.
        
        :param rnn: Seq2Seq RNN model
        :param train_graph: Tensorflow graph
        :param epochs: Number of epochs
        :param learning_rate: Learning rate
        :param display_step: Interval for displaying debug message
        :param checkpoint: Location where to save model
        """
        
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())
            
            batchGenerator = DataBatchGenerator()
            
            for epoch_i in range(1, epochs+1):
                batches = batchGenerator.get_batches(batch_size,
                                                     train_source,
                                                     train_target,
                                                     source_pad_int,
                                                     target_pad_int)
                
                for batch_i, (source_batch, target_batch, source_lengths, target_lengths)                         in enumerate(batches):
                    
                    # Training step
                    feed = {rnn.inputs: source_batch,
                            rnn.targets: target_batch,
                            rnn.lr: learning_rate,
                            rnn.source_seq_len: source_lengths,
                            rnn.target_seq_len: target_lengths}
                    
                    loss, _ = sess.run([rnn.cost, rnn.train_op],
                                       feed_dict=feed)
                    
                    # Debug message
                    if batch_i % display_step == 0 and batch_i > 0:
                        
                        # Calculate validation cost
                        feed = {rnn.inputs: valid_source_batch,
                                rnn.targets: valid_target_batch,
                                rnn.lr: learning_rate,
                                rnn.source_seq_len: valid_source_lengths,
                                rnn.target_seq_len: valid_target_lengths}
                        
                        validation_loss = sess.run([rnn.cost],
                                                    feed_dict=feed)
                        
                        print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(train_source) // batch_size, 
                                  loss, 
                                  validation_loss[0]))
                        
            # Save model
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)
            
            print('\nModel Trained and Saved\n')


# In[30]:


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


# In[31]:


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


# In[32]:


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


# In[33]:


trainingValidationSetCreator = TrainingValidationSetCreator()


# In[34]:


train_source, train_target, valid_source, valid_target =     trainingValidationSetCreator.create_train_val_sets(batch_size, 
                                                       source_letter_ids, 
                                                       target_letter_ids)


# In[35]:


source_pad_int = source_letter_to_int['<PAD>']
target_pad_int = target_letter_to_int['<PAD>']


# In[36]:


validationSetBatchCreator = ValidationSetBatchCreator()


# In[37]:


(valid_source_batch, valid_target_batch,  valid_source_lengths, valid_target_lengths) =     validationSetBatchCreator.get_val_set_batches(
        batch_size, valid_source, valid_target, source_pad_int, target_pad_int)


# In[41]:


epochs = 60
learning_rate = 0.001
display_step = 20
checkpoint = "best_model.ckpt" 


# In[42]:


modelTrainer = ModelTrainer()


# In[43]:


modelTrainer.train_seq2seq_model(rnn,
                                 train_graph, 
                                 epochs, 
                                 learning_rate, 
                                 display_step, 
                                 checkpoint)


# ## Checking Prediction

# In[44]:


class PredictionChecker:
    
    def check_prediction(self, 
                         checkpoint, 
                         text,
                         source_letter_to_int,
                         target_letter_to_int):
        """
        Check sample predictions from the trained model.
        
        :param checkpoint: Checkpoint where the model is saved
        :param text: Input text
        :param source_letter_to_int: Mapping of source letters to ints
        :param target_letter_to_int: mapping of target letters to ints
        """
        
        loaded_graph = tf.Graph()
        
        answer_logits = self.get_answer_logits(text, loaded_graph, checkpoint)
        
        self.print_prediction(answer_logits, text, source_letter_to_int, target_letter_to_int)
        
    
    def get_answer_logits(self, text, loaded_graph, checkpoint):
        """
        Run trainded model with given input text.
        
        :param text: Input text
        :param loaded_graph: Loaded TensorFlow graph of the trained model
        :param checkpoint: Checkpoint name
        :return: Logits from running the model
        """
    
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)
            
            # Load tensors
            inputs = loaded_graph.get_tensor_by_name('inputs:0')
            logits = loaded_graph.get_tensor_by_name('predictions:0')
            source_seq_len = loaded_graph.get_tensor_by_name('source_seq_len:0')
            target_seq_len = loaded_graph.get_tensor_by_name('target_seq_len:0')
            
            # Multiply by batch_size to match the model's input parameters
            feed = {inputs: [text]*batch_size,
                    source_seq_len: [len(text)]*batch_size,
                    target_seq_len: [len(text)]*batch_size}
            
            answer_logits = sess.run(logits,
                                     feed_dict=feed)[0]
            
            return answer_logits

    
    def print_prediction(self, answer_logits, text, source_letter_to_int, target_letter_to_int):
        """
        Print prediction.
        
        :param answer_logits: Logits from the model
        :param text: Input text
        :param source_letter_to_int: Mapping of source letters to ints
        :param target_letter_to_int: mapping of target letters to ints
        """
    
        pad = source_letter_to_int["<PAD>"] 

        print('Original Text:', input_sentence)

        print('\nSource')
        print('  Word Ids:    {}'.format([i for i in text]))
        print('  Input Words: {}'.format(
            " ".join([source_int_to_letter[i] for i in text])))

        print('\nTarget')
        print('  Word Ids:       {}'.format(
            [i for i in answer_logits if i != pad]))
        print('  Response Words: {}'.format(
            " ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))


# In[45]:


class InputSentencePreparer:

    def source_to_seq(self, text, source_letter_to_int):
        """
        Prepare the text for the model.
        
        :param: Input text
        :param source_letter_to_int: Mapping of source letters to ints
        :return: Prepared input sentence to feed to the model
        """
        seq_len = 7
        return [source_letter_to_int.get(word, source_letter_to_int['<UNK>'])                 for word in text]+ [source_letter_to_int['<PAD>']]*(seq_len-len(text))


# In[46]:


input_sentence = 'hello'


# In[47]:


inputSentencePreparer = InputSentencePreparer()


# In[48]:


text = inputSentencePreparer.source_to_seq(input_sentence, source_letter_to_int)


# In[49]:


predictionChecker = PredictionChecker()


# In[50]:


predictionChecker.check_prediction(checkpoint, 
                                   text,
                                   source_letter_to_int,
                                   target_letter_to_int)

