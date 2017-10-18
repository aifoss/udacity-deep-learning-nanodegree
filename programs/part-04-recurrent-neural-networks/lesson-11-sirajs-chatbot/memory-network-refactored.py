
# coding: utf-8

# # Memory Network

# In[1]:


from __future__ import print_function

from functools import reduce

import tarfile
import re

import numpy as np

from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences


# ## Loading Data

# In[2]:


class DataLoader:
    
    def get_file(self):
        try:
            path = get_file('babi-tasks-v1-2.tar.gz', 
                            origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
        except:
            print('Error downloading dataset, please download it manually:\n'
                  '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
                  '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
            raise
        
        tar = tarfile.open(path)

        challenges = {
            # QA1 with 10,000 samples
            'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
            # QA2 with 10,000 samples
            'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
        }

        challenge_type = 'single_supporting_fact_10k'
        challenge = challenges[challenge_type]
        
        return tar, challenge_type, challenge


# In[3]:


print('Loading data ...')

dataLoader = DataLoader()

tar, challenge_type, challenge = dataLoader.get_file()

print('Data loading complete')


# # Preprocessing Data

# In[4]:


class StoryExtractor:
    
    def get_stories(self, f, only_supporting=False, max_length=None):
        """
        Given a file name, read the file,
        retrieve the stories,
        and then convert the sentences into a single story.
        If max_length is supplied,
        any stories longer than max_length tokens will be discarded.
        """
        data = self.parse_stories(f.readlines(), only_supporting=only_supporting)
        flatten = lambda data: reduce(lambda x, y: x + y, data)
        data = [(flatten(story), q, answer) for story, q, answer in data if not max_length                 or len(flatten(story)) < max_length]
        return data
    
    
    def parse_stories(self, lines, only_supporting=False):
        """
        Parse stories provided in the bAbi tasks format
        If only_supporting is true, only the sentences
        that support the answer are kept.
        """
        
        data = []
        story = []
        
        for line in lines:
            line = line.decode('utf-8').strip()
            nid, line = line.split(' ', 1)
            nid = int(nid)
            
            if nid == 1:
                story = []
                
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = self.tokenize(q)
                substory = None
                
                if only_supporting:
                    # Only select the related substory
                    supporting = map(int, supporting.split())
                    substory = [story[i-1] for i in supporting]
                else:
                    # Provide all the substories
                    substory = [x for x in story if x]
                    
                data.append((substory, q, a))
                story.append('')
            else:
                sent = self.tokenize(line)
                story.append(sent)
                
        return data
    
    
    def tokenize(self, sent):
        """
        Return the tokens of a sentence including punctuation.
        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        """
        return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


# In[5]:


print('Extracting stories for the challenge:', challenge_type)

storyExtractor = StoryExtractor()

train_stories = storyExtractor.get_stories(tar.extractfile(challenge.format('train')))
test_stories = storyExtractor.get_stories(tar.extractfile(challenge.format('test')))


# In[6]:


vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))


# In[7]:


print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')


# In[8]:


class SentenceVectorizer:
    
    def vectorize_stories(self, data, word_idx, story_maxlen, query_maxlen):
        """
        Vectorize story sentences.
        """
        X = []
        Xq = []
        Y = []
        
        for story, query, answer in data:
            x = [word_idx[w] for w in story]
            xq = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            y = np.zeros(len(word_idx) + 1)
            y[word_idx[answer]] = 1
            X.append(x)
            Xq.append(xq)
            Y.append(y)
            
        return (pad_sequences(X, maxlen=story_maxlen),
                pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))


# In[9]:


print('Vectorizing the word sequences...')

word_idx = dict((c, i+1) for i, c in enumerate(vocab))

sentenceVectorizer = SentenceVectorizer()

inputs_train, queries_train, answers_train =     sentenceVectorizer.vectorize_stories(train_stories,
                                         word_idx,
                                         story_maxlen,
                                         query_maxlen)
    
inputs_test, queries_test, answers_test =     sentenceVectorizer.vectorize_stories(test_stories,
                                         word_idx,
                                         story_maxlen,
                                         query_maxlen)


# In[10]:


print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')


# ## Building Memory Network

# In[11]:


class MemoryNetworkBuilder:
 
    def create_placeholders(self, story_maxlen, query_maxlen):
        """
        Create placeholders.
        """
        input_sequence = Input((story_maxlen,))
        question = Input((query_maxlen,))
        
        return (input_sequence, question)
    
    
    def build_encoders(self, vocab_size, query_maxlen):
        """
        Build encoders.
        """
        input_encoder_m = self.build_encoder_m(vocab_size)
        input_encoder_c = self.build_encoder_c(vocab_size, query_maxlen)
        question_encoder = self.build_question_encoder(vocab_size, query_maxlen)
        
        return (input_encoder_m, input_encoder_c, question_encoder)
        
        
    def build_encoder_m(self, vocab_size):
        """
        Embed the input sequence into a sequence of vectors.
        """
        input_encoder_m = Sequential()
        input_encoder_m.add(Embedding(input_dim=vocab_size,
                                      output_dim=64))
        input_encoder_m.add(Dropout(0.3))
        
        return input_encoder_m # (samples, story_maxlen, embedding_dim)
    
    
    def build_encoder_c(self, vocab_size, query_maxlen):
        """
        Embed the input into a sequence of vectors of size query_maxlen.
        """
        input_encoder_c = Sequential()
        input_encoder_c.add(Embedding(input_dim=vocab_size,
                                      output_dim=query_maxlen))
        input_encoder_c.add(Dropout(0.3))
        
        return input_encoder_c # (samples, story_maxlen, query_maxlen)
    
    
    def build_question_encoder(self, vocab_size, query_maxlen):
        """
        Embed the question into a sequence of vectors.
        """
        question_encoder = Sequential()
        question_encoder.add(Embedding(input_dim=vocab_size,
                                       output_dim=64,
                                       input_length=query_maxlen))
        question_encoder.add(Dropout(0.3))
        
        return question_encoder # (samples, query_maxlen, embedding_dim)
    
    
    def get_encoder_outputs(self, 
                            input_sequence, 
                            question, 
                            input_encoder_m, 
                            input_encoder_c, 
                            question_encoder):
        """
        Encode input sequence and questions (which are indices)
        to sequences of dense vectors.
        """
        input_encoded_m = input_encoder_m(input_sequence)
        input_encoded_c = input_encoder_c(input_sequence)
        question_encoded = question_encoder(question)
        
        return (input_encoded_m, input_encoded_c, question_encoded)
    
    
    def compute_match(self, input_encoded_m, question_encoded):
        """
        Compute a match between the first input vector sequence
        and the question vector sequence.
        """
        match = dot([input_encoded_m, question_encoded], axes=(2, 2))
        match = Activation('softmax')(match)    
        
        return match # (samples, story_maxlen, query_maxlen)

    
    def get_response(self, match, input_encoded_c):
        """
        Add the match matrix with the second input vector sequence.
        """
        response = add([match, input_encoded_c]) 
        response = Permute((2, 1))(response)     
        
        return response # (samples, query_maxlen, story_maxlen)
    
    
    def get_answer(self, response, question_encoded, vocab_size):
        """
        Concatenate the response matrix with the question vector sequence.
        """
        
        # Concatenate the response matrix with the question vector sequence
        answer = concatenate([response, question_encoded])
        answer = LSTM(32)(answer) # (samples, 32)

        answer = Dropout(0.3)(answer)
        answer = Dense(vocab_size)(answer)
        
        answer = Activation('softmax')(answer)
        
        return answer # (samples, vocab_size)


# In[12]:


class MemoryNetworkModelBuilder:
    
    def build_model(self):
        """
        Build memory network model.
        """
        
        mnBuilder = MemoryNetworkBuilder()
        
        # Create placeholders
        input_sequence, question =             mnBuilder.create_placeholders(story_maxlen, query_maxlen)
        
        # Build encoders and get outputs
        input_encoder_m, input_encoder_c, question_encoder =             mnBuilder.build_encoders(vocab_size, query_maxlen)
            
        input_encoded_m, input_encoded_c, question_encoded =             mnBuilder.get_encoder_outputs(input_sequence, 
                                          question, 
                                          input_encoder_m, 
                                          input_encoder_c, 
                                          question_encoder)
        
        # Get match, response, and answer
        match = mnBuilder.compute_match(input_encoded_m, question_encoded)
        response = mnBuilder.get_response(match, input_encoded_c)
        answer = mnBuilder.get_answer(response, question_encoded, vocab_size)
        
        # Build model
        model = Model([input_sequence, question], answer)
        model.compile(optimizer='rmsprop', 
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model


# In[14]:


print('Building the model...')

modelBuilder = MemoryNetworkModelBuilder()

model = modelBuilder.build_model()

print('Model built')


# ## Training Model

# In[15]:


class ModelTrainer:
    
    def train_model(self, 
                    model, 
                    inputs_train, 
                    queries_train, 
                    answers_train,
                    inputs_test,
                    queries_test,
                    answers_test):
        """
        Train the model.
        """
        model.fit([inputs_train, queries_train], answers_train,
                  batch_size=32,
                  epochs=120,
                  validation_data=([inputs_test, queries_test], answers_test))


# In[16]:


print('Training the model...')

modelTrainer = ModelTrainer()

modelTrainer.train_model(model, 
                         inputs_train, queries_train, answers_train,
                         inputs_test, queries_test, answers_test)

print('Training complete')

