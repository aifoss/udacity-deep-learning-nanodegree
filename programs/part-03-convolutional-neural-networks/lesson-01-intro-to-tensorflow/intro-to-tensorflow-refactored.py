
# coding: utf-8

# # Intro to TensorFlow

# In[1]:


import math

import numpy as np
import tensorflow as tf

from pprint import pprint

from tensorflow.examples.tutorials.mnist import input_data
from sklearn import preprocessing


# ## HelloWorld

# In[2]:


class HelloWorld:
    
    def run(self):
        hello_constant = tf.constant('Hello World!')
        
        with tf.Session() as sess:
            output = sess.run(hello_constant)
            print(output)


# In[3]:


helloWorld = HelloWorld()
helloWorld.run()


# ## HelloTensorWorld

# In[4]:


class HelloTensorWorld:
    
    def run(self):
        output = None
        x = tf.placeholder(tf.int32)
        
        with tf.Session() as sess:
            output = sess.run(x, feed_dict={x:123})
            
        return output


# In[5]:


helloTensorWorld = HelloTensorWorld()
output = helloTensorWorld.run()
print(output)


# ## TensorflowMath

# In[6]:


class TensorflowMath:
    
    def add(self, x, y):
        z = tf.add(x, y)
        res = self.run(x, y, z, "{}+{}={}")
        return res
    
    def subtract(self, x, y):
        z = tf.subtract(x, y)
        res = self.run(x, y, z, "{}-{}={}")
        return res
    
    def multiply(self, x, y):
        z = tf.multiply(x, y)
        res = self.run(x, y, z, "{}*{}={}")
        return res

    def divide(self, x, y):
        z = tf.divide(x, y)
        res = self.run(x, y, z, "{}/{}={}")
        return res
    
    def run(self, x, y, z, template):
        with tf.Session() as sess:
            res = sess.run(z)
            print(template.format(x.eval(), y.eval(), res))
        return res


# In[7]:


tensorflowMath = TensorflowMath()

x = tf.constant(5)
y = tf.constant(2)

z = tensorflowMath.add(x, y)
z = tensorflowMath.subtract(x, y)
z = tensorflowMath.multiply(x, y)
z = tensorflowMath.divide(x, y)
z = tensorflowMath.subtract(tf.cast(z, tf.int32), tf.constant(1))


# ## LinearClassifier

# In[8]:


class LinearClassifier:
    
    def get_weights(self, n_features, n_labels):
        """
        Return TensorFlow weights
        :param n_features: Number of features
        :param n_labels: Number of labels
        :return: TensorFlow weights
        """
        return tf.Variable(tf.truncated_normal((n_features, n_labels)))

    def get_biases(self, n_labels):
        """
        Return TensorFlow bias
        :param n_labels: Number of labels
        :return: TensorFlow bias
        """
        return tf.Variable(tf.zeros(n_labels))

    def apply_linear_function(self, input, w, b):
        """
        Return linear function in TensorFlow
        :param input: TensorFlow input
        :param w: TensorFlow weights
        :param b: TensorFlow biases
        :return: TensorFlow linear function
        """
        return tf.add(tf.matmul(input, w), b)


# ## MNISTDataExtractor

# In[9]:


class MNISTDataExtractor:
    
    def get_mnist_features_labels(self, n_labels):
        """
        Gets the first <n> labels from the MNIST dataset
        :param n_labels: Number of labels to use
        :return: Tuple of feature list and label list
        """
        
        mnist_features = []
        mnist_labels = []
        
        mnist = input_data.read_data_sets('./datasets/ud730/mnist', one_hot=True)
        
        # Look at 10000 images
        for mnist_feature, mnist_label in zip(*mnist.train.next_batch(10000)):
            
            # Add features and labels if it's for the first <n>th labels
            if mnist_label[:n_labels].any():
                mnist_features.append(mnist_feature)
                mnist_labels.append(mnist_label[:n_labels])
                
        return mnist_features, mnist_labels


# ## MNISTClassifier

# In[10]:


class MNISTClassifier:
    
    def classify_mnist_images(self, n_features=784, n_labels=3):
        # Features and Labels
        features = tf.placeholder(tf.float32)
        labels = tf.placeholder(tf.float32)
        
        # Weights and Biases
        linearClassifier = LinearClassifier()
        w = linearClassifier.get_weights(n_features, n_labels)
        b = linearClassifier.get_biases(n_labels)
        
        # Linear Function xW + b
        logits = linearClassifier.apply_linear_function(features, w, b)
        
        # Training data
        dataExtractor = MNISTDataExtractor()
        train_features, train_labels = dataExtractor.get_mnist_features_labels(n_labels)
        
        with tf.Session() as sess:
            # Initialize session variables
            sess.run(tf.global_variables_initializer())
            
            # Softmax
            prediction = tf.nn.softmax(logits)
            
            # Cross entropy
            cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
            
            # Training loss
            loss = tf.reduce_mean(cross_entropy)
            
            # Learnign rate
            learning_rate = 0.08
            
            # Graident descent
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            
            # Run optimizer and get loss
            _, train_loss = sess.run([optimizer, loss],
                                      feed_dict={features: train_features,
                                                 labels: train_labels})
            
        print('Loss: {}'.format(train_loss))


# In[11]:


mnistClassifier = MNISTClassifier()
mnistClassifier.classify_mnist_images()


# ## TensorflowSoftmax

# In[12]:


class TensorflowSoftmax:
    
    def run(self, logit_data):
        logits = tf.placeholder(tf.float32)
        
        # Calculate the softmax of the logits
        softmax = tf.nn.softmax(logits)
        
        with tf.Session() as sess:
            # Feed in the logit data
            output = sess.run(softmax, feed_dict={logits: logit_data})
            
        return output


# In[13]:


tensorflowSoftmax = TensorflowSoftmax()

logit_data = [2.0, 1.0, 0.1]
output = tensorflowSoftmax.run(logit_data)
print(output)

logit_data = [1.0, 1.0, 1.0]
output = tensorflowSoftmax.run(logit_data)
print(output)


# ## OneHotEncoder

# In[14]:


class OneHotEncoder:
    
    def one_hot_encode(self, labels):
        lb = preprocessing.LabelBinarizer()
        lb.fit(labels)
        one_hot = lb.transform(labels)
        return one_hot


# In[15]:


oneHotEncoder = OneHotEncoder()
labels = np.array([1,5,3,2,1,4,2,1,3])
one_hot = oneHotEncoder.one_hot_encode(labels)
print(one_hot)


# ## CrossEntropyCalculator

# In[16]:


class CrossEntropyCalculator:
    
    def compute_cross_entropy(self, softmax_data, one_hot_data):
        softmax = tf.placeholder(tf.float32)
        one_hot = tf.placeholder(tf.float32)
        
        cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))
        
        with tf.Session() as sess:
            output = sess.run(cross_entropy, feed_dict={softmax: softmax_data, one_hot: one_hot_data})
        
        return output


# In[17]:


crossEntropyCalculator = CrossEntropyCalculator()

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

cross_entropy = crossEntropyCalculator.compute_cross_entropy(softmax_data, one_hot_data)

print(cross_entropy)


# ## MiniBatcher

# In[18]:


class MiniBatcher:
    
    def get_batches(self, batch_size, features, labels):
        """
        Create batches of features and labels
        :param batch_size: The batch size
        :param features: List of features
        :param labels: List of labels
        :return: Batches of (Features, Labels)
        """
        
        assert len(features) == len(labels)
        
        batches = []
        sample_size = len(features)
        
        for start_i in range(0, sample_size, batch_size):
            end_i = start_i + batch_size
            batch = [features[start_i:end_i], labels[start_i:end_i]]
            batches.append(batch)
        
        return batches


# In[19]:


miniBatcher = MiniBatcher()

batch_size = 3

# 4 Samples of features
example_features = [
    ['F11','F12','F13','F14'],
    ['F21','F22','F23','F24'],
    ['F31','F32','F33','F34'],
    ['F41','F42','F43','F44']]

# 4 Samples of labels
example_labels = [
    ['L11','L12'],
    ['L21','L22'],
    ['L31','L32'],
    ['L41','L42']]

batches = miniBatcher.get_batches(batch_size, example_features, example_labels)

pprint(batches)


# ## MiniBatchMNISTClassifier

# In[20]:


class MiniBatchMNISTClassifier:
    
    def classify_mnist_images(self, n_features=784, n_labels=10, learning_rate=0.001):
        # Import MNIST data
        mnist = input_data.read_data_sets('./datasets/ud730/mnist', one_hot=True)
        
        # Features are already scaled and shuffled
        train_features = mnist.train.images
        test_features = mnist.test.images
        
        train_labels = mnist.train.labels.astype(np.float32)
        test_labels = mnist.test.labels.astype(np.float32)
        
        # Features and Labels
        features = tf.placeholder(tf.float32, [None, n_features])
        labels = tf.placeholder(tf.float32, [None, n_labels])
        
        # Weights, Biases, and Logits
        linearClassifier = LinearClassifier()
        weights = linearClassifier.get_weights(n_features, n_labels)
        biases = linearClassifier.get_biases(n_labels)
        logits = linearClassifier.apply_linear_function(features, weights, biases)
        
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

        # Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Set batch size
        batch_size = 128
        assert batch_size is not None, 'You must set the batch size'
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            # Train optimizer on all batches
            miniBatcher = MiniBatcher()
            batches = miniBatcher.get_batches(batch_size, train_features, train_labels)
            
            for batch_features, batch_labels in batches:
                sess.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})
                
            # Calculate accuracy for test dataset
            test_accuracy = sess.run(accuracy, 
                                     feed_dict={features: test_features, labels: test_labels})

        print('Test Accuracy: {}'.format(test_accuracy))


# In[21]:


miniBatchMNISTClassifier = MiniBatchMNISTClassifier()
miniBatchMNISTClassifier.classify_mnist_images()

