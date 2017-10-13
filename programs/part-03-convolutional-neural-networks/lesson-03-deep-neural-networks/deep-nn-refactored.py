
# coding: utf-8

# # Deep Neural Networks

# In[1]:


import math

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


# ## TwoLayerNeuralNetwork

# In[2]:


class TwoLayerNeuralNetwork:
    
    def build(self, features, hidden_weights, out_weights, n_hidden_classes, n_out_classes):
        self.features = tf.Variable(features)
        
        self.weights = [tf.Variable(hidden_weights),
                        tf.Variable(out_weights)]
        self.biases = [tf.Variable(tf.zeros(n_hidden_classes)),
                       tf.Variable(tf.zeros(n_out_classes))]
        
        self.hidden_layer = tf.add(tf.matmul(self.features, self.weights[0]), self.biases[0])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)
        
        self.logits = tf.add(tf.matmul(self.hidden_layer, self.weights[1]), self.biases[1])


# ## TwoLayerNetworkRunner

# In[3]:


class TwoLayerNetworkRunner:
    
    def run(self, network):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(network.logits)
            return output


# In[4]:


features = [
    [1.0, 2.0, 3.0, 4.0], 
    [-1.0, -2.0, -3.0, -4.0], 
    [11.0, 12.0, 13.0, 14.0]]

hidden_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]

out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

n_hidden_classes = 3
n_out_classes = 2

network = TwoLayerNeuralNetwork()
network.build(features, hidden_weights, out_weights, n_hidden_classes, n_out_classes)

networkRunner = TwoLayerNetworkRunner()
output = networkRunner.run(network)

print(output)


# ## MNISTModel

# In[5]:


class MNISTModel:

    def build(self,
              n_input_h=28, # MNISt data input image height
              n_input_w=28, # MNIST data input image width
              n_input=784,  # MNIST data input (28*28)
              n_classes=10, # output classes (0-9 digits)
              n_hidden=256, # number of hidden-layer features
              learning_rate=0.001):
        
        # Store layer weights and biases
        self.weights = {
            'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])),
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([n_hidden])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }
        
        # Create placeholders
        self.x = tf.placeholder("float", [None, n_input_h, n_input_w, 1])
        self.y = tf.placeholder("float", [None, n_classes])
        self.x_flat = tf.reshape(self.x, [-1, n_input])
        
        # Hidden layer with RELU activation
        self.hidden_layer = tf.add(tf.matmul(self.x_flat, self.weights['hidden']), self.biases['hidden'])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)
        
        # Output layer with linear activation
        self.logits = tf.add(tf.matmul(self.hidden_layer, self.weights['out']), self.biases['out'])
        
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        print("Model Built!")


# ## MNISTModelTrainer

# In[6]:


class MNISTModelTrainer:
    
    def train_model(self, 
                    model, 
                    mnist, 
                    epochs=20, 
                    batch_size=128, 
                    display_step=1,
                    test_size=256):
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(epochs):
                total_batch = int(mnist.train.num_examples//batch_size)
                
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    
                    sess.run(model.optimizer, feed_dict={model.x: batch_x, model.y: batch_y})
                    
                if epoch % display_step == 0:
                    cost = sess.run(model.cost, feed_dict={model.x: batch_x, model.y: batch_y})
                    print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost))
                    
            print("Model Trained!")
            
            correct_prediction = tf.equal(tf.argmax(model.logits, 1), tf.argmax(model.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print("Accuracy:", accuracy.eval({model.x: mnist.test.images[:test_size], 
                                              model.y: mnist.test.labels[:test_size]}))


# In[7]:


mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

print("MNIST data extracted")


# In[8]:


n_input_h = 28        # MNISt data input image height
n_input_w = 28        # MNIST data input image width
n_input = 784         # MNIST data input (28*28)
n_classes = 10        # output classes (0-9 digits)
n_hidden = 256        # number of hidden-layer features
learning_rate = 0.001

model = MNISTModel()
model.build(n_input_h, n_input_w, n_input, n_classes, n_hidden, learning_rate)


# In[9]:


epochs = 20 
batch_size = 128
display_step = 1
test_size = 256

trainer = MNISTModelTrainer()
trainer.train_model(model, mnist, epochs, batch_size, display_step, test_size)


# ## VariableSaver

# In[16]:


class VariableSaver:
    
    def save_variables(self):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print('Weights:\n{}'.format(sess.run(weights)))
            print('Bias:\n{}\n'.format(sess.run(bias)))
            
            saver.save(sess, save_file)
            
            print('Variables saved')


# In[17]:


variableSaver = VariableSaver()

save_file='./model.ckpt'

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

variableSaver.save_variables()


# ## VariableRestorer

# In[18]:


class VariableRestorer:
    
    def load_variables(self):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, save_file)
            
            print('Weights:\n{}'.format(sess.run(weights)))
            print('Bias:\n{}\n'.format(sess.run(bias)))
            
            print('Variables loaded')


# In[19]:


variableRestorer = VariableRestorer()
variableRestorer.load_variables()


# ## NamedVariableSaveDemo

# In[14]:


class NamedVariableSaveDemo:
    
    def run(self):
        tf.reset_default_graph()
        
        weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
        bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
        
        print('Save Weights as: {}'.format(weights.name))
        print('Save Bias as: {}'.format(bias.name))
        print('')
        
        saver = tf.train.Saver()
        save_file = 'model.ckpt'
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            print('Save Weight:\n{}'.format(sess.run(weights)))
            print('Save Bias:\n{}'.format(sess.run(bias)))
            print('')
            
            saver.save(sess, save_file)
            
        # Remove the previous weights and bias
        tf.reset_default_graph()
        
        bias = tf.Variable(tf.truncated_normal([3]), name='bias_0')
        weights = tf.Variable(tf.truncated_normal([2, 3]), name='weights_0')
        
        print('Load Weights as: {}'.format(weights.name))
        print('Load Bias as: {}'.format(bias.name))
        print('')
        
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, save_file)
            
            print('Load Weight:\n{}'.format(sess.run(weights)))
            print('Load Bias:\n{}'.format(sess.run(bias)))
            print('')
            
        print('Loaded weights and bias successfully')


# In[15]:


NamedVariableSaveDemo().run()


# ## MNISTModel2

# In[20]:


class MNISTModel2:
    
    def build(self, learning_rate=0.001, n_input=784, n_classes=10):
        self.features = tf.placeholder(tf.float32, [None, n_input])
        self.labels = tf.placeholder(tf.float32, [None, n_classes])
        
        self.weights = tf.Variable(tf.random_normal([n_input, n_classes]))
        self.bias = tf.Variable(tf.random_normal([n_classes]))
        
        self.logits = tf.add(tf.matmul(self.features, self.weights), self.bias)
        
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, 
                                                                labels=self.labels)
        self.cost = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)                .minimize(self.cost)
            
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        
        print('Model Built!')


# ## MNISTModelTrainer2

# In[21]:


class MNISTModelTrainer2:
    
    def train_model(self, model, mnist, epochs=100, batch_size=128, save_file='./train_model.ckpt'):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(epochs):
                n_batches = math.ceil(mnist.train.num_examples//batch_size)
                
                for i in range(n_batches):
                    batch_features, batch_labels = mnist.train.next_batch(batch_size)
                    sess.run(model.optimizer,
                             feed_dict={model.features: batch_features,
                                        model.labels: batch_labels})
                    
                if epoch % 10 == 0:
                    valid_accuracy = sess.run(model.accuracy,
                                              feed_dict={model.features: mnist.validation.images,
                                                         model.labels: mnist.validation.labels})
                    
                    print('Epoch {:<3} - Validation Accuracy: {}'.format(epoch, valid_accuracy))
             
            saver.save(sess, save_file)
                          
            print('Trained Model Saved!')


# ## MNISTModelLoader

# In[22]:


class MNISTModelLoader:
    
    def load_model(self, save_file='./train_model.ckpt'):
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            saver.restore(sess, save_file)
            
            test_accuracy = sess.run(model.accuracy,
                                     feed_dict={model.features: mnist.test.images,
                                                model.labels: mnist.test.labels})
        print('Test Accuracy: {}'.format(test_accuracy))


# In[23]:


# Remove previous Tensors and Operations
tf.reset_default_graph()


# In[24]:


model = MNISTModel2()
model.build()


# In[25]:


# Import MNIST data
mnist = input_data.read_data_sets('.', one_hot=True)

modelTrainer = MNISTModelTrainer2()
modelTrainer.train_model(model, mnist)


# In[26]:


modelLoader = MNISTModelLoader()
modelLoader.load_model()


# ## ModelWithDropout

# In[27]:


class ModelWithDropout:
    
    def build(self, 
              features, 
              hidden_weights, 
              out_weights, 
              n_hidden_classes, 
              n_out_classes):
        
        self.features = tf.Variable(features)
        
        self.weights = [tf.Variable(hidden_weights), 
                        tf.Variable(out_weights)]
        self.biases = [tf.Variable(tf.zeros(n_hidden_classes)), 
                       tf.Variable(tf.zeros(n_out_classes))]
        
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.hidden_layer = tf.add(tf.matmul(self.features, self.weights[0]), self.biases[0])
        self.hidden_layer = tf.nn.relu(self.hidden_layer)
        self.hidden_layer = tf.nn.dropout(self.hidden_layer, self.keep_prob)
        
        self.logits = tf.add(tf.matmul(self.hidden_layer, self.weights[1]), self.biases[1])
        
        print('Model with Dropout Built!')


# ## ModelWithDropoutRunner

# In[28]:


class ModelWithDropoutRunner:
    
    def run_model(self, model, keep_prob=0.5):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(model.logits, feed_dict={model.keep_prob: keep_prob})
            print('Model Output:\n{}'.format(output))


# In[29]:


features = [
    [0.0, 2.0, 3.0, 4.0], 
    [0.1, 0.2, 0.3, 0.4], 
    [11.0, 12.0, 13.0, 14.0]]

hidden_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]

out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

n_hidden_classes = 3
n_out_classes = 2

model = ModelWithDropout()
model.build(features, hidden_weights, out_weights, n_hidden_classes, n_out_classes)


# In[32]:


modelRunner = ModelWithDropoutRunner()
modelRunner.run_model(model)

