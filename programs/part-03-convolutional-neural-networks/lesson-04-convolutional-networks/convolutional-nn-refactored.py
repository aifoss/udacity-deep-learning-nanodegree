
# coding: utf-8

# # Convolutional Neural Networks

# In[1]:


import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# ## ConvLayerOutputShapeCalculator

# In[2]:


class ConvLayerOutputShapeCalculator:
    
    def compute_conv_output_shape(self, 
                                  input_shape, 
                                  filter_shape, 
                                  n_filters, 
                                  stride, 
                                  padding_size):

        print('Input Shape: {}'.format(input_shape))
        print('Filter Shape: {}'.format(filter_shape))
        print('Num Filters: {}'.format(n_filters))
        print('')
        
        conv_h = self.compute_conv_height(input_shape[0], filter_shape[0], padding_size, stride)
        conv_w = self.compute_conv_width(input_shape[1], filter_shape[1], padding_size, stride)
        conv_d = n_filters
        
        print('Conv Height: {}'.format(conv_h))
        print('Conv Width: {}'.format(conv_w))
        print('Conv Depth: {}'.format(conv_d))
        print('')
        
        return (conv_h, conv_w, conv_d)
    
    def compute_conv_height(self, input_h, filter_h, p, stride):
        return int((input_h - filter_h + 2*p) / stride) + 1
    
    def compute_conv_width(self, input_w, filter_w, p, stride):
        return int((input_w - filter_w + 2*p) / stride) + 1


# In[4]:


input_shape = [32, 32, 3]
filter_shape = [8, 8, 3]
n_filters = 20
stride = 2
padding_size = 1

calculator = ConvLayerOutputShapeCalculator()
output_shape = calculator.compute_conv_output_shape(input_shape, 
                                                    filter_shape, 
                                                    n_filters, 
                                                    stride, 
                                                    padding_size)

print('Conv Output Shape: {}'.format(output_shape))


# ## ConvLayerOutputShapeDemo

# In[5]:


class ConvLayerOutputShapeDemo:
    
    def run(self, input_shape, filter_shape, n_filters, stride, padding='SAME'):        
        input = tf.placeholder(tf.float32,
                               (None, input_shape[0], input_shape[1], input_shape[2]))
        
        print('Input Shape: {}'.format(input.shape))
        print('Filter Shape: {}'.format((filter_shape[0], filter_shape[1])))
        print('Output Depth: {}'.format(n_filters))
        print('')
        
        conv_weights = tf.Variable(tf.truncated_normal(
            (filter_shape[0], filter_shape[1], input_shape[2], n_filters)))
        conv_bias = tf.Variable(tf.zeros(n_filters))
        
        print('Conv Weights Shape: {}'.format(conv_weights.shape))
        print('Conv Bias Shape: {}'.format(conv_bias.shape))
        print('')
        
        conv_strides = [1, stride, stride, 1] # (batch, height, width, depth)
        
        print('Conv Strides: {}'.format(conv_strides))
        print('')
        
        conv = tf.nn.conv2d(input, conv_weights, conv_strides, padding) + conv_bias
        
        print('Conv Output Shape ({}): {}'.format(padding, conv.shape))


# In[6]:


demo = ConvLayerOutputShapeDemo()
demo.run(input_shape, filter_shape, n_filters, stride, padding='VALID')


# In[7]:


demo.run(input_shape, filter_shape, n_filters, stride, padding='SAME')


# ## ConvLayerParamNumCalculator

# In[10]:


class ConvLayerParamNumCalculator:
    
    def compute_num_params_no_param_sharing(self,
                                            input_shape,
                                            filter_shape,
                                            n_filters,
                                            stride,
                                            p_size):
        
        num_filter_neurons = self.compute_num_filter_neurons(filter_shape)
        num_output_neurons = self.compute_num_output_neurons(input_shape, 
                                                             filter_shape, 
                                                             n_filters, 
                                                             stride, 
                                                             p_size)
        return int(num_filter_neurons * num_output_neurons)
    
    def compute_num_params_with_param_sharing(self,
                                              input_shape,
                                              filter_shape,
                                              n_filters):
        
        num_filter_neurons = self.compute_num_filter_neurons(filter_shape)
        num_channels = n_filters
        
        return num_filter_neurons * num_channels
        
    def compute_num_filter_neurons(self, filter_shape):
        num_weights = filter_shape[0] * filter_shape[1] * filter_shape[2]
        num_biases = 1
        return num_weights + num_biases
    
    def compute_num_output_neurons(self, input_shape, filter_shape, n_filters, stride, p_size):
        output_shape = ConvLayerOutputShapeCalculator()            .compute_conv_output_shape(input_shape, 
                                       filter_shape,
                                       n_filters,
                                       stride,
                                       p_size)
        return output_shape[0] * output_shape[1] * output_shape[2]


# In[11]:


calculator = ConvLayerParamNumCalculator()

num_params = calculator.compute_num_params_no_param_sharing(input_shape, 
                                                            filter_shape, 
                                                            n_filters,
                                                            stride, 
                                                            padding_size)

print('Number of Parameters w/o Parameter Sharing: {}\n'.format(num_params))

num_params = calculator.compute_num_params_with_param_sharing(input_shape,
                                                              filter_shape,
                                                              n_filters)

print('Number of Paramters with Parameter Sharing: {}\n'.format(num_params))


# ## PoolLayerOutputShapeCalculator

# In[12]:


class PoolLayerOutputShapeCalculator:
    
    def compute_output_shape(self, input_shape, filter_shape, stride):
        output_h = self.compute_output_height(input_shape, filter_shape, stride)
        output_w = self.compute_output_width(input_shape, filter_shape, stride)
        output_d = input_shape[2]
        return (output_h, output_w, output_d)
        
    def compute_output_height(self, input_shape, filter_shape, stride):
        return int((input_shape[0]-filter_shape[0])/stride) + 1
    
    def compute_output_width(self, input_shape, filter_shape, stride):
        return int((input_shape[1]-filter_shape[1])/stride) + 1


# In[13]:


input_shape = [4, 4, 5]
filter_shape = [2, 2]
stride = 2

calculator = PoolLayerOutputShapeCalculator()
output_shape = calculator.compute_output_shape(input_shape, filter_shape, stride)

print('Pool Layer Output Shape: {}'.format(output_shape))


# ## PoolLayerOutputShapeDemo

# In[14]:


class PoolLayerOutputShapeDemo:
    
    def run(self, input_shape, filter_shape, stride, padding='SAME'):
        input = tf.placeholder(tf.float32, (None, *input_shape))
        
        print('Input Shape: {}'.format(input.shape))
        print('')
        
        filters = [1, *filter_shape, 1]
        strides = [1, stride, stride, 1]
        
        print('Filters: {}'.format(filters))
        print('Strides: {}'.format(strides))
        print('')
        
        pool = tf.nn.max_pool(input, filters, strides, padding)
        
        print('Pool Output Shape ({}): {}'.format(padding, pool.shape))


# In[15]:


demo = PoolLayerOutputShapeDemo()
demo.run(input_shape, filter_shape, stride, 'SAME')


# In[16]:


demo.run(input_shape, filter_shape, stride, 'VALID')


# ## ConvNetwork

# In[17]:


class ConvNetwork:
    
    def build(self, 
              input_shape,
              conv_1_filter_shape,
              conv_1_n_outputs,
              conv_2_filter_shape,
              conv_2_n_outputs,
              fc_dim,
              fc_n_outputs,
              out_n_classes,
              dropout,
              learning_rate):
        
        self.x, self.y, self.keep_prob = self.create_placeholders(input_shape, out_n_classes)
        
        self.weights, self.biases = self.create_weights_and_biases(conv_1_filter_shape,
                                                                   conv_1_n_outputs,
                                                                   conv_2_filter_shape,
                                                                   conv_2_n_outputs,
                                                                   fc_dim,
                                                                   fc_n_outputs,
                                                                   out_n_classes)
        
        self.logits = self.build_layers(self.x, self.weights, self.biases, self.keep_prob)
        
        self.cost, self.optimizer = self.define_cost_and_optimizer(self.logits, 
                                                                   self.y, 
                                                                   learning_rate)
        
        self.accuracy = self.define_accuracy(self.logits, self.y)
        
    def create_placeholders(self, input_shape, out_n_classes):
        x = tf.placeholder(tf.float32, [None, *input_shape])
        y = tf.placeholder(tf.float32, [None, out_n_classes])
        keep_prob = tf.placeholder(tf.float32)
        return x, y, keep_prob
 
    def create_weights_and_biases(self,
                                  conv_1_filter_shape,
                                  conv_1_n_outputs,
                                  conv_2_filter_shape,
                                  conv_2_n_outputs,
                                  fc_dim,
                                  fc_n_outputs,
                                  out_n_classes):
        
        weights = {
            'wc1': tf.Variable(tf.random_normal([*conv_1_filter_shape, conv_1_n_outputs])),
            'wc2': tf.Variable(tf.random_normal([*conv_2_filter_shape, conv_2_n_outputs])),
            'wfc': tf.Variable(tf.random_normal([fc_dim, fc_n_outputs])),
            'out': tf.Variable(tf.random_normal([fc_n_outputs, out_n_classes]))
        }
        
        biases = {
            'bc1': tf.Variable(tf.random_normal([conv_1_n_outputs])),
            'bc2': tf.Variable(tf.random_normal([conv_2_n_outputs])),
            'bfc': tf.Variable(tf.random_normal([fc_n_outputs])),
            'out': tf.Variable(tf.random_normal([out_n_classes]))
        }
        
        return weights, biases
 
    def build_layers(self, x, weights, biases, dropout):
        # Conv layer 1 - 28*28*1 to 14*14*32
        conv1 = self.get_conv2d_layer(x, weights['wc1'], biases['bc1'])
        conv1 = self.get_maxpool2d_layer(conv1, k=2)

        # Conv layer 2 - 14*14*32 to 7*7*64
        conv2 = self.get_conv2d_layer(conv1, weights['wc2'], biases['bc2'])
        conv2 = self.get_maxpool2d_layer(conv2, k=2)

        # Fully-connected layer - 7*7*64 to 1024
        fc = self.get_fc_layer(conv2, weights['wfc'], biases['bfc'], dropout)
        
        # Output Layer - class prediction - 1024 to 10
        out = self.get_output_layer(fc, weights['out'], biases['out'])
        
        return out
    
    def get_conv2d_layer(self, x, w, b, s=1):
        x = tf.nn.conv2d(x, w, strides=[1, s, s, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)
    
    def get_maxpool2d_layer(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    
    def get_fc_layer(self, x, w, b, dropout):
        x = tf.reshape(x, [-1, w.get_shape().as_list()[0]])
        x = tf.add(tf.matmul(x, w), b)
        x = tf.nn.relu(x)
        return tf.nn.dropout(x, dropout)
    
    def get_output_layer(self, x, w, b):
        return tf.add(tf.matmul(x, w), b)
    
    def define_cost_and_optimizer(self, logits, y, learning_rate):
        cost = tf.reduce_mean(                              tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)            .minimize(cost)
        return cost, optimizer

    def define_accuracy(self, logits, y):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# ## ConvNetworkTrainer

# In[18]:


class ConvNetworkTrainer:
    
    def train_network(self, 
                      network, 
                      mnist,
                      epochs=10,
                      batch_size=128,
                      test_valid_size=256,
                      dropout=0.75):
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
                
            for epoch in range(epochs):

                for batch_idx in range(mnist.train.num_examples//batch_size):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                    sess.run(network.optimizer,
                             feed_dict={network.x: batch_x,
                                        network.y: batch_y,
                                        network.keep_prob: dropout})

                    # Calculate batch loss and validation accuracy
                    loss = sess.run(network.cost,
                                    feed_dict={network.x: batch_x,
                                               network.y: batch_y,
                                               network.keep_prob: 1.})

                    valid_acc = sess.run(network.accuracy,
                                         feed_dict={
                                            network.x: mnist.validation.images[:test_valid_size],
                                            network.y: mnist.validation.labels[:test_valid_size],
                                            network.keep_prob: 1.})

                    logStr = 'Epoch {:>2}, Batch {:>3} - Loss: {:>10.4f} Validation Accuracy: {:.6f}'

                    print(logStr.format(epoch+1, batch_idx+1, loss, valid_acc))
                          
            # Calculate test accuracy
            test_acc = sess.run(network.accuracy,
                                feed_dict={
                                    network.x: mnist.test.images[:test_valid_size],
                                    network.y: mnist.test.labels[:test_valid_size],
                                    network.keep_prob: 1.})   
                          
            print('Testing Accuracy: {}'.format(test_acc))


# In[19]:


input_shape = [28, 28, 1]
conv_1_filter_shape = [5, 5, 1]
conv_1_n_outputs = 32
conv_2_filter_shape = [5, 5, 32]
conv_2_n_outputs = 64
fc_dim = 7*7*64
fc_n_outputs = 1024
out_n_classes = 10
dropout = 0.75
learning_rate = 0.00001
epochs = 10
batch_size = 128
test_valid_size = 256


# In[20]:


network = ConvNetwork()

network.build(input_shape,
              conv_1_filter_shape,
              conv_1_n_outputs,
              conv_2_filter_shape,
              conv_2_n_outputs,
              fc_dim,
              fc_n_outputs,
              out_n_classes,
              dropout,
              learning_rate)


# In[21]:


mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)


# In[22]:


networkTrainer = ConvNetworkTrainer()

networkTrainer.train_network(network,
                             mnist,
                             epochs,
                             batch_size,
                             test_valid_size,
                             dropout)


# ## FilterShapeCalculator

# In[23]:


class FilterShapeCalculator:
    
    def compute_filter_shape(self, input_shape, output_shape, strides):
        filter_height = None
        filter_width = None
        
        for f_h in range(input_shape[1], 1, -1):
            # out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
            out_h = math.ceil(float(input_shape[1] - f_h + 1) / float(strides[1]))

            if out_h == output_shape[1]:
                filter_height = f_h
                break
                
        for f_w in range(input_shape[2], 1, -1):
            # out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
            out_w = math.ceil(float(input_shape[2] - f_w + 1) / float(strides[2]))
            
            if out_w == output_shape[2]:
                filter_width = f_w
                break
                
        return (filter_height, filter_width)


# In[24]:


input_shape = (1, 4, 4, 1)
output_shape = (1, 2, 2, 3)
strides = (1, 2, 2, 1)

calculator = FilterShapeCalculator()
filter_shape = calculator.compute_filter_shape(input_shape, output_shape, strides)
print(filter_shape)


# ## TensorflowConvLayer

# In[25]:


class TensorflowConvLayer:
    
    def conv2d(self, input, input_shape, output_shape):
        """
        Setup the strides, padding and filter weight/bias such that
        the output shape is (1, 2, 2, 3).
        """
        
        # Set the stride for each dimension (batch_size, height, width, depth)
        strides = [1, 2, 2, 1]
        
        calculator = FilterShapeCalculator()
        filter_shape = calculator.compute_filter_shape(input_shape, output_shape, strides)
        
        # Filter (weights and bias)
        # The shape of the filter weight is (height, width, input_depth, output_depth)
        # The shape of the filter bias is (output_depth,)
        filter_weights = tf.Variable(tf.truncated_normal((filter_shape[0], 
                                                          filter_shape[1], 
                                                          input_shape[3], 
                                                          output_shape[3])))
        filter_biases = tf.Variable(tf.zeros(output_shape[3]))

        # Set the padding, either 'VALID' or 'SAME'
        padding = 'VALID'
        
        return tf.add(tf.nn.conv2d(input, filter_weights, strides, padding), filter_biases)


# In[26]:


# tf.nn.conv2d() requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape(input_shape)

X = tf.constant(x)

convLayer = TensorflowConvLayer()
conv_output = convLayer.conv2d(X, input_shape, output_shape)

print(conv_output.shape)


# ## TensorflowPoolingLayer

# In[27]:


class TensorflowPoolingLayer:
    
    def maxpool2d(self, input, input_shape, output_shape):
        """
        Set the values to `strides` and `ksize` such that
        the output shape after pooling is (1, 2, 2, 1).
        """
        
        # Set the stride for each dimension (batch_size, height, width, depth)
        strides = [1, 2, 2, 1]
        
        # Set the ksize (filter size) for each dimension (batch_size, height, width, depth)
        calculator = FilterShapeCalculator()
        filter_shape = calculator.compute_filter_shape(input_shape, output_shape, strides)
        ksize = [1, filter_shape[0], filter_shape[1], 1]
        
        # Set the padding, either 'VALID' or 'SAME'.
        padding = 'VALID'
        
        return tf.nn.max_pool(input, ksize, strides, padding)


# In[28]:


output_shape = (1, 2, 2, 1)

poolLayer = TensorflowPoolingLayer()
pool_output = poolLayer.maxpool2d(X, input_shape, output_shape)

print(pool_output.shape)

