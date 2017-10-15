
# coding: utf-8

# # Autoencoders

# In[1]:


get_ipython().magic('matplotlib inline')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


# ## DataLoader

# In[2]:


class DataLoader:
    
    def load_mnist_data(self):
        """
        Load MNIST dataset.
        """
        print('\nLoading MNIST dataset ...\n')
        mnist = input_data.read_data_sets('MNIST_data', validation_size=0)
        print('\nData loading complete\n')
        return mnist


# ## ImageDisplayer

# In[3]:


class ImageDisplayer:
    
    def show_train_image(self, mnist, img_idx, img_shape, cmap):
        """
        Display a single image in the MNIST training dataset.
        """
        img = mnist.train.images[img_idx]
        plt.imshow(img.reshape(img_shape), cmap=cmap)
        
        
    def show_test_images(self, input_imgs, output_imgs, img_shape, cmap):
        """
        Display first 10 images in the MNIST testing set in original and reconstructed forms. 
        """
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20,4))
            
        for images, row in zip([input_imgs, output_imgs], axes):
            for img, ax in zip(images, row):
                ax.imshow(img.reshape(image_shape), cmap=cmap)
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        fig.tight_layout(pad=0.1)


# # Loading Data

# In[4]:


dataLoader = DataLoader()

mnist = dataLoader.load_mnist_data()


# In[5]:


image_shape = (28, 28)
image_size = mnist.train.images.shape[1]
cmap = 'Greys_r'


# In[6]:


imageDisplayer = ImageDisplayer()

imageDisplayer.show_train_image(mnist, 2, image_shape, cmap)


# # Running Simple Autoencoder

# ## ImageFormatter

# In[7]:


class ImageFormatter:
    
    def format_in_imgs(self, imgs, img_shape, reshape, denoising, noise_factor):
        """
        Format input images to feed to network. 
        """
        in_imgs = imgs
        if reshape == True:
            in_imgs = self.reshape_images(in_imgs, img_shape)
        if denoising == True:
            in_imgs = self.add_noise_and_clip_images(in_imgs, noise_factor)
        return in_imgs
    
    
    def format_out_imgs(self, imgs, img_shape, reshape):
        """
        Format output images to feed to network.
        """
        out_imgs = imgs
        if reshape == True:
            out_imgs = self.reshape_images(out_imgs, img_shape)
        return out_imgs
    
        
    def reshape_images(self, imgs, img_shape):
        """
        Reshape input images.
        """
        return imgs.reshape((-1, *img_shape, 1))
    
    
    def add_noise_and_clip_images(self, imgs, noise_factor):
        """
        Add noise to input images and clip them.
        """
        # Add random noise to the input images
        noisy_imgs = imgs + noise_factor * np.random.randn(*imgs.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)
        return noisy_imgs


# ## ModelRunner

# In[8]:


class ModelRunner:
        
    def run_autoencoder(self, 
                        model, 
                        mnist, 
                        img_shape, 
                        epochs, 
                        batch_size, 
                        reshape=False,
                        denoising=False,
                        noise_factor=None):
        """
        Train an autoencoder network model.
        : model: Autoencoder model to train
        : mnist: MNIST dataset
        : img_shape: Image shape
        : epochs: Number of epochs 
        : batch_size: Size of train batch
        : reshape: Whether or not to reshape images
        : denoising: Whether or not the model is for denoising
        : noise_factor: Noise factor to use in case of denoising
        """
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            imageFormatter = ImageFormatter()
            
            print('\nTraining autoencoder ...\n')
            
            # Train the autoencoder
            for e in range(epochs):
                
                for ii in range(mnist.train.num_examples//batch_size):
                    batch = mnist.train.next_batch(batch_size)
                    imgs = batch[0]
                    
                    in_imgs = imageFormatter.format_in_imgs(
                                imgs, img_shape, reshape, 
                                denoising, noise_factor)
                    out_imgs = imageFormatter.format_out_imgs(
                                imgs, img_shape, reshape)
                    
                    feed_dict = {model.inputs: in_imgs, 
                                 model.targets: out_imgs}
                    
                    batch_cost, _ = sess.run([model.cost, model.optimizer], 
                                             feed_dict=feed_dict)
                    
                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Training loss: {:.4f}".format(batch_cost))
                    
            print('\nTraining complete\n')
            
            print('\nTesting autoencoder with sample images ...\n')
            
            # Test the autoencoder with sample images from the test set
            imgs = mnist.test.images[:10]
            
            in_imgs = imageFormatter.format_in_imgs(
                        imgs, img_shape, reshape, denoising, noise_factor)
            
            feed_dict={model.inputs: in_imgs}
            
            out_imgs = sess.run(model.decoded, 
                                feed_dict=feed_dict)
            
            imageDisplayer.show_test_images(in_imgs, out_imgs, img_shape, cmap)
            
            print('\nTraining and testing complete\n')


# ## SimpleAutoencoder

# In[9]:


class SimpleAutoencoder:

    def build(self, image_size, encoding_dim, learning_rate):
        """
        Build a simple autoencoder with a single RELU hidden layer.
        : image_size: Size of input image
        : encoding_dim: Size of the (hidden) encoding layer
        : learning_rate: Learning rate
        """
        
        print('\nBuilding a simple image autoencoder ...\n')
        
        # Create input and target placeholders
        self.inputs = tf.placeholder(tf.float32, (None, image_size), name='inputs')
        self.targets = tf.placeholder(tf.float32, (None, image_size), name='targets')

        # Output of hidden layer, single fully connected layer here with ReLU activation
        self.encoded = tf.layers.dense(self.inputs, encoding_dim, activation=tf.nn.relu)

        # Output layer logits, fully connected layer with no activation
        logits = tf.layers.dense(self.encoded, image_size, activation=None)
        
        # Sigmoid output from logits
        self.decoded = tf.nn.sigmoid(logits, name='output')
        
        # Sigmoid cross-entropy loss
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=logits)
        # Cost
        self.cost = tf.reduce_mean(loss)
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        
        print('Simple image autoencoder built\n')


# In[10]:


encoding_dim = 32
learning_rate = 0.001


# In[11]:


simpleAutoencoder = SimpleAutoencoder()

simpleAutoencoder.build(image_size, encoding_dim, learning_rate)


# In[12]:


epochs = 20
batch_size = 200


# In[13]:


modelRunner = ModelRunner()

modelRunner.run_autoencoder(simpleAutoencoder, mnist, image_shape, epochs, batch_size)


# ## ConvolutionalAutoencoder

# In[14]:


class ConvolutionalAutoencoder:
    
    def build(self, image_shape, learning_rate, conv_outputs=(None,16,8,8,8,8,16)):
        """
        Build an image autoencoder using convolution.
        : image_shape: Image shape
        : learning_rate: Learning rate
        """
        
        print('\nBuilding a convolutional autoencoder ...\n')
        
        ##### Placeholders
        # 28x28x1 input
        self.inputs = tf.placeholder(tf.float32, (None, *image_shape, 1), name='inputs')
        self.targets = tf.placeholder(tf.float32, (None, *image_shape, 1), name='targets')
        
        ##### Encoder
        # 4x4x8 encoder output
        encoder_logits = self.build_encoder(self.inputs, conv_outputs)
        self.encoded = tf.identity(encoder_logits, name='encoded')
        
        ##### Decoder
        # 28x28x1 decoder output
        decoder_logits = self.build_decoder(encoder_logits, conv_outputs)
        self.decoded = tf.nn.sigmoid(decoder_logits, name='decoded')

        ##### Cost and Optimizer
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=decoder_logits)
        self.cost = tf.reduce_mean(loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
        
        print('Convolutional autoencoder built\n')
    
    def build_encoder(self, _input, conv_outputs):
        # 28x28x16 (or 28x28x32) convolution
        conv1 = self.conv2d(_input, conv_outputs[1])
        # 14x14x16 (or 14x14x32) maxpool
        maxpool1 = self.maxpool2d(conv1)
        
        # 14x14x8 (or 14x14x32) convolution
        conv2 = self.conv2d(maxpool1, conv_outputs[2])
        # 7x7x8 (or 7x7x32) maxpool
        maxpool2 = self.maxpool2d(conv2)
        
        # 7x7x8 (or 7x7x16) convolution
        conv3 = self.conv2d(maxpool2, conv_outputs[3])
        # 4x4x8 (or 4x4x16) maxpool
        maxpool3 = self.maxpool2d(conv3)
        
        return maxpool3
    
    def build_decoder(self, _input, conv_outputs):
        upsample_sizes = (None, (7,7), (14,14), (28,28))
        
        # 7x7x8 (or 7x7x16) upsample
        upsample1 = self.upsample(_input, (7,7))
        # 7x7x8 (or 7x7x16) convolution
        conv4 = self.conv2d(upsample1, conv_outputs[4])
        
        # 14x14x8 (or 14x14x32) upsample
        upsample2 = self.upsample(conv4, (14,14))
        # 14x14x8 (or 14x14x32) convolution
        conv5 = self.conv2d(upsample2, conv_outputs[5])
        
        # 28x28x8 (or 28x28x32) upsample
        upsample3 = self.upsample(conv5, (28,28))
        # 28x28x16 (or 28x28x32) convolution
        conv6 = self.conv2d(upsample3, conv_outputs[6])
        
        # 28x28x1 output
        logits = self.conv2d(conv6, 1, None)
        
        return logits
    
    
    def conv2d(self, x, num_outputs, activation=tf.nn.relu):
        return tf.layers.conv2d(x, 
                                num_outputs, 
                                (3,3), # kernels 
                                padding='same', 
                                activation=activation)
    
    def maxpool2d(self, x):
        return tf.layers.max_pooling2d(x, 
                                       (2,2), # kernels
                                       (2,2), # strides
                                       padding='same')

    def upsample(self, x, size):
        return tf.image.resize_nearest_neighbor(x, size)


# # Running Convolutional Autoencoder

# In[15]:


conv_outputs = (None,16,8,8,8,8,16)


# In[16]:


convAutoencoder = ConvolutionalAutoencoder()

convAutoencoder.build(image_shape, learning_rate, conv_outputs)


# In[17]:


modelRunner.run_autoencoder(
    convAutoencoder, mnist, image_shape, epochs, batch_size, True)


# # Running Denoising Autoencoder

# In[18]:


conv_outputs = (None,32,32,16,16,32,32)


# In[19]:


denoisingAutoencoder = ConvolutionalAutoencoder()

denoisingAutoencoder.build(image_shape, learning_rate, conv_outputs)


# In[21]:


noise_factor = 0.5


# In[20]:


modelRunner.run_autoencoder(denoisingAutoencoder,
                            mnist,
                            image_shape,
                            epochs,
                            batch_size,
                            True,
                            True,
                            noise_factor)

