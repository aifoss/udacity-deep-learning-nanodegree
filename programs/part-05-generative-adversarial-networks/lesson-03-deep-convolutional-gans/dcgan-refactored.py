
# coding: utf-8

# # Deep Convolutional GANs

# In[1]:


from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
from scipy.io import loadmat

import pickle as pkl
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")


# ## Getting Data

# In[2]:


class DataDownloader:

    def download_data(self, data_url, data_dir, train_datafile, test_datafile):
        if not isdir(data_dir):
            raise Exception("Data directory doesn't exist!")
        
        class DLProgress(tqdm):
            last_block = 0
            
            def hook(self, block_num=1, block_size=1, total_size=None):
                self.total = total_size
                self.update((block_num-self.last_block)*block_size)
                self.last_block = block_num
        
        if not isfile(train_datafile):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
                urlretrieve(data_url, train_datafile, pbar.hook)
        
        if not isfile(test_datafile):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Testing Set') as pbar:
                urlretrieve(data_url, test_datafile, pbar.hook)


# In[3]:


class DataLoader:
    
    def load_data(self, train_datafile, test_datafile):
        trainset = loadmat(train_datafile)
        testset = loadmat(test_datafile)
        return (trainset, testset)


# In[4]:


get_ipython().system('mkdir data')


# In[5]:


data_url = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
data_dir = 'data/'
train_datafile = data_dir + 'train_32x32.mat'
test_datafile = data_dir + 'test_32x32.mat'


# In[6]:


dataDownloader = DataDownloader()

dataDownloader.download_data(data_url, data_dir, train_datafile, test_datafile)


# In[7]:


dataLoader = DataLoader()

trainset, testset = dataLoader.load_data(train_datafile, test_datafile)


# ## Displaying Images

# In[8]:


class ImageDisplayer:
    
    def show_trainset_samples(self, trainset):
        idx = np.random.randint(0, trainset['X'].shape[3], size=36)
        fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, figsize=(5,5),)
        
        for ii, ax in zip(idx, axes.flatten()):
            ax.imshow(trainset['X'][:,:,:,ii], aspect='equal')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            
        plt.subplots_adjust(wspace=0, hspace=0)
        
        plt.show()
        
    
    def show_generated_samples(self, epoch, samples, nrows, ncols, figsize=(5,5)):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,
                                 figsize=figsize)
        
        for ax, img in zip(axes.flatten(), samples[epoch]):
            ax.axis('off')
            img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
            ax.set_adjustable('box-forced')
            im = ax.imshow(img, aspect='equal')

        plt.subplots_adjust(wspace=0, hspace=0)
        
        #return fig, axes
        
        plt.show()


# In[9]:


imageDisplayer = ImageDisplayer()

imageDisplayer.show_trainset_samples(trainset)


# ## Preparing Dataset

# In[10]:


class Dataset:
    
    def __init__(self, train, test, val_frac=0.5, shuffle=False, scale_func=None):
        split_idx = int(len(test['y'])*(1 - val_frac))
        self.test_x, self.valid_x = test['X'][:,:,:,:split_idx], test['X'][:,:,:,split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        
        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)
        
        if scale_func is None:
            self.scaler = self.scale
        else:
            self.scaler = scale_func
            
        self.shuffle = shuffle
        
        
    def batches(self, batch_size):
        if self.shuffle:
            idx = np.arange(len(dataset.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
        
        n_batches = len(self.train_y)//batch_size
        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii:ii+batch_size]
            y = self.train_y[ii:ii+batch_size]
            
            yield self.scaler(x), y
            
            
    def scale(self, x, feature_range=(-1, 1)):
        # scale to (0, 1)
        x = ((x - x.min())/(255 - x.min()))

        # scale to feature_range
        min, max = feature_range
        x = x * (max - min) + min
        return x


# In[11]:


dataset = Dataset(trainset, testset)


# ## Building DCGAN Model

# In[12]:


class DCGANBuilder:
    
    def model_inputs(self, real_dim, z_dim):
        """
        Build model input placeholders.
        """
        inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
        inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        return (inputs_real, inputs_z)
    
    
    def generator(self, z, output_dim, reuse=False, alpha=0.2, training=True):
        """
        Build generator network.
        """
        with tf.variable_scope('generator', reuse=reuse):
            
            # First, fully-connected layer
            fc1 = tf.layers.dense(z, 4*4*512)
            
            # Reshape it to start the convolutional stack
            rs2 = tf.reshape(fc1, (-1, 4, 4, 512))
            # Apply batch normalization
            bn2 = tf.layers.batch_normalization(rs2, training=training)
            # Apply Leaky ReLU
            relu2 = tf.maximum(alpha*bn2, bn2)
            # 4x4x512 now
            
            # Apply transposed convolution
            conv3 = tf.layers.conv2d_transpose(relu2, 256, 5, strides=2, padding='same')
            # Apply batch normalization
            bn3 = tf.layers.batch_normalization(conv3, training=training)
            # Apply Leaky ReLU
            relu3 = tf.maximum(alpha*bn3, bn3)
            # 8x8x256 now
            
            # Apply transposed convolution
            conv4 = tf.layers.conv2d_transpose(relu3, 128, 5, strides=2, padding='same')
            # Apply batch normalization
            bn4 = tf.layers.batch_normalization(conv4, training=training)
            # Apply Leaky ReLU
            relu4 = tf.maximum(alpha*bn4, bn4)
            # 16x16x128 now
            
            # Output layer
            logits = tf.layers.conv2d_transpose(relu4, output_dim, 5, strides=2, padding='same')
            # 32x32x3 now
            
            # Apply tanh
            out = tf.tanh(logits)
            
            return out
        
        
    def discriminator(self, x, reuse=False, alpha=0.2):
        """
        Build discriminator network.
        """
        with tf.variable_scope('discriminator', reuse=reuse):
            
            # Input is 32x32x3
            
            # Apply convolution
            conv1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
            # Apply Leaky ReLU
            relu1 = tf.maximum(alpha*conv1, conv1)
            # 16x16x64 now
            
            # Apply convolution
            conv2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same')
            # Apply batch normalization
            bn2 = tf.layers.batch_normalization(conv2, training=True)
            # Apply Leaky ReLU
            relu2 = tf.maximum(alpha*bn2, bn2)
            # 8x8x128 now
            
            # Apply convolution
            conv3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same')
            # Apply batch normalization
            bn3 = tf.layers.batch_normalization(conv3, training=True)
            # Apply Leaky ReLU
            relu3 = tf.maximum(alpha*bn3, bn3)
            # 4x4x256 now
            
            # Flatten
            flat = tf.reshape(relu3, (-1, 4*4*256))
            
            # Fully-connected output layer
            logits = tf.layers.dense(flat, 1)
            
            # Apply sigmoid
            out = tf.sigmoid(logits)
            
            return out, logits
    
    
    def model_loss(self, input_real, input_z, output_dim, alpha=0.2):
        """
        Get the loss for the discriminator and generator.
        
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """
        g_model = self.generator(input_z, output_dim, alpha=alpha)
        d_model_real, d_logits_real = self.discriminator(input_real, alpha=alpha)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True, alpha=alpha)
        
        def get_loss(logits, labels):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, 
                                                                          labels=labels))
        
        d_loss_real = get_loss(d_logits_real, tf.ones_like(d_model_real))
        d_loss_fake = get_loss(d_logits_fake, tf.zeros_like(d_model_fake))
        g_loss = get_loss(d_logits_fake, tf.ones_like(d_model_fake))
        
        d_loss = d_loss_real + d_loss_fake
        
        return (d_loss, g_loss)

    
    def model_opt(self, d_loss, g_loss, learning_rate, beta1):
        """
        Get optimization operations.
        
        :param d_loss: Discriminator loss Tensor
        :param g_loss: Generator loss Tensor
        :param learning_rate: Learning Rate Placeholder
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :return: A tuple of (discriminator training operation, generator training operation)
        """
        # Get weights and bias to update
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]
        
        # Optimize
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)                            .minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1)                            .minimize(g_loss, var_list=g_vars)    
        
        return (d_train_opt, g_train_opt)


# In[13]:


class DCGAN:
    
    def __init__(self, real_size, z_size, learning_rate, alpha=0.2, beta1=0.5):
        """
        Build GAN.
        """
        tf.reset_default_graph()
        
        dcganBuilder = DCGANBuilder()
        
        self.input_real, self.input_z = dcganBuilder.model_inputs(real_size, z_size)
        
        self.d_loss, self.g_loss = dcganBuilder.model_loss(self.input_real, 
                                                           self.input_z,
                                                           real_size[2], 
                                                           alpha=alpha)
        
        self.d_opt, self.g_opt = dcganBuilder.model_opt(self.d_loss, 
                                                        self.g_loss, 
                                                        learning_rate, 
                                                        beta1)


# In[14]:


real_size = (32,32,3)
z_size = 100
learning_rate = 0.0002
alpha = 0.2
beta1 = 0.5


# In[15]:


dcgan = DCGAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)


# ## Training DCGAN Model

# In[23]:


class DCGANTrainer:
    
    def train_model(self, dcgan, dataset):
        """
        Train DCGAN model.
        """
        
        sample_z = np.random.uniform(-1, 1, size=(72, z_size))
        
        losses, samples = [], []
        steps = 0
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            
            dcganBuilder = DCGANBuilder()
            imageDisplayer = ImageDisplayer()
            
            for e in range(epochs):
                for x, y in dataset.batches(batch_size):
                    steps += 1
                    
                    # Sample random noise for G
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                    
                    # Run optimizers
                    _ = sess.run(dcgan.d_opt, feed_dict={dcgan.input_real: x,
                                                         dcgan.input_z: batch_z})
                    _ = sess.run(dcgan.g_opt, feed_dict={dcgan.input_z: batch_z,
                                                         dcgan.input_real: x})
                    
                    if steps % print_every == 0:
                        # At the end of each epoch, get the losses and print them out
                        train_loss_d = dcgan.d_loss.eval({dcgan.input_z: batch_z,
                                                          dcgan.input_real: x})
                        train_loss_g = dcgan.g_loss.eval({dcgan.input_z: batch_z})
                        
                        print("Epoch {}/{}...".format(e+1, epochs),
                              "Step {}...".format(steps),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                        
                        # Save losses to view after training
                        losses.append((train_loss_d, train_loss_g))
                
                # Generate samples per each ecpoh
                gen_samples = sess.run(
                    dcganBuilder.generator(dcgan.input_z, 3, reuse=True, training=False),
                    feed_dict={dcgan.input_z: sample_z})
                samples.append(gen_samples)
                imageDisplayer.show_generated_samples(-1, samples, 6, 12, figsize=figsize)
        
            saver.save(sess, './checkpoints/generator.ckpt')
        
        with open('samples.pkl', 'wb') as f:
            pkl.dump(samples, f)

        return (losses, samples)


# In[17]:


class PlotDisplayer:
    
    def show_training_losses(self, losses):
        fig, ax = plt.subplots()
        losses = np.array(losses)
        plt.plot(losses.T[0], label='Discriminator', alpha=0.5)
        plt.plot(losses.T[1], label='Generator', alpha=0.5)
        plt.title("Training Losses")
        plt.legend()


# In[26]:


epochs = 10
batch_size = 128
print_every = 50 
show_every = 1000
figsize = (10,5)


# In[19]:


get_ipython().system('mkdir checkpoints')


# In[27]:


dcganTrainer = DCGANTrainer() 

losses, samples = dcganTrainer.train_model(dcgan, dataset)


# In[28]:


plotDisplayer = PlotDisplayer()

plotDisplayer.show_training_losses(losses)


# In[29]:


imageDisplayer.show_generated_samples(-1, samples, 6, 12, figsize=(10,5))
