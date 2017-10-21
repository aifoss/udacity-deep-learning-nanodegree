
# coding: utf-8

# # Face Generation GAN

# In[1]:


get_ipython().magic('matplotlib inline')

import os
import math
import hashlib
import zipfile
import gzip
import shutil
import warnings

from urllib.request import urlretrieve
from distutils.version import LooseVersion
from PIL import Image
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot

import numpy as np
import tensorflow as tf


# ## Downloading and Extracting Data

# In[6]:


class DataExtractor:
    
    def download_extract(self, database_name, data_path):
        """
        Download and extract database.
        
        :param database_name: Database name
        :param data_path: Data save directory
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'

        if database_name == DATASET_CELEBA_NAME:
            url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
            hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
            extract_path = os.path.join(data_path, 'img_align_celeba')
            save_path = os.path.join(data_path, 'celeba.zip')
            extract_fn = self.unzip
        elif database_name == DATASET_MNIST_NAME:
            url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
            hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
            extract_path = os.path.join(data_path, 'mnist')
            save_path = os.path.join(data_path, 'train-images-idx3-ubyte.gz')
            extract_fn = self.ungzip

        if os.path.exists(extract_path):
            print('Found {} Data'.format(database_name))
            return

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        if not os.path.exists(save_path):
            with DLProgress(unit='B', unit_scale=True, miniters=1, 
                            desc='Downloading {}'.format(database_name)) as pbar:
                urlretrieve(url,
                            save_path,
                            pbar.hook)

        assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code,             '{} file is corrupted.  Remove the file and try again.'.format(save_path)

        os.makedirs(extract_path)
        
        try:
            extract_fn(save_path, extract_path, database_name, data_path)
        except Exception as err:
            shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
            raise err

        # Remove compressed data
        os.remove(save_path)
    
    
    def unzip(self, save_path, _, database_name, data_path):
        """
        Unzip wrapper with the same interface as _ungzip.
        
        :param save_path: The path of the gzip files
        :param database_name: Name of database
        :param data_path: Path to extract to
        :param _: HACK - Used to have to same interface as _ungzip
        """
        print('Extracting {}...'.format(database_name))
        with zipfile.ZipFile(save_path) as zf:
            zf.extractall(data_path)


    def ungzip(self, save_path, extract_path, database_name, _):
        """
        Unzip a gzip file and extract it to extract_path.
        
        :param save_path: The path of the gzip files
        :param extract_path: The location to extract the data to
        :param database_name: Name of database
        :param _: HACK - Used to have to same interface as _unzip
        """
        # Get data from save_path
        with open(save_path, 'rb') as f:
            with gzip.GzipFile(fileobj=f) as bytestream:
                magic = self.read32(bytestream)
                if magic != 2051:
                    raise ValueError('Invalid magic number {} in file: {}'                                     .format(magic, f.name))
                num_images = self.read32(bytestream)
                rows = self.read32(bytestream)
                cols = self.read32(bytestream)
                buf = bytestream.read(rows * cols * num_images)
                data = np.frombuffer(buf, dtype=np.uint8)
                data = data.reshape(num_images, rows, cols)

        # Save data to extract_path
        for image_i, image in enumerate(
                tqdm(data, unit='File', unit_scale=True, miniters=1, 
                     desc='Extracting {}'.format(database_name))):
            Image.fromarray(image, 'L').save(
                os.path.join(extract_path, 'image_{}.jpg'.format(image_i)))
            
            
    def read32(self, bytestream):
        """
        Read 32-bit integer from bytesteam.
        
        :param bytestream: A bytestream
        :return: 32-bit integer
        """
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]    


# In[3]:


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


# In[4]:


# data_dir = './data'
data_dir = '/input/R5KrjnANiKVhLWAkpXhNBe' # FloydHub


# In[7]:


dataExtractor = DataExtractor()

dataExtractor.download_extract('mnist', data_dir)
dataExtractor.download_extract('celeba', data_dir)


# ## Exploring Image Data

# In[8]:


class Dataset(object):
    """
    Dataset
    """
    def __init__(self, dataset_name, data_files):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        DATASET_CELEBA_NAME = 'celeba'
        DATASET_MNIST_NAME = 'mnist'
        IMAGE_WIDTH = 28
        IMAGE_HEIGHT = 28

        if dataset_name == DATASET_CELEBA_NAME:
            self.image_mode = 'RGB'
            image_channels = 3

        elif dataset_name == DATASET_MNIST_NAME:
            self.image_mode = 'L'
            image_channels = 1

        self.data_files = data_files
        self.shape = len(data_files), IMAGE_WIDTH, IMAGE_HEIGHT, image_channels
        
        
    def get_batches(self, batch_size):
        """
        Generate batches
        :param batch_size: Batch Size
        :return: Batches of data
        """
        IMAGE_MAX_VALUE = 255

        current_index = 0
        while current_index + batch_size <= self.shape[0]:
            data_batch = DataBatchGenerator().get_batch(
                self.data_files[current_index:current_index + batch_size],
                *self.shape[1:3],
                self.image_mode)

            current_index += batch_size

            yield data_batch / IMAGE_MAX_VALUE - 0.5


# In[9]:


class DataBatchGenerator:
    
    def get_batch(self, image_files, width, height, mode):
        """
        Generate image data batch.
        """
        data_batch = np.array(
            [self.get_image(sample_file, width, height, mode) \
                 for sample_file in image_files]).astype(np.float32)

        # Make sure the images are in 4 dimensions
        if len(data_batch.shape) < 4:
            data_batch = data_batch.reshape(data_batch.shape + (1,))

        return data_batch
    
    
    def get_image(self, image_path, width, height, mode):
        """
        Read image from image_path.
        
        :param image_path: Path of image
        :param width: Width of image
        :param height: Height of image
        :param mode: Mode of image
        :return: Image data
        """
        image = Image.open(image_path)

        # HACK - Check if image is from the CELEBA dataset
        if image.size != (width, height):  
            # Remove most pixels that aren't part of a face
            face_width = face_height = 108
            j = (image.size[0] - face_width) // 2
            i = (image.size[1] - face_height) // 2
            image = image.crop([j, i, j + face_width, i + face_height])
            image = image.resize([width, height], Image.BILINEAR)

        return np.array(image.convert(mode))


# In[10]:


class ImageDisplayer:
    
    def images_square_grid(self, images, mode):
        """
        Save images as a square grid.
        
        :param images: Images to be used for the grid
        :param mode: The mode to use for images
        :return: Image of images in a square grid
        """
        # Get maximum size for square grid of images
        save_size = math.floor(np.sqrt(images.shape[0]))

        # Scale to 0-255
        images = (((images-images.min())*255)/(images.max()-images.min())).astype(np.uint8)

        # Put images in a square arrangement
        images_in_square = np.reshape(
                images[:save_size*save_size],
                (save_size, save_size, images.shape[1], images.shape[2], images.shape[3]))
        if mode == 'L':
            images_in_square = np.squeeze(images_in_square, 4)

        # Combine images to grid image
        new_im = Image.new(mode, (images.shape[1] * save_size, images.shape[2] * save_size))
        for col_i, col_images in enumerate(images_in_square):
            for image_i, image in enumerate(col_images):
                im = Image.fromarray(image, mode)
                new_im.paste(im, (col_i * images.shape[1], image_i * images.shape[2]))

        return new_im


# In[11]:


show_n_images = 25


# In[12]:


dataBatchGenerator = DataBatchGenerator()
imageDisplayer = ImageDisplayer()


# In[13]:


mnist_images = dataBatchGenerator.get_batch(
    glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')

pyplot.imshow(imageDisplayer.images_square_grid(mnist_images, 'L'), cmap='gray')


# In[14]:


celeba_images = dataBatchGenerator.get_batch(
    glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')

pyplot.imshow(imageDisplayer.images_square_grid(celeba_images, 'RGB'))


# ## Checking TensorFlow/GPU

# In[15]:


class TensorFlowGPUChecker:
    
    def check(self):
        # Check TensorFlow Version
        assert LooseVersion(tf.__version__) >= LooseVersion('1.0'),            'Please use TensorFlow version 1.0 or newer.  You are using {}'.            format(tf.__version__)
        print('TensorFlow Version: {}'.format(tf.__version__))

        # Check for a GPU
        if not tf.test.gpu_device_name():
            warnings.warn('No GPU found. Please use a GPU to train your neural network.')
        else:
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# In[16]:


tfgChecker = TensorFlowGPUChecker()

tfgChecker.check()


# ## Building and Training GAN

# In[17]:


class GAN:
    
    def __init__(self, input_real, input_z, lr, d_loss, g_loss, d_opt, g_opt):
        self.input_real = input_real
        self.input_z = input_z
        self.lr = lr
        self.d_loss = d_loss
        self.g_loss = g_loss
        self.d_opt = d_opt
        self.g_opt = g_opt


# In[18]:


class GANBuilder:
    
    def build_gan(self, data_shape, beta1):
        """
        Build GAN.
        """        
        input_real, input_z, lr = self.model_inputs(data_shape[1], 
                                                    data_shape[2], 
                                                    data_shape[3], 
                                                    z_dim)
        d_loss, g_loss = self.model_loss(input_real, input_z, data_shape[3])
        d_opt, g_opt = self.model_opt(d_loss, g_loss, lr, beta1)

        return GAN(input_real, input_z, lr, d_loss, g_loss, d_opt, g_opt)
    
    
    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        """
        Create the model inputs.
        
        :param image_width: The input image width
        :param image_height: The input image height
        :param image_channels: The number of image channels
        :param z_dim: The dimension of Z
        :return: Tuple of (tensor of real input images, tensor of z data, learning rate)
        """

        image_shape = (image_width, image_height, image_channels)
        input_real = tf.placeholder(tf.float32, (None, *image_shape), name='input_real')
        input_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        lr = tf.placeholder(tf.float32, name='learning_rate')

        return (input_real, input_z, lr)

    
    def discriminator(self, images, reuse=False):
        """
        Create the discriminator network.
        
        :param images: Tensor of input image(s)
        :param reuse: Boolean if the weights should be reused
        :return: Tuple of (tensor output of the discriminator, 
                           tensor logits of the discriminator)
        """
        
        def conv2d(x, out_dim, k=5, s=2, padding='same', use_batch_norm=True, training=True):
            layer = tf.layers.conv2d(x, out_dim, k, s, padding)
            if use_batch_norm == True:
                layer = tf.layers.batch_normalization(layer, training=training)
            layer = tf.maximum(alpha*layer, layer)
            return layer

        with tf.variable_scope('discriminator', reuse=reuse):
            layer_1 = conv2d(images, 32, use_batch_norm=False, training=False) # => 14x14x32                       
            layer_2 = conv2d(layer_1,  64)  # => 7 x 7x 64
            layer_3 = conv2d(layer_2, 128)  # => 4 x 4x128 
            layer_4 = conv2d(layer_3, 256)  # => 2 x 2x256

            flat = tf.reshape(layer_4, (-1, 2*2*256))

            logits = tf.layers.dense(flat, 1)
            output = tf.sigmoid(logits)

            return output, logits
    
    
    def generator(self, z, out_channel_dim, is_train=True):
        """
        Create the generator network.
        
        :param z: Input z
        :param out_channel_dim: The number of channels in the output image
        :param is_train: Boolean if generator is being used for training
        :return: The tensor output of the generator
        """

        def fully_connect(x, dims, training):
            layer = tf.layers.dense(x, dims[0]*dims[1]*dims[2])
            layer = tf.reshape(layer, (-1, dims[0], dims[1], dims[2]))
            layer = tf.layers.batch_normalization(layer, training=training)
            layer = tf.maximum(alpha*layer, layer)
            return layer

        def conv2d_transpose(x, out_dim, k=5, s=2, padding='same', 
                             use_batch_norm=True, use_leaky_relu=True, training=True):
            layer = tf.layers.conv2d_transpose(x, out_dim, k, s, padding)
            if use_batch_norm == True:
                layer = tf.layers.batch_normalization(layer, training=training)
            if use_leaky_relu == True:
                layer = tf.maximum(alpha*layer, layer)
            return layer
        
        with tf.variable_scope('generator', reuse=not is_train):
            layer_1 = fully_connect(z, (4, 4, 512), is_train)           # => 4x4x512
            layer_2 = conv2d_transpose(layer_1, 128, k=4, s=1, 
                                       padding='valid',
                                       training=is_train)               # => 7x7x128  
            layer_3 = conv2d_transpose(layer_2, 64, training=is_train)  # => 14x14x64
            layer_4 = conv2d_transpose(layer_3, 32, training=is_train)  # => 28x28x32 

            logits = conv2d_transpose(layer_4, out_channel_dim, 3, 1, 
                                      'same', False, False, True)       # => 28x28x3(1)
            
            output = tf.tanh(logits)

            return output
    
    
    def model_loss(self, input_real, input_z, out_channel_dim):
        """
        Get the loss for the discriminator and generator.
        
        :param input_real: Images from the real dataset
        :param input_z: Z input
        :param out_channel_dim: The number of channels in the output image
        :return: A tuple of (discriminator loss, generator loss)
        """

        g_model = self.generator(input_z, out_channel_dim, is_train=True)
        d_model_real, d_logits_real = self.discriminator(input_real, reuse=False)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

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
        :return: A tuple of (discriminator training operation, 
                             generator training operation)
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


# In[19]:


class GANTrainer:
    
    def train_gan(self, epoch_count, batch_size, z_dim, learning_rate, beta1, 
                  get_batches, data_shape, data_image_mode):
        """
        Train the GAN.
        
        :param epoch_count: Number of epochs
        :param batch_size: Batch Size
        :param z_dim: Z dimension
        :param learning_rate: Learning Rate
        :param beta1: The exponential decay rate for the 1st moment in the optimizer
        :param get_batches: Function to get batches
        :param data_shape: Shape of the data
        :param data_image_mode: The image mode to use for images ("RGB" or "L")
        """

        gan = GANBuilder().build_gan(data_shape, beta1)
        
        step = 0
        print_every = 10
        show_every = 100
        n_images_to_show = 10

        d_losses, g_losses = [], []

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(epoch_count):          
                for batch_images in get_batches(batch_size):
                    batch_images *= 2
                    step += 1

                    # Sample random noise for G
                    batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                    # Run optimizers
                    _ = sess.run(gan.d_opt, feed_dict={gan.input_real: batch_images,
                                                       gan.input_z: batch_z,
                                                       gan.lr: learning_rate})
                    _ = sess.run(gan.g_opt, feed_dict={gan.input_real: batch_images,
                                                       gan.input_z: batch_z,
                                                       gan.lr: learning_rate})

                    if step % print_every == 0:
                        train_loss_d = gan.d_loss.eval({gan.input_real: batch_images,
                                                        gan.input_z: batch_z})
                        train_loss_g = gan.g_loss.eval({gan.input_z: batch_z})
                        
                        self.print_losses(epoch_count, epoch_i, step,
                                          train_loss_d, train_loss_g, 
                                          d_losses, g_losses)

                    if step % show_every == 0:
                        self.print_cumulative_average_losses(train_loss_d, train_loss_g, 
                                                             d_losses, g_losses)

                        self.show_generator_output(sess, n_images_to_show, gan.input_z, 
                                                   data_shape[3], data_image_mode)
                        
    
    def print_losses(self, epochs, epoch_i, step, train_loss_d, train_loss_g, d_losses, g_losses):
        d_losses.append(train_loss_d)
        g_losses.append(train_loss_g)

        print("Epoch {}/{}...".format(epoch_i+1, epochs),
              "Step: {}...".format(step),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
        
    
    def print_cumulative_average_losses(self, train_loss_d, train_loss_g, d_losses, g_losses):
        cum_avg_d_loss = np.mean(d_losses)
        cum_avg_g_loss = np.mean(g_losses)

        print("Cumulative Average Generator Loss / Discriminator Loss: {}".format(
            cum_avg_g_loss / cum_avg_d_loss))
        print("Current Generator Loss / Discriminator Loss: {}".format(
            train_loss_g / train_loss_d))
        
    def show_generator_output(self, sess, n_images, input_z, out_channel_dim, image_mode):
        """
        Show example output for the generator.
        
        :param sess: TensorFlow session
        :param n_images: Number of Images to display
        :param input_z: Input Z Tensor
        :param out_channel_dim: The number of channels in the output image
        :param image_mode: The mode to use for images ("RGB" or "L")
        """
        cmap = None if image_mode == 'RGB' else 'gray'
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = sess.run(
            GANBuilder().generator(input_z, out_channel_dim, False),
                                   feed_dict={input_z: example_z})

        images_grid = ImageDisplayer().images_square_grid(samples, image_mode)
        
        pyplot.imshow(images_grid, cmap=cmap)
        pyplot.show()


# ## Training Model on MNIST Dataset

# In[20]:


batch_size = 64
z_dim = 100
learning_rate = 0.0002
alpha = 0.1
beta1 = 0.5

epochs = 2

mnist_dataset = Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))

ganTrainer = GANTrainer()

with tf.Graph().as_default():
    ganTrainer.train_gan(epochs, batch_size, 
                         z_dim, learning_rate, beta1, 
                         mnist_dataset.get_batches,
                         mnist_dataset.shape, 
                         mnist_dataset.image_mode)


# ## Training Model on CELEBA Dataset

# In[21]:


batch_size = 64
z_dim = 100
learning_rate = 0.0002
alpha = 0.1
beta1 = 0.5
epochs = 1

celeba_dataset = Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))

ganTrainer = GANTrainer()

with tf.Graph().as_default():
    ganTrainer.train_gan(epochs, batch_size, 
                         z_dim, learning_rate, beta1, 
                         celeba_dataset.get_batches,
                         celeba_dataset.shape, 
                         celeba_dataset.image_mode)


# ### Submitting This Project
# When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_face_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
