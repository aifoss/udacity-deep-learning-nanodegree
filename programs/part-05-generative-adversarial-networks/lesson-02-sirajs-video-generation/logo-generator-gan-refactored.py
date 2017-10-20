
# coding: utf-8

# In[1]:


from __future__ import print_function

from keras import utils
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.optimizers import SGD, Adam

import numpy as np

from PIL import Image, ImageOps

import argparse
import math
import os
import os.path
import glob

K.set_image_dim_ordering('th') # ensure our dimension notation matches


# In[2]:


G_WGTS_FILE = 'goodgenerator.h5'
D_WGTS_FILE = 'gooddiscriminator.h5'

ENTROPY = 'binary_crossentropy'
SGD_OPT = 'SGD'

EPOCHS = 400
BATCH_SIZE = 10


# In[3]:


get_ipython().system('mkdir logo-generated-images')


# ## Loading Data

# In[4]:


class DataLoader:
    
    def get_train_image_set(self):
        '''
        Return formatted train image set from loaded dataset.
        '''
        X_train = self.load_train_image_data()
        X_train = (X_train.astype(np.float32)-127.5)/127.5
        X_train = X_train.reshape((X_train.shape[0], 1) + X_train.shape[1:])
        print('X_train.shape: {}'.format(X_train.shape))
        return X_train
    
    
    def load_train_image_data(self, pixels=128, verbose=False):
        '''
        Load image dataset from directory.
        '''
        print("Loading data...")
        
        X_train = []
        paths = glob.glob(os.path.normpath(os.getcwd()+'/logos/*.jpg'))
        
        for path in paths:
            if verbose: print(path)
            
            im = Image.open(path)
            im = ImageOps.fit(im, (pixels, pixels), Image.ANTIALIAS)
            im = ImageOps.grayscale(im)
            im = np.asarray(im)
            
            X_train.append(im)
            
        print("Finished loading data")
        
        return np.array(X_train)


# In[5]:


dataLoader = DataLoader()

X_train = dataLoader.get_train_image_set()


# ## Creating Models

# In[6]:


class ModelCreator:
    
    def generator_model(self):
        '''
        Build generator model.
        '''
        model = Sequential()
        model.add(Dense(input_dim=100, units=1024))
        model.add(Activation('tanh'))
        model.add(Dense(128*8*8))
        model.add(BatchNormalization())
        model.add(Activation('tanh'))
        model.add(Reshape((128, 8, 8), input_shape=(128*8*8,)))
        model.add(UpSampling2D(size=(4, 4)))
        model.add(Convolution2D(64, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        model.add(UpSampling2D(size=(4, 4)))
        model.add(Convolution2D(1, (5, 5), padding='same'))
        model.add(Activation('tanh'))
        return model
    
    
    def discriminator_model(self):
        '''
        Build discriminator model.
        '''
        model = Sequential()
        model.add(Convolution2D(64, (5, 5), padding='same', input_shape=(1, 128, 128)))
        model.add(Activation('tanh'))
        model.add(AveragePooling2D(pool_size=(4, 4)))
        model.add(Convolution2D(128, (5, 5)))
        model.add(Activation('tanh'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('tanh'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        return model
    
    
    def generator_containing_discriminator(self, generator, discriminator):
        '''
        Build generator model containing discriminator.
        '''
        model = Sequential()
        model.add(generator)
        discriminator.trainable = False
        model.add(discriminator)
        return model


# In[7]:


class ModelCompiler:
    
    def compile_models(self, load_weights=False):
        '''
        Build and compile models.
        
        :param load_weights: Indicates whether or not to load weights from files.
        '''
        
        modelCreator = ModelCreator()
        
        # Create models
        generator = modelCreator.generator_model()
        discriminator = modelCreator.discriminator_model()
        
        # Load weights
        if load_weights:
            self.load_weights(generator, discriminator)
  
        # Create generator containing discriminator
        discriminator_on_generator =             modelCreator.generator_containing_discriminator(generator, discriminator)

        # Add losses and optimizers
        g_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        d_opt = SGD(lr=0.0005, momentum=0.9, nesterov=True)
        
        generator.compile(loss=ENTROPY, optimizer=SGD_OPT)
        discriminator_on_generator.compile(loss=ENTROPY, optimizer=g_opt)
        discriminator.trainable = True
        discriminator.compile(loss=ENTROPY, optimizer=d_opt)
        
        return (generator, discriminator, discriminator_on_generator)
    
        
    def load_weights(self, generator, discriminator):
        '''
        Load weights from files.
        '''
        generator.load_weights(G_WGTS_FILE, True)
        discriminator.load_weights(D_WGTS_FILE, True)
        
        
    def print_models(self, generator, discriminator, discriminator_on_generator):
        print('\ngenerator:')
        generator.summary()
        print('\n\ndiscriminator:')
        discriminator.summary()
        print('\n\ndiscriminator_on_generator:')
        discriminator_on_generator.summary()
        print('')


# In[8]:


modelCompiler = ModelCompiler()

models = modelCompiler.compile_models()

modelCompiler.print_models(*models)


# In[19]:


class ModelTrainer:
    
    def train_models(self, models, X_train, load_weights=False):
        '''
        Train models.
        
        :param models: Models to train
        :param X_train: Training image dataset
        :param load_weights: Whether or not to create models with weights loaded from files
        '''
        
        # Load models with saved weights
        if load_weights:
            models = ModelCompiler().compile_models(True)
        
        generator = models[0]
        discriminator = models[1]
        discriminator_on_generator = models[2]
        
        imageProcessor = ImageProcessor()
        
        # Initialize noise
        noise = np.zeros((BATCH_SIZE, 100))
        
        for epoch in range(EPOCHS):
            print('\nEpoch: {}'.format(epoch+1))
            
            n_batches = int(X_train.shape[0]/BATCH_SIZE)
            
            for idx in range(n_batches):
                # Randomize noise
                Util().randomize_noise(noise, BATCH_SIZE)
                    
                image_batch = X_train[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE]
                generated_images = generator.predict(noise, verbose=0)
                
                # Save image
                if idx % 20 == 0 and epoch % 10 == 0:
                    imageProcessor.save_image(epoch, idx, generated_images)
                
                # Compute and log discriminator loss
                self.compute_d_loss(idx, discriminator, image_batch, generated_images)        
               
                # Randomize noise
                Util().randomize_noise(noise, BATCH_SIZE)
            
                # Compute and log generator loss
                self.compute_g_loss(idx, discriminator, discriminator_on_generator, noise)
            
                if epoch % 10 == 9:
                    self.save_weights(generator, discriminator)

        print('\nTraining complete\n')
        

    def compute_d_loss(self, idx, discriminator, image_batch, generated_images):
        '''
        Compute discriminator loss.
        '''
        X = np.concatenate((image_batch, generated_images))
        y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
        d_loss = discriminator.train_on_batch(X, y)
        print("batch %d d_loss : %f" % (idx+1, d_loss))

        
    def compute_g_loss(self, idx, discriminator, discriminator_on_generator, noise):
        '''
        Compute generator loss.
        '''
        discriminator.trainable = False
        g_loss = discriminator_on_generator.train_on_batch(noise, [1]*BATCH_SIZE)
        discriminator.trainable = True
        print("batch %d g_loss : %f" % (idx+1, g_loss))
    
    
    def save_weights(self, generator, discriminator):
        '''
        Save trained weights to files.
        '''
        generator.save_weights(G_WGTS_FILE)
        discriminator.save_weights(D_WGTS_FILE)


# In[20]:


class Util:
    
    def randomize_noise(self, noise, batch_size):
        '''
        Randomize noise matrix.
        '''
        for i in range(batch_size):
            noise[i, :] = np.random.uniform(-1, 1, 100)


# In[11]:


class ImageProcessor:
    
    def save_image(self, epoch, idx, images):
        '''
        Save generated images.
        '''
        image = self.combine_images(images)
        image = image * 127.5 + 127.5

        path_str = os.getcwd() + '/logo-generated-images/'                               + str(epoch+1)+'_'+str(idx)+'.png'
        dst_path = os.path.normpath(path_str)
        
        Image.fromarray(image.astype(np.uint8)).save(dst_path)

        
    def combine_images(self, images):
        '''
        Combine generated images.
        '''
        num = images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = images.shape[2:]
        
        image = np.zeros((height*shape[0], width*shape[1]),
                          dtype=images.dtype)
        
        for idx, img in enumerate(images):
            i = int(idx/width)
            j = idx % width
            
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = img[0, :, :]
        
        return image
    
    
    def clean(self, image):
        '''
        Clean image.
        '''
        for i in range(1, image.shape[0]-1):
            for j in range(1, image.shape[1]-1):
                if image[i][j]                         + image[i+1][j] + image[i][j+1]                         + image[i-1][j] + image[i][j-1] > 127 * 5:
                    image[i][j] = 255


# In[12]:


modelTrainer = ModelTrainer()

modelTrainer.train_models(models, X_train, False)


# ## Generating Image

# In[21]:


class ImageGenerator:
    
    def generate_image(self, batch_size):
        modelCreator = ModelCreator()
        
        generator = modelCreator.generator_model()
        generator.compile(loss=ENTROPY, optimizer=SGD_OPT)
        generator.load_weights(G_WGTS_FILE)
        
        imageProcessor = ImageProcessor()
        
        noise = np.zeros((batch_size, 100))
        Util().randomize_noise(noise, batch_size)
  
        generated_images = generator.predict(noise, verbose=1)
        
        print('generated_images.shape: {}'.format(generated_images.shape))
        
        for image in generated_images:
            image = image[0]
            image = image*127.5+127.5
            Image.fromarray(image.astype(np.uint8)).save("dirty.png")
            Image.fromarray(image.astype(np.uint8)).show()
            imageProcessor.clean(image)
            image = Image.fromarray(image.astype(np.uint8))
            image.show()        
            image.save("clean.png")


# In[22]:


imageGenerator = ImageGenerator()

imageGenerator.generate_image(1)

