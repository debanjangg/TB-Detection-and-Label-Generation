import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from PIL import Image


N = 10

epochs = 30
latentdim = 128

adamrate = 0.0002
adambeta = 0.5

batchsize = 64


def Generator(latentsize):
    model = tf.keras.Sequential()

    model.add(layers.Dense(7 * 7 * 128, input_dim=latentsize))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((7, 7, 128)))

    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='glorot_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2DTranspose(128, 4, strides=2, padding='same', kernel_initializer='glorot_normal'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(1, 7, padding='same', activation='tanh', kernel_initializer='glorot_normal'))
    
    return model

def Discriminator():

    model = tf.keras.Sequential()

    model.add(layers.Conv2D(32, 3, padding='same', strides=2, input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, 3, padding='same', strides=1))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, 3, padding='same', strides=2))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, 3, padding='same', strides=1))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.GlobalMaxPooling2D()),
    model.add(layers.Dense(1))
    
    return model


discriminator = Discriminator()
generator = Generator(latentdim)


class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latentdim):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latentdim = latentdim

    def compile(self, Doptimizer, Goptimizer, loss):
        super(GAN, self).compile()
        self.Doptimizer = Doptimizer
        self.Goptimizer = Goptimizer
        self.loss = loss

    def train_step(self, real):
        if isinstance(real, tuple):
            real = real[0]
        
        batchsize = tf.shape(real)[0]
        randomlatent = tf.random.normal(shape=(batchsize, self.latentdim))


        generatedimages = self.generator(randomlatent)

        combinedimages = tf.concat([generatedimages, real], axis=0)


        labels = tf.concat([tf.ones((batchsize, 1)), tf.zeros((batchsize, 1))], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(combinedimages)
            Dloss = self.loss(labels, predictions)
        grads = tape.gradient(Dloss, self.discriminator.trainable_weights)
        self.Doptimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        randomlatent = tf.random.normal(shape=(batchsize, self.latentdim))

        misleadinglabels = tf.zeros((batchsize, 1))

        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(randomlatent))
            Gloss = self.loss(misleadinglabels, predictions)
        grads = tape.gradient(Gloss, self.generator.trainable_weights)
        self.Goptimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {'Discriminator Loss': Dloss, 'Generator Loss': Gloss}


gan = GAN(discriminator=discriminator, generator=generator, latentdim=latentdim)
gan.compile(Doptimizer=tf.keras.optimizers.Adam(learning_rate=adamrate, beta_1=adambeta), Goptimizer=tf.keras.optimizers.Adam(learning_rate=adamrate, beta_1=adambeta),loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))


(Xtrain, _), (Xtest, _) = tf.keras.datasets.mnist.load_data()
digits = np.concatenate([Xtrain, Xtest])
digits = digits.astype('float32') / 255
digits = np.reshape(digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batchsize).prefetch(32)


gan.fit(dataset, epochs=epochs, callbacks=[Logger(numimg=3, latentdim=latentdim)])