import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers

# keep images in a folder named 'celeba' in the same directory as file

images = keras.preprocessing.image_dataset_from_directory(
    "celeba", label_mode=None, image_size=(64, 64), batch_size=64
)
images = images.map(lambda rgb: rgb / 255.0)

latent_dimensions = 128

"""
Returns a discriminator model
"""
def discriminator():
    model = keras.Sequential(name="discriminator")
    model.add(layers.Conv2D(128, kernel_size=5, strides=2, padding="same", input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(512, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1024, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(1024, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


"""
Returns a generator model
"""
def generator(latent_dimensions):
    model = keras.Sequential(name="generator")
    model.add(layers.Dense(4 * 4 * 1024, input_shape=(latent_dimensions,)))
    model.add(layers.Reshape((4, 4, 1024)))
    model.add(layers.Conv2DTranspose(512, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(256, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(64, kernel_size=5, strides=2, padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, kernel_size=5, padding="same", activation="tanh"))
    return model

class GAN(keras.Model):
    def __init__(self, discriminator_model, generator_model, latent_dimensions):
        super(GAN, self).__init__()
        self.discriminator_model = discriminator_model
        self.generator_model = generator_model
        self.latent_dimensions = latent_dimensions

    def compile(self, discriminator_optimizer, generator_optimizer, loss_function):
        super(GAN, self).compile()
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        self.loss_function = loss_function
        self.discriminator_loss_metric = keras.metrics.Mean(name="discriminator_loss")
        self.generator_loss_metric = keras.metrics.Mean(name="generator_loss")

    @property
    def metrics(self):
        return [self.discriminator_loss_metric, self.generator_loss_metric]

    def train_step(self, real_images):
        batch_size = batch_size = tf.shape(real_images)[0]

        d_loss = self.train_discriminator(real_images, batch_size)
        g_loss = self.train_generator(batch_size)

        self.discriminator_loss_metric.update_state(d_loss)
        self.generator_loss_metric.update_state(g_loss)

        return {
            "discriminator_loss": self.discriminator_loss_metric.result(),
            "generator_loss": self.generator_loss_metric.result(),
        }
    
    """
    Train the discriminator. The loss function is dependent on the difference between the true
    classifications of the inputs and what was predicted by the discriminator
    """
    def train_discriminator(self, real_images, batch_size):
        # Generate images from points in latent space
        generator_inputs = tf.random.normal(shape=(batch_size, self.latent_dimensions))
        generated_images = self.generator_model(generator_inputs)

        # Combine real and fake, add labels with random noise
        images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train discriminator model
        with tf.GradientTape() as tape:
            predictions = self.discriminator_model(images)
            d_loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(d_loss, self.discriminator_model.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(gradients, self.discriminator_model.trainable_weights))

        return d_loss

    """
    Train the generator. The loss function is dependent on the difference between all 'real' labels
    and the discriminator predictions. The generator is seeking to produce images that are deemed real
    by the discriminator
    """
    def train_generator(self, batch_size):
        # Sample latent space, create array of ideal outputs
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dimensions))
        goal_labels = tf.zeros((batch_size, 1))

        # Train Generator model
        with tf.GradientTape() as tape:
            predictions = self.discriminator_model(self.generator_model(random_latent_vectors))
            g_loss = self.loss_function(goal_labels, predictions)
        gradients = tape.gradient(g_loss, self.generator_model.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(gradients, self.generator_model.trainable_weights))

        return g_loss


class ImageSaver(keras.callbacks.Callback):
    def __init__(self, image_count=3, latent_dimensions=128):
        self.image_count= image_count
        self.latent_dimensions = latent_dimensions

    def on_epoch_end(self, epoch, logs):
        generator_inputs = tf.random.normal(shape=(self.image_count, self.latent_dimensions))
        output_images = self.model.generator_model(generator_inputs)
        output_images *= 255
        output_images.numpy()
        for i in range(self.image_count):
            img = keras.preprocessing.image.array_to_img(output_images[i])
            img.save("generated_img_%03d_%d.png" % (epoch, i))

epochs = 100

gan = GAN(discriminator_model=discriminator(), generator_model=generator(latent_dimensions), latent_dimensions=latent_dimensions)
gan.compile(
    discriminator_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    generator_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_function=keras.losses.BinaryCrossentropy(),
)

history = gan.fit(
    images, epochs=epochs, callbacks=[ImageSaver(image_count=10, latent_dimensions=latent_dimensions)]
)

discriminator_loss_values = history.history['discriminator_loss']
generator_loss_values = history.history['generator_loss']
min_axis_value = min(min(generator_loss_values, discriminator_loss_values)) - 0.1
max_axis_value = max(max(generator_loss_values, discriminator_loss_values)) + 0.1

plt.plot(discriminator_loss_values, label='discriminator loss')
plt.plot(generator_loss_values, label = 'generator loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([min_axis_value, max_axis_value])
plt.legend(loc='upper right')
plt.show()