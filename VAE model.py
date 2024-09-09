import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# 1. Data Loading and Preprocessing with Progress Bar
def load_data(base_path, img_size=(128, 128)):
    images = []
    labels = []

    # Using tqdm for progress bar
    for img_file in tqdm(sorted(os.listdir(base_path)), desc=f"Loading Images from {base_path}"):
        if img_file.startswith('case'):
            img = load_img(os.path.join(base_path, img_file), target_size=img_size, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0  # Normalize image
            images.append(img_array)

            case_id = int(img_file.split('_')[1])  # Extract the case ID
            labels.append(case_id)

    return np.array(images), np.array(labels)

# 2. VAE Model Definition
def build_vae(input_shape=(128, 128, 1), latent_dim=2):
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    z = Sampling()([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(32 * 32 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((32, 32, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    decoder = models.Model(latent_inputs, outputs, name='decoder')

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=vae.custom_loss)
    return vae, encoder, decoder

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))  # Random normal tensor
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return reconstruction

    def custom_loss(self, inputs, reconstruction):
        z_mean, z_log_var, _ = self.encoder(inputs)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(tf.keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2))
        )
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        return reconstruction_loss + kl_loss

# 3. Progress Tracking with Callbacks
def create_callbacks(model_name):
    log_dir = f"logs/{model_name}"
    checkpoint_path = f"checkpoints/{model_name}_epoch{{epoch:02d}}.keras"
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )
    
    return [tensorboard_callback, checkpoint_callback]

# 4. Train the VAE with Callbacks
def train_vae(vae, X_train, X_val, epochs=20, batch_size=32, model_name="VAE"):
    callbacks = create_callbacks(model_name)
    vae.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

# 5. Visualize Latent Space using UMAP
def visualize_latent_space(encoder, X_test, y_test):
    z_mean, _, _ = encoder.predict(X_test)
    umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='correlation').fit_transform(z_mean)

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(umap_2d[:, 0], umap_2d[:, 1], c=y_test, cmap='Spectral', s=10, alpha=0.75)
    plt.title('UMAP Projection of VAE Latent Space')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.colorbar(scatter)
    plt.show()

# 6. Classification using Latent Space
def classify_latent_space(encoder, X_train, X_test, y_train, y_test):
    z_mean_train, _, _ = encoder.predict(X_train)
    z_mean_test, _, _ = encoder.predict(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_mean_train, y_train)
    y_pred = clf.predict(z_mean_test)

    print(classification_report(y_test, y_pred))

# Main Execution
if __name__ == "__main__":
    # Paths to your data directories (non-segmented images only)
    train_path = "C:/Users/macke/OneDrive/Desktop/OASIS/keras_png_slices_train"
    val_path = "C:/Users/macke/OneDrive/Desktop/OASIS/keras_png_slices_validate"
    test_path = "C:/Users/macke/OneDrive/Desktop/OASIS/keras_png_slices_test"
    
    # Load non-segmented data
    X_train, y_train = load_data(train_path)
    X_val, y_val = load_data(val_path)
    X_test, y_test = load_data(test_path)

    # Build and train VAE model
    vae, encoder, decoder = build_vae(input_shape=(128, 128, 1), latent_dim=2)
    print("\nModel created. Starting training...")
    train_vae(vae, X_train, X_val, epochs=20, model_name="VAE_NonSegmented")
    print("Training completed.")

    # Visualize latent space
    print("\nVisualizing the latent space...")
    visualize_latent_space(encoder, X_test, y_test)

    # Classify using the latent space
    print("\nClassifying images based on latent space...")
    classify_latent_space(encoder, X_train, X_test, y_train, y_test)

    print("Script execution completed.")
