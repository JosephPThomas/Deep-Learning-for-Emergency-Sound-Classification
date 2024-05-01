import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import librosa
import os

# Define a function to load audio files from multiple directories
def load_audio_files(directories, target_length=None):
    audio_files = []
    for directory in directories:
        for filename in os.listdir(directory):
            if filename.endswith('.wav'):  # Assuming audio files are in .wav format
                file_path = os.path.join(directory, filename)
                audio, _ = librosa.load(file_path, duration=1)  # Load audio file

                # Pad or truncate audio to target length
                if target_length is not None:
                    if len(audio) < target_length:
                        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
                    elif len(audio) > target_length:
                        audio = audio[:target_length]

                audio_files.append(audio)
    return audio_files

# List of directories for clean and noisy audio datasets
clean_audio_dirs = ['dataset/classification_training/alarm_tsunami', 'dataset/classification_training/alarm_danger', 'dataset/classification_training/alarm_gas', 'dataset/classification_training/alarm_fire']
noisy_audio_dirs = ['dataset/denoise_training/alarm_tsunami_noise', 'dataset/denoise_training/alarm_danger_noise', 'dataset/denoise_training/alarm_gas_noise', 'dataset/denoise_training/alarm_fire_noise']

# Define a common length for audio files
target_length = 16000  # Example: 16000 samples for 1 second of audio at 16 kHz

# Load clean audio dataset
clean_audio = load_audio_files(clean_audio_dirs, target_length=target_length)

# Load corresponding noisy audio dataset
noisy_audio = load_audio_files(noisy_audio_dirs, target_length=target_length)

# Convert lists to numpy arrays
clean_audio = np.array(clean_audio)
noisy_audio = np.array(noisy_audio)

# Reshape input data to include a channel dimension
clean_audio = clean_audio[..., np.newaxis]  # Add a new axis for the channel
noisy_audio = noisy_audio[..., np.newaxis]

# Print shapes to verify dimensions
print("Shape of clean audio dataset:", clean_audio.shape)
print("Shape of noisy audio dataset:", noisy_audio.shape)

# Define the denoising autoencoder model
def denoising_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    encoded = layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)
    encoded = layers.MaxPooling1D(2, padding='same')(encoded)
    
    # Decoder
    decoded = layers.Conv1D(64, 3, activation='relu', padding='same')(encoded)
    decoded = layers.UpSampling1D(2)(decoded)
    decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same')(decoded)
    
    # Model
    autoencoder = models.Model(inputs, decoded)
    return autoencoder

# Define input shape (length of audio samples)
input_shape = noisy_audio.shape[1:]  # Input shape is determined by the shape of noisy audio samples

# Create the denoising autoencoder model
model = denoising_autoencoder(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with noisy audio as input and clean audio as output
history = model.fit(noisy_audio, clean_audio, epochs=15, batch_size=32, shuffle=True, verbose=1)

# Print loss information
print("Loss:", history.history['loss'])

# Save the model in the native Keras format
model.save('denoising_autoencoder_model.keras')