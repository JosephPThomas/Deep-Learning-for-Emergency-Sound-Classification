import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import os
from pydub import AudioSegment

# Load the denoising model
denoising_model = load_model('denoising_autoencoder_model.keras')

# Function to denoise audio using the model
def denoise_audio(audio, model):
    # Resample the audio to match the model's input shape
    target_length = 16000  # Target length for resampling
    audio = librosa.resample(audio, orig_sr=len(audio), target_sr=target_length)

    # Preprocess audio to match the input shape of the model
    audio = audio[np.newaxis, ..., np.newaxis]

    # Debugging: Print the shape of the input audio
    print("Input audio shape:", audio.shape)

    # Denoise the audio using the model
    denoised_audio = model.predict(audio)

    # Remove the channel dimension and squeeze the denoised audio
    denoised_audio = np.squeeze(denoised_audio, axis=-1)

    return denoised_audio


# Directory path for noisy audio files
directory_path = 'dataset/denoise_training/alarm_tsunami_noise'#modify the directory accordingly

# Iterate through audio files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.wav'):  # Assuming all files are .wav format
        noisy_audio_path = os.path.join(directory_path, filename)
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=None)
        
        # Denoise the audio
        denoised_audio = denoise_audio(noisy_audio, denoising_model)
        
        # Save the denoised audio
        denoised_audio_path = os.path.join('dataset/denoised_audio/alarm_tsunami_denoise', 'denoised_' + filename)#modify the directory accordingly
        sf.write(denoised_audio_path, np.ravel(denoised_audio), sr)
        print("Denoising completed. Denoised audio saved to:", denoised_audio_path)
        denoised_audio = denoise_audio(noisy_audio, denoising_model)
        sf.write(denoised_audio_path, np.ravel(denoised_audio), sr)
        print("Denoising completed. Denoised audio saved to:", denoised_audio_path)