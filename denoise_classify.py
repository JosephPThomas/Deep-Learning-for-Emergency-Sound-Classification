import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import os
from pydub import AudioSegment
# Load the denoising model
denoising_model = load_model('denoising_autoencoder_model.keras')

# Load the classification model
classification_model = load_model('model_alarm_classification_RNN.h5')

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

# Function to preprocess audio and extract MFCC features
def preprocess_audio(audio_data, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).T, axis=0)
    return np.expand_dims(mfcc, axis=0)

# Function to classify audio in real-time
def classify_combined_audio(audio_data, model, segment_duration=1):
    # Preprocess the segment
    processed_audio = preprocess_audio(audio_data, 16000)
    
    # Predict the class probabilities
    class_probabilities = model.predict(processed_audio)
    
    # Get the predicted class label
    predicted_class_index = np.argmax(class_probabilities)
    
    # Map the predicted index to the actual class label
    class_labels = ['danger', 'fire', 'gas', 'non', 'tsunami']
    predicted_class_label = class_labels[predicted_class_index]   
    print(f"Predicted Label:", predicted_class_label)
    print("Class Probabilities:", class_probabilities)
    return predicted_class_label

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
        # Classify the denoised audio
        clean_audio, sr = librosa.load(denoised_audio_path, sr=None)
        clean_audio_segment = AudioSegment(
            data=clean_audio.tobytes(),
            sample_width=clean_audio.dtype.itemsize,
            frame_rate=sr,
            channels=1  
        )
        # Increase the volume by 6 dB (adjust the value as needed)
        clean_audio_segment = clean_audio_segment + 6

        # Export the denoised audio to a new file
        clean_audio_segment.export("denoised_audio_louder.wav", format="wav")
        clean_audio, sr = librosa.load("denoised_audio_louder.wav", sr=None)
        predicted_class_label = classify_combined_audio(clean_audio, classification_model)

