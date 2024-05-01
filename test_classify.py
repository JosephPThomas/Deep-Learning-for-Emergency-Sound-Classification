import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.models import load_model
import os

# Load the trained model
model = load_model('model_alarm_classification_RNN.h5')

# Function to preprocess audio and extract MFCC features
def preprocess_audio(audio_data, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13).T, axis=0)
    return np.expand_dims(mfcc, axis=0)

# Function to classify audio in real-time
def classify_combined_audio(audio_file, segment_duration=1):
    # Load audio file
    audio_data, sr = librosa.load(audio_file, sr=None)
    
    # Calculate the number of segments based on segment duration
    num_segments = len(audio_data) // (sr * segment_duration)
    
    # Iterate through each segment and classify
    for i in range(num_segments):
        start_idx = i * sr * segment_duration
        end_idx = (i + 1) * sr * segment_duration
        segment_audio = audio_data[start_idx:end_idx]
        
        # Preprocess the segment
        processed_audio = preprocess_audio(segment_audio, sr)
        
        # Predict the class probabilities
        class_probabilities = model.predict(processed_audio)
        
        # Get the predicted class label
        predicted_class_index = np.argmax(class_probabilities)
        
        # Map the predicted index to the actual class label
        class_labels = ['danger', 'fire', 'gas', 'non', 'tsunami']
        predicted_class_label = class_labels[predicted_class_index]
        
        print(f"Segment {i+1} - Predicted Label:", predicted_class_label)
        print("Class Probabilities:", class_probabilities)
        return predicted_class_label

# Directory path for noisy audio files
directory_path = 'dataset/denoised_audio/alarm_tsunami_denoise'
# Iterate through audio files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.wav'):  # Assuming all files are .wav format
        audio_path = os.path.join(directory_path, filename)
        # Classify the denoised audio
        predicted_class_label = classify_combined_audio(audio_path)