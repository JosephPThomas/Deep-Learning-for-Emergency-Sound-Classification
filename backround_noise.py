import os
import librosa
from audiomentations import AddBackgroundNoise
import soundfile as sf  # This is used instead of librosa for writing WAV files

# Function to add background noise to an audio file
def add_background_noise_to_audio(input_audio_path, output_audio_path, background_noise_files, min_snr_in_db=0, max_snr_in_db=20):
    # Load audio file
    audio, sr = librosa.load(input_audio_path, sr=None)
    
    # Define augmentation pipeline with background noise
    augment = AddBackgroundNoise(sounds_path=background_noise_files, min_snr_in_db=min_snr_in_db, max_snr_in_db=max_snr_in_db, p=1.0)
    
    # Apply augmentation
    augmented_audio = augment(samples=audio, sample_rate=sr)
    
    # Save augmented audio to file
    sf.write(output_audio_path, augmented_audio, sr)

# Directory containing the audio files
input_directory = "dataset/classification_training"

# Directory containing the background noise audio files
background_noise_directory = "dataset/background_noise"

# Output directory to save augmented audio files
output_directory = "dataset/denoise_training"

# List background noise audio files
background_noise_files = [os.path.join(background_noise_directory, filename) for filename in os.listdir(background_noise_directory) if filename.endswith(".wav")]

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Minimum and maximum SNR for background noise
min_snr_in_db = 0
max_snr_in_db = 20

# Iterate over all files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_audio_path = os.path.join(input_directory, filename)
        output_audio_path = os.path.join(output_directory, filename)
        
        # Add background noise to the audio file
        add_background_noise_to_audio(input_audio_path, output_audio_path, background_noise_files, min_snr_in_db, max_snr_in_db)
        print(f"Added background noise to {filename}")

print("Background noise addition to all audio files complete.")