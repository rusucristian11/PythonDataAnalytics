import soundfile as sf

def get_sample_rate(audio_path):
    audio_data, sample_rate = sf.read(audio_path)
    return sample_rate

# Usage example
sample_rate = get_sample_rate('audio.wav')
print(f"The sample rate of the audio file is: {sample_rate} Hz")
