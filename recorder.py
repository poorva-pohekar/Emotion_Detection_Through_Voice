import sounddevice as sd
from scipy.io.wavfile import write

def record_audio(filename="input.wav", duration=4, fs=44100):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)
    print("Recording saved.")
