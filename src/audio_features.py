import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3  # seconds


def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            duration=DURATION,
            offset=0.5
        )

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Take mean across time axis
        mfcc = np.mean(mfcc.T, axis=0)

        return mfcc

    except Exception as e:
        print("Error processing file:", file_path)
        return None
