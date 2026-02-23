from audio_features import extract_features

file_path = input("Enter audio file path: ")

features = extract_features(file_path)

if features is not None:
    print("Feature shape:", features.shape)
    print("Feature vector:", features)
else:
    print("Feature extraction failed.")
