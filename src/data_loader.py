import os

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"

def show_dataset_structure():
    print("Training classes:")
    print(os.listdir(TRAIN_DIR))

    print("\nTesting classes:")
    print(os.listdir(TEST_DIR))

if __name__ == "__main__":
    show_dataset_structure()
