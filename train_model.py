from ml_detector import MLScreamDetector

def main():
    detector = MLScreamDetector()
    
    # Paths to your training data folders
    scream_folder = "data/screams"
    normal_folder = "data/normal"
    
    print("Starting model training...")
    detector.train(scream_folder, normal_folder)
    print("Model training completed and saved!")

if __name__ == "__main__":
    main()