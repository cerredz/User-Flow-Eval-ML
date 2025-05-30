import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Add parent directory to path to import Neural_Network
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.Neural_Network import load_data, get_training_data, run_neural_network

def plot_confusion_matrix(model, x, y):
    y_pred = model.predict(x)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    
    # Ensure the visualization folder exists
    os.makedirs("visualization", exist_ok=True)
    # Save the confusion matrix plot
    plt.savefig("visualization/confusion_matrix.png")
    plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    probabilities, usernames, paths = load_data()
    x, y, all_pages = get_training_data(paths, probabilities)
    model, scaler = run_neural_network(x, y)
    plot_confusion_matrix(model, x, y)

    
    




