import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import numpy as np
from helpers.destination_pages import get_desired_pages, get_undesired_pages
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

def load_data():
    prob_data = np.genfromtxt("../../data/probability-data.csv", delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    user_data = np.genfromtxt("../../data/user-traversals.csv", delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    usernames = [row[0] for row in user_data]
    paths = [str(row[1]).strip().split("->") for row in user_data]
    paths = [[page.strip() for page in path if page.strip()] for path in paths]
    paths = [path for path in paths if len(path) >= 2]
    
    probabilities = {1: {}, 2: {}, 3: {}}
    for row in prob_data:
        order = row[0]
        path = row[1].strip()
        probability = row[2]
        probabilities[order][path] = probability

    return probabilities, usernames, paths

def get_training_data(paths, probabilities):
    x = [] # features
    y = [] # labels
    default_prob = 0.05
    weights = {1: 1.0, 2: .7, 3: .4}

    all_pages = sorted(set(page for path in paths for page in path))
    for path in paths:
        for i in range(len(path) - 1):
            # first order
            last_page = path[i]
            next_page = path[i + 1]
            probs_1 = []
            for page in all_pages:
                key = f"{last_page}->{page}"
                prob = probabilities[1].get(key, default_prob)
                probs_1.append(prob * weights[1])

            # Second-order: Last 2 pages
            probs_2 = [default_prob * weights[2]] * len(all_pages)
            if i >= 1:
                last_2_pages = '->'.join(path[i-1:i+1])
                for j, page in enumerate(all_pages):
                    key = f"{last_2_pages}->{page}"
                    prob = probabilities[2].get(key, default_prob)
                    probs_2[j] = prob * weights[2]

            # Third-order: Last 3 pages
            probs_3 = [default_prob * weights[3]] * len(all_pages)
            if i >= 2:
                last_3_pages = '->'.join(path[i-2:i+1])
                for j, page in enumerate(all_pages):
                    key = f"{last_3_pages}->{page}"
                    prob = probabilities[3].get(key, default_prob)
                    probs_3[j] = prob * weights[3]

            # Combine all probabilities into one list
            features = probs_1 + probs_2 + probs_3
            x.append(features)

             # Label: 1 if desired, 0 if undesired
            desired = get_desired_pages().get(last_page, [])
            undesired = get_undesired_pages().get(last_page, [])
            y.append(1 if next_page in desired else 0)

    print("Length of x:", len(x))
    print("Length of y:", len(y))
    return np.array(x), np.array(y), all_pages

def run_neural_network(x, y):
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)
    
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=10000, random_state=42)
    model.fit(X_train_res, Y_train_res)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    threshold = 0.2
    y_pred = (y_pred_proba > threshold).astype(int)
    print("Actual Labels:", Y_test)
    print("Predicted Labels:", y_pred)
    print("Classification Report:")
    print(classification_report(Y_test, y_pred, zero_division=0))
    print("Predicted Probabilities for Desired Flow (Test Set):", y_pred_proba)
    return model, scaler

def predict_desired_probability(current_path, probabilities, model, all_pages, default_prob=.1):
    last_page = current_path[-1]
    probs_1 = [probabilities[1].get(f"{last_page}->{page}", default_prob) * 1.0 for page in all_pages]

    probs_2 = [default_prob * 0.7] * len(all_pages)
    if len(current_path) >= 2:
        last_2_pages = '->'.join(current_path[-2:])
        probs_2 = [probabilities[2].get(f"{last_2_pages}->{page}", default_prob) * 0.7 for page in all_pages]
    
    probs_3 = [default_prob * 0.4] * len(all_pages)
    if len(current_path) >= 3:
        last_3_pages = '->'.join(current_path[-3:])
        probs_3 = [probabilities[3].get(f"{last_3_pages}->{page}", default_prob) * 0.4 for page in all_pages]

    features = np.array(probs_1 + probs_2 + probs_3).reshape(1, -1)
    proba = model.predict_proba(features)[0, 1]
    
    return proba


if __name__ == "__main__":
    probabilities, usernames, paths = load_data()
    
    x, y, all_pages = get_training_data(paths, probabilities)
    model, scaler = run_neural_network(x, y)


    