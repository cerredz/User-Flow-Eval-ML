import numpy as np

# load the data that is necessary to do the markov chaining algorithm
def load_data():
    data = np.genfromtxt("../../data/user-traversals.csv", delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    usernames= [str(row[0]) for row in data]
    paths_strs = [str(row[1]) for row in data]

    # split each path into a list of pages, using "->" as the delimiter
    paths = [path_str.split("->") for path_str in paths_strs]

    return usernames, paths

    

# compute the markov chain probabilies for the user flow data
def compute_markov_chain(paths: list, order: int):
    pass


if __name__ == "__main__":

    orders = [1,2, 3]
    probabilities = {order: {} for order in orders}
    print(load_data())