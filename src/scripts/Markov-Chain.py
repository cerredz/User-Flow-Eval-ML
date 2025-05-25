import numpy as np
from pprint import pprint
from collections import defaultdict
import time

DEBUG = False
# load the data that is necessary to do the markov chaining algorithm
def load_data():
    data = np.genfromtxt("../../data/user-traversals.csv", delimiter=",", skip_header=1, dtype=None, encoding="utf-8")
    usernames = [str(row[0]) for row in data]
    paths = [str(row[1]).split("->") for row in data]
    first_order_paths  = [p for p in paths if len(p) > 1]
    second_order_paths = [p for p in paths if len(p) > 2]
    third_order_paths  = [p for p in paths if len(p) > 3]

    if DEBUG:
        print("Usernames:")
        pprint(usernames)
        print("\nFirst Order Paths:")
        pprint(first_order_paths)
        print("\nSecond Order Paths:")
        pprint(second_order_paths)
        print("\nThird Order Paths:")
        pprint(third_order_paths)

    return usernames, first_order_paths, second_order_paths, third_order_paths

# function to setup the transition and state counts in order to compute the probabilities needed for the markov chain
def setup_counts(data: list[list[str]], transition_counts: dict, state_counts: dict, n_order: int):
    for path in data:
        for j in range(n_order, len(path)):
            state = tuple(path[j-n_order:j])  # tuple so it can be a dict key
            next_page = path[j]
            if DEBUG:
                print(f"State: {state} -> Next: {next_page}")

            state_counts[n_order][state] += 1
            transition_counts[n_order][state][next_page] += 1

    if DEBUG:
        print("\nState Counts:")
        pprint(state_counts)
        print("\nTransition Counts:")
        pprint(transition_counts)
    

# function to compute the probabilites, given the transition and state counts needed for the markov chain
def compute_probabilites(probabilites: dict, transition_counts: dict, state_counts: dict):

    # loop through each order of counts
    for key, value in transition_counts.items():
        # loop through each dictionary in each order
        for page, transition in value.items():
            # loop through each destination page in each dictionary of each order
            for next_page, count in transition.items():
                state_key = tuple(page + tuple([next_page]))
                probabilites[key][state_key] = count / state_counts[key][page] if state_counts[key][page] != 0 else 0.0


    if DEBUG:
        print("\nProbabilities:")
        pprint(probabilites)


if __name__ == "__main__":

    start = time.time()
    # define markov chain order
    orders = [1,2, 3]

    # setup dictionaries for markov chains
    transition_counts = {order: defaultdict(lambda: defaultdict(int)) for order in orders}
    state_counts = {order: defaultdict(int) for order in orders}

    # load in our data 
    usernames, first_order_paths, second_order_paths, third_order_paths = load_data()

    # setup the transition/state counts for each order of the markov chain
    setup_counts(first_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=1)
    setup_counts(second_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=2)
    setup_counts(third_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=3)

    # compute the resulting markov chain probabilites using the counts
    probabilities = {order: defaultdict(float) for order in orders}
    compute_probabilites(probabilites=probabilities, transition_counts=transition_counts, state_counts=state_counts)

    # save the results into the data folder

    end = time.time()
    print(f"🟢 Computed Markov Chain Algorithm in: {end - start:.4f} seconds")
    



    


    