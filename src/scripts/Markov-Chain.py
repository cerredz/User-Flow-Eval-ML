import numpy as np
from pprint import pprint
from collections import defaultdict

DEBUG = True
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
    


# compute the markov chain probabilies for the user flow data
def compute_markov_chain(paths: list, n_order: int):
    pass


if __name__ == "__main__":

    orders = [1,2, 3]
    transition_counts = {order: defaultdict(lambda: defaultdict(int)) for order in orders}
    state_counts = {order: defaultdict(int) for order in orders}
    usernames, first_order_paths, second_order_paths, third_order_paths = load_data()

    setup_counts(first_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=1)
    setup_counts(second_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=2)
    setup_counts(third_order_paths, transition_counts=transition_counts, state_counts=state_counts, n_order=3)

    


    