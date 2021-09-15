"""
Calculate RRA based off of Lifecycle Investing excel sheet
"""
import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('max_percent', help='The maximum percent of your net worth you would be willing to put at risk for the chance at doubling your money. Should be between 0 and 50.',
                    type=float)
args = parser.parse_args()


def get_percent_from_rra(rra):
    return 1 - (2 - 2**(1 - rra))**(1 / (1 - rra))


def find_nearest(array: list, value: float):
    """
    Finds the nearest item to value in the array, returning the index.
    """
    
    deltas = np.abs(np.array(array)-value)
    index = np.argmin(deltas)

    return index


percent = args.max_percent / 100

rras = []

increment = 0.1
curr = 1.00000000000003
stop = 10

while curr <= stop:
    rras.append(curr)
    curr += increment

# rras should now be a list of rra values covering the desired range.

percents = [get_percent_from_rra(rra) for rra in rras]

index = find_nearest(percents, percent)

rra = rras[index]
rounded = round(rra, 2)

print(f'RRA: {rounded}')

