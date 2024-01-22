# ========================================================================================
# ========================================================================================
# ========================================================================================
# ============                                                       =====================
# ============                                                       =====================
# ============                                                       =====================
# ============       PLEASE DO NOT MODIFY THIS FILE                  =====================
# ============                                                       =====================
# ============                                                       =====================
# ============                                                       =====================
# ========================================================================================
# ========================================================================================
# ========================================================================================

from collections import namedtuple
import math
import random
import sys
from typing import List

# Define a point
Point = namedtuple("Point", ["x", "y"])


def length(point1: Point, point2: Point) -> float:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]
    This function calculates the distance between points point1 and point2

    param: point1: point 1
    param: point2: point 2

    return: Euclidean distance between point1 and point2
    """

    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def tour_cost(solution: List, points: List) -> float:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]
    This function calculates the cost of a tour, this is the cost of visiting a set of
    points in the given order

    param: points: list of Point objects

    return: cost of tour as sum of distances between consecutive points
    """

    # calculate the length of the tour
    nodeCount = len(solution)
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount - 1):
        obj += length(points[solution[index]], points[solution[index + 1]])

    return obj


def read_data(file_location: str) -> List:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]
    This function reads the input data using the specified location

    param: file_location: relative path where the input is stored
    return: list of points
    """

    input_data_file = open(file_location, "r")
    input_data = input_data_file.read()

    # parse the input
    lines = input_data.split("\n")

    # Read number of points
    nodeCount = int(lines[0])

    # Ingest points
    points = []
    for i in range(1, nodeCount + 1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))

    # Return
    return points


def trivial_solution(points: List) -> List:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]
    This function builds a trivial solution for the points to visit

    param: points: list of points to visit

    return: trivial solution, initial sequence
    """

    # build a trivial solution
    # visit the nodes in the order they appear in the file
    nodeCount = len(points)
    solution = range(0, nodeCount)

    # Return
    return solution


def write_solution(
    output_directory: str, input_file: str, total_cost: str, solution: List
) -> None:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]
    This function writes the solution as computed in the heuristic
    """
    # Split input file
    input_name = input_file.split("/")[-1]

    with open(f"{output_directory}/sequence_{input_name}_{total_cost}", "w") as fp:
        for item in solution:
            # write each item on a new line
            fp.write("%s\n" % item)
