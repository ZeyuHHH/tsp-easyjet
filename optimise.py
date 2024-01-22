# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================
# ============                                                       =====================
# ============                                                       =====================
# ============                                                       =====================
# ============       WARNING FOR SUBMISSIONS                         =====================
# ============       Please modify only the code for the             =====================
# ============       function "custom_heuristic"                     =====================
# ============                                                       =====================
# ============                                                       =====================
# ============                                                       =====================
# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================
# ========================================================================================

from collections import namedtuple
import math
import random
import sys
from typing import List
from tsp_utils import length, tour_cost, read_data, Point, write_solution
from lk_heuristic.nodes import Node2D
from lk_heuristic.tsp import Tsp


def custom_heuristic(points: List) -> List:
    """
    This function is the core function to implement

    param: points: points to visit

    return: solution: sequence of point visits
    """

    nodes = []
    for i in range(len(points)):
        nodes.append(Node2D(float(points[i][0]), float(points[i][1])))

    def euc_2d(n1, n2):
        return ((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2) ** 0.5

    best_tour = None  # the best tsp tour nodes found so far
    best_cost = math.inf  # the best cost found so far
    runs = 50
    solution_method = "lk2_improve"

    tsp = Tsp(
        nodes,
        euc_2d,
        shuffle=False,
        backtracking=(5, 5),
        reduction_level=4,
        reduction_cycle=4,
        tour_type="cycle",
    )

    for run in range(1, runs + 1):
        if tsp.shuffle:
            tsp.tour.shuffle()
        tsp.tour.set_cost(tsp.cost_matrix)
        tsp.methods[solution_method]()
        if tsp.tour.cost < best_cost:
            best_tour = tsp.tour.get_nodes()
            best_cost = tsp.tour.cost

    solution = []

    for i in range(1, len(best_tour)):
        solution.append(best_tour[i].id)

    # Return
    return solution


# ========================================================================================
# =============                                                 ==========================
# =============     PLEASE DO NOT MODIFY CODE BELOW THIS LINE   ==========================
# =============                                                 ==========================
# ========================================================================================


def solve_tsp(input_file: str) -> List:
    """
    [PLEASE DO NOT MODIFY THE CODE IN THIS FUNCTION]

    This function runs the following steps
    - Read data (using read_data function from tsp_utils)
    - Runs custom heuristic as implemented by team
    - Evaluates and prints out the cost of the solution

    """
    # Read data
    points = read_data(input_file)

    # Build solution using your custom heuristic
    solution = custom_heuristic(points)

    # Calculate cost of solution
    total_cost = tour_cost(solution, points)

    # prepare the solution in the specified output format
    print("Total cost: %.2f" % total_cost + "\n")
    print("Sequence \n" + " ".join(map(str, solution)))

    return solution


# ================================
# PLEASE DO NOT MODIFY THIS CODE
# ================================
if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Output directory
        output_directory = sys.argv[1].strip()

        # Read input file
        input_file = sys.argv[2].strip()

        # Run optimisation
        solution = solve_tsp(input_file)

        # Write output
        write_solution(output_directory, input_file, solution)

    else:
        print("")
        print("[INPUT ERROR] This script requires two arguments:")
        print(
            "   - The directory to write the output (should be submission_teamX) with X in {1...9}"
        )
        print("   - An input dataset (e.g. ./data/tsp_51)")
        print(
            "Correct call format: $> python optimise.py submission_teamX ./data/tsp_51"
        )
        print("")
