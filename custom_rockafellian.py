from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

class Utility():
    pass

class Problem():
    def __init__(self, goods: int, agents: int, args: list[list]):
        """
        Init the Problem instance with standard utilities and similar
        intervals for all allocations.
        """
        self.goods = goods
        self.agents = agents
        self.args = args
        self.objective = self.sum_non_linear

    def sum_linear(self, endowments, coeficients):
        total = 0
        for i in range(self.agents):
            total += sum([endowments[min((i-1)*self.goods, 0) + j] * coeficients[i][j] for j in range(self.goods)])
        return -total

    def sum_non_linear(self, endowments, coeficients):
        total = 0
        for i in range(self.agents):
            total += sum([endowments[min((i-1)*self.goods, 0) + j] ** coeficients[i][j][1] * coeficients[i][j][0] for j in range(self.goods)])
        return -total

    def set_variables(self, survival_floor, satiation_ceiling, endowments):
        """
        Asserts the dimentions of survival and endowment matrices 
        match. Verify the correspondent I and L (number of agents and
        number of goods). Set the survival and endowment intervals.


        """
        self.survival_bounds = Bounds(survival_floor, satiation_ceiling)
        self.endowments = endowments

    def add_linear_constraints(self):
        
        matrix = self.goods*[self.agents*[1]]
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.agents] = 1
        lb = self.goods*[self.agents]
        ub = self.goods*[self.agents]
        linear_constraint = LinearConstraint(matrix, lb, ub)
        
        self.constraints = []
        self.constraints.append(linear_constraint)

    def _set_utilities(self, utilities):
        """
        Set the Utility instances corresponding to the entered params.
        """
        pass

    def _excess_demand(self):
        pass

    def _excess_demand_price(self):
        pass

    def solve_problem(self):
        res = minimize(self.objective, self.endowments, args = self.args,
                       method = "trust-constr", bounds = self.survival_bounds,
                       constraints = self.constraints)
        print(res.x)
        print(res.)

    def add_relaxation(self):
        pass

    def solve_relaxations(self):
        pass

if __name__ == "__main__":
    print("Starting script")

    # first_b = Bounds(0,0.1)
    # second_b = Bounds([1, 2], [4, 3.5])
    # third_b = Bounds(3*[2, 1.5])

    # print(f"First bound: {first_b}")
    # print(f"Second bound: {second_b}")
    # print(f"Third bound: {third_b}")

    args0 = [[0.5, -0.5], [-0.5, 0.5]]
    # Each list carries [multiplier, exponent]
    args = [[[1, 1], [-1/8, -8]], [[-1/8, -8], [1, 1]]]
    problem = Problem(2, 2, args)
    survival_floor = 2*2*[0.3]
    satiation_ceiling = 2*2*[3]
    endowments = [0.5, 0.05, 0.05, 0.5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()