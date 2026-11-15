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
    def __init__(self, goods: int, agents: int):
        """
        Init the Problem instance with standard utilities and similar
        intervals for all allocations.
        """
        self.goods = goods
        self.agents = agents
        lb = self.agents*self.goods*[0]
        ub = self.agents*self.goods*[self.agents]
        self.survival_bounds = Bounds(lb, ub)
        matrix = self.goods*[self.agents*[1]]
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.agents] = 1
        lb = self.goods*[self.agents]
        ub = self.goods*[self.agents]
        linear_constraint = LinearConstraint(matrix, lb, ub)
        print(matrix, lb, ub)
        self.constraints = []
        self.constraints.append(linear_constraint)
        self.endowments = np.ones(self.agents*self.goods)
        # self.args = self.agents*[self.goods*[0.5]]
        self.args = [[0.5, -0.5], [-0.5, 0.5]]
        self.objective = self.sum_linear

    def sum_linear(self, endowments, coeficients):
        total = 0
        for i in range(self.agents):
            total += sum([endowments[min((i-1)*self.goods, 0) + j] * coeficients[i][j] for j in range(self.goods)])
        return -total

    def set_parameters(self, survival, endowment):
        """
        Asserts the dimentions of survival and endowment matrices 
        match. Verify the correspondent I and L (number of agents and
        number of goods). Set the survival and endowment intervals.


        """
        self.survival_bounds = Bounds()

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

    problem = Problem(2, 2)
    problem.solve_problem()