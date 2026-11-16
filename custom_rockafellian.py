from __future__ import annotations

import matplotlib
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

    def sum_non_linear(self, endowments, coeficients, verbose = False):
        total = 0
        for i in range(self.agents):
            partial = [(coeficients[i][j][0] * endowments[min((i-1)*self.goods, 0) + j] ** coeficients[i][j][1]) for j in range(self.goods)]
            utility = sum(partial)
            total += utility
            if verbose:
                print(f"For agent {i}, utility was {utility}\n Partials were {partial}")

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
        
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.agents] = 1
        amount = self.goods*[0]
        for l in range(self.goods):
            amount[l] = sum(self.endowments[l::self.goods])
        lb = amount
        ub = amount
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
        print(res.success, res.message)
        print(self.objective(res.x, self.args, True))

    def add_relaxation(self):
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.agents] = 1
        amount = self.goods*[0]
        for l in range(self.goods):
            amount[l] = sum(self.endowments[l::self.goods])
        lb = amount
        ub = amount
        linear_constraint = LinearConstraint(matrix, lb, ub)
        
        self.relaxed_constraints = []
        self.relaxed_constraints.append(linear_constraint)

    def solve_relaxations(self, endowments, coeficients, verbose = False):
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.agents] = 1
        amount = self.goods*[0]
        for l in range(self.goods):
            amount[l] = sum(endowments[l::self.goods])
        amount = np.array(amount)

        account = matrix @ amount
        account = sum(account)
        if account <= self.v:
            

        total = 0
        for i in range(self.agents):
            partial = [(coeficients[i][j][0] * endowments[min((i-1)*self.goods, 0) + j] ** coeficients[i][j][1]) for j in range(self.goods)]
            utility = sum(partial)
            total += utility
            if verbose:
                print(f"For agent {i}, utility was {utility}\n Partials were {partial}")

        return -total

if __name__ == "__main__":
    print("Starting script")

    """ TESTANDO CLASSE BOUNDS
    first_b = Bounds(0,0.1)
    second_b = Bounds([1, 2], [4, 3.5])
    third_b = Bounds(3*[2, 1.5])

    print(f"First bound: {first_b}")
    print(f"Second bound: {second_b}")
    print(f"Third bound: {third_b}")"""

    args0 = [[0.5, -0.5], [-0.5, 0.5]]
    # Each list carries [multiplier, exponent]
    args = [[[1, 1], [-1/8, -8]], [[-1/8, -8], [1, 1]]]
    problem = Problem(2, 2, args)
    survival_floor = 2*2*[3]
    satiation_ceiling = 2*2*[30]

    """ Solving feasible problem """
    endowments = [5, 1, 1, 5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()

    """ Reaching unfeasability due to initial endowments """
    endowments = [0.5, 0.05, 0.05, 0.5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()