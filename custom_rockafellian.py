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
        self.survival_floor = survival_floor
        self.satiation_ceiling = satiation_ceiling
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

    # def add_relaxation(self):
    #     matrix = np.zeros((self.goods, self.agents*self.goods))
    #     for l in range(self.goods):
    #         matrix[l][l::self.agents] = 1
    #     amount = self.goods*[0]
    #     for l in range(self.goods):
    #         amount[l] = sum(self.endowments[l::self.goods])
    #     lb = amount
    #     ub = amount
    #     print(matrix)
    #     print(amount)
    #     linear_constraint = LinearConstraint(matrix, lb, ub)
        
    #     self.relaxed_constraints = []
    #     self.relaxed_constraints.append(linear_constraint)

    def solve_relaxation(self, endowments, coeficients, survival_floor, penalty, verbose = False):
        # matrix = np.zeros((self.goods, self.agents*self.goods))
        # for l in range(self.goods):
        #     matrix[l][l::self.agents] = 1
        # amount = self.goods*[0]
        # for l in range(self.goods):
        #     amount[l] = sum(endowments[l::self.goods])
        # amount = np.array(amount)

        # account = matrix @ amount
        # account = sum(account)
        # if account <= self.v:
        #     pass

        v = np.linalg.norm(endowments - survival_floor)
        total_penalty = (penalty/2)*v

        total = 0
        for i in range(self.agents):
            partial = [(coeficients[i][j][0] * endowments[min((i-1)*self.goods, 0) + j] ** coeficients[i][j][1]) for j in range(self.goods)]
            utility = sum(partial)
            total += utility
            if verbose:
                print(f"For agent {i}, utility was {utility}\n Partials were {partial}")
        if verbose:
            print(f"Total penalty was {total_penalty}")

        return -total + total_penalty

    def run_relaxation(self):
        self.lambdas = np.logspace(0, 7, 36)
        iteration = 0
        tol = 10**(-7)
        not_stationary = True
        self.objective = self.solve_relaxation

        def get_z(x):
            return sum(x - self.survival_floor)
        
        res_trajectory = []
        v_trajectory = []
        while not_stationary:
            self.relaxation_args = (self.args, self.survival_floor, self.lambdas[iteration])
            res = minimize(self.objective, self.endowments, args = self.relaxation_args,
                            method = "trust-constr", bounds = self.survival_bounds)

            v_trajectory.append(get_z(res.x))
            
            if iteration > 0:
                advance = np.linalg.norm(res.x - res_trajectory[-1].x)
                if advance < tol:
                    not_stationary = False
            res_trajectory.append(res)
            iteration += 1
            if iteration >= 36:
                not_stationary = False

        print("Optimal x: ", res.x)
        print(res.success, res.message)
        print(self.objective(res.x, self.args, self.survival_floor, self.relaxation_args[-1], True))

        self.result = {"res_trajectory": res_trajectory,
                       "lambdas": self.lambdas,
                       "iterations": iteration,
                       "diff": advance,
                       "v_trajectory": v_trajectory}

        
            


if __name__ == "__main__":
    print("Starting script")

    args0 = [[0.5, -0.5], [-0.5, 0.5]]
    # Each list carries [multiplier, exponent]
    args = ([[[1, 1], [-1/8, -8]], [[-1/8, -8], [1, 1]]])
    problem = Problem(2, 2, args)
    survival_floor = 2*2*[3]
    satiation_ceiling = 2*2*[30]

    """ Solving feasible problem """
    endowments = [5, 1, 1, 5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()

    """ Reaching unfeasability due to initial endowments """
    endowments = [5, 0.5, 0.5, 5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()

    """ Finding v for increasing penalty weigths """
    problem.run_relaxation()
    result = problem.result
    print(f"Em {result["iterations"]}, {result["v_trajectory"][-1]} was obtained.")