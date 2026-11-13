from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

class Utility():
    pass

class Problem():
    def __init__(self, L, I):
        """
        Init the Problem instance with standard utilities and similar
        intervals for all allocations.
        """
        pass

    def set_parameters(self, survival, endowment):
        """
        Asserts the dimentions of survival and endowment matrices 
        match. Verify the correspondent I and L (number of agents and
        number of goods). Set the survival and endowment intervals.


        """

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
        pass

    def add_relaxation(self):
        pass

    def solve_relaxations(self):
        pass

