from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint

class WalrasianProblem():
    """Represent and solve resource-allocation problems for multiple agents.
    In the context of walrasian market equilibrium.

    Parameters
    ----------
    goods : int
        Number of distinct goods in the allocation.
    agents : int
        Number of agents sharing the goods.
    args : list[list]
        Utility coefficients for each agent and good.
    """

    def __init__(self, goods: int, agents: int, args: list[list]):
        """Initialize a Walrasian allocation problem.

        Parameters
        ----------
        goods : int
            Number of goods allocated to each agent.
        agents : int
            Number of agents participating in the allocation.
        args : list[list]
            Utility parameters for every agent and good. For nonlinear
            utilities, each entry should contain ``[coefficient, exponent]``.
        """
        self.goods = goods
        self.agents = agents
        self.args = np.asarray(args, dtype=float)
        if self.args.shape != (agents, goods, 2):
            raise ValueError(f"args must have shape {(agents, goods, 2)}, got {self.args.shape}")
        self.objective = self.sum_non_linear

    def sum_linear(self, endowments, coeficients):
        """Calculate the negative aggregate linear utility.

        Parameters
        ----------
        endowments : array-like
            Flattened allocation vector containing the goods assigned to
            every agent.
        coeficients : array-like
            Linear utility coefficient for each agent and good.

        Returns
        -------
        float
            Negative sum of the agents' total linear utilities.
        """
        total = 0
        for i in range(self.agents):
            total += sum([endowments[i*self.goods + j] * coeficients[i][j] for j in range(self.goods)])
        return -total

    def sum_non_linear(self, endowments, coeficients, verbose = False):
        """Calculate the negative aggregate nonlinear utility.

        Parameters
        ----------
        endowments : array-like
            Flattened allocation vector for all agents.
        coeficients : array-like
            Utility coefficients and exponents for each agent and good.
            Each entry must be ``[coefficient, exponent]``.
        verbose : bool, optional
            If ``True``, print each agent's utility and partial utilities.

        Returns
        -------
        float
            Negative sum of all agents' nonlinear utilities.
        """
        total = 0
        for i in range(self.agents):
            partial = [(coeficients[i][j][0] * endowments[i*self.goods + j] ** coeficients[i][j][1]) for j in range(self.goods)]
            utility = sum(partial)
            total += utility
            if verbose:
                print(f"For agent {i}, utility was {utility}\n Partials were {partial}")

        return -total

    def set_variables(self, survival_floor, satiation_ceiling, endowments):
        """Configure allocation bounds and initial endowments.

        Parameters
        ----------
        survival_floor : array-like
            Lower bounds for the allocation of every agent and good.
        satiation_ceiling : array-like
            Upper bounds for the allocation of every agent and good.
        endowments : array-like
            Initial allocation vector used as the optimization seed.

        Raises
        ------
        ValueError
            If the bound or endowment lengths do not match the allocation size.
        """
        expected_length = self.agents * self.goods
        if len(survival_floor) != expected_length:
            raise ValueError(
                f"survival_floor must have length {expected_length}, "
                f"got {len(survival_floor)}"
            )
        if len(satiation_ceiling) != expected_length:
            raise ValueError(
                f"satiation_ceiling must have length {expected_length}, "
                f"got {len(satiation_ceiling)}"
            )
        if len(endowments) != expected_length:
            raise ValueError(
                f"endowments must have length {expected_length}, "
                f"got {len(endowments)}"
            )

        self.survival_floor = survival_floor
        self.satiation_ceiling = satiation_ceiling
        self.survival_bounds = Bounds(survival_floor, satiation_ceiling)
        self.endowments = endowments

    def add_linear_constraints(self):
        """Build resource-conservation constraints for the problem.

        Creates one equality constraint per good, requiring the sum of that
        good across all agents to remain equal to the initial total endowment.
        The resulting constraint is stored in ``self.constraints``.

        Returns
        -------
        None
        """
        
        matrix = np.zeros((self.goods, self.agents*self.goods))
        for l in range(self.goods):
            matrix[l][l::self.goods] = 1
        amount = self.goods*[0]
        for l in range(self.goods):
            amount[l] = sum(self.endowments[l::self.goods])
        lb = [-np.inf] * self.goods
        ub = amount
        linear_constraint = LinearConstraint(matrix, lb, ub)
        
        self.constraints = []
        self.constraints.append(linear_constraint)

    def solve_problem(self):
        """Solve the constrained nonlinear allocation problem.

        Uses SciPy's ``trust-constr`` method with the configured bounds and
        linear constraints. It prints the result and returns the optimizer
        result object.

        Returns
        -------
        scipy.optimize.OptimizeResult
            Result returned by ``scipy.optimize.minimize``.
        """
        res = minimize(self.objective, self.endowments, args = self.args,
                       method = "trust-constr", bounds = self.survival_bounds,
                       constraints = self.constraints)
        print(res.x)
        print(res.success, res.message)
        print(self.objective(res.x, self.args, True))
        return res

    def sum_non_linear_penalized(self, endowments, coeficients, initial_endowment, penalty, verbose = False):
        """Calculate negative nonlinear utility plus a quadratic-style penalty.

        Parameters
        ----------
        endowments : array-like
            Flattened allocation vector for all agents.
        coeficients : array-like
            Utility coefficients and exponents for each agent and good.
        initial_endowment : array-like
            Reference lower-bound vector used to measure constraint violation.
        penalty : float
            Nonnegative penalty weight applied to the distance from the floor.
        verbose : bool, optional
            If ``True``, print agent utilities and the total penalty.

        Returns
        -------
        float
            Negative aggregate utility plus the penalization term.
        """
        v = np.sum(np.maximum(sum(endowments - initial_endowment), 0)**2)
        total_penalty = (penalty/2)*v

        total = 0
        for i in range(self.agents):
            partial = [(coeficients[i][j][0] * endowments[i*self.goods + j] ** coeficients[i][j][1]) for j in range(self.goods)]
            utility = sum(partial)
            total += utility
            if verbose:
                print(f"For agent {i}, utility was {utility}\n Partials were {partial}")
        if verbose:
            print(f"Total penalty was {total_penalty}")

        return -total + total_penalty

    def run_relaxation(self):
        """Trace solutions across an increasing sequence of penalty weights.

        Repeatedly solves the penalized problem using logarithmically spaced
        penalty weights. The routine stops when successive allocations
        converge or all weights have been evaluated.

        Returns
        -------
        None

        Notes
        -----
        The trajectories and final iteration details are stored in
        ``self.result``.
        """
        self.lambdas = np.logspace(0, 7, 36)
        iteration = 0
        tol = 10**(-7)
        not_stationary = True
        self.relaxed_objective = self.sum_non_linear_penalized

        def get_v(x):
            v = np.sum(np.maximum(sum(x - self.endowments), 0)**2)
            return v
        
        res_trajectory = []
        v_trajectory = []
        initial_x = self.endowments
        while not_stationary:
            self.relaxation_args = (self.args, self.endowments, self.lambdas[iteration])
            res = minimize(self.relaxed_objective, initial_x, args = self.relaxation_args,
                            method = "trust-constr", bounds = self.survival_bounds)

            v_trajectory.append(get_v(res.x))
            initial_x = res.x
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
        print(self.relaxed_objective(res.x, self.args, self.endowments, self.relaxation_args[-1], True))

        self.result = {"res_trajectory": res_trajectory,
                       "lambdas": self.lambdas,
                       "iterations": iteration,
                       "diff": advance,
                       "v_trajectory": v_trajectory}


def various_plots(problem, result_relaxed, result_original, endowments, survival_floor, args, ideal):
    n = result_relaxed["iterations"]
    lam = result_relaxed["lambdas"][:n]
    traj = result_relaxed["res_trajectory"]
    X = np.array([r.x for r in traj])                  
    e = np.array(endowments).reshape(2, 2)              
    fl = np.array(survival_floor).reshape(2, 2)
    Z = X.reshape(n, 2, 2).sum(axis=1) - e.sum(axis=0)     
    V = np.maximum(Z, 0)                            
    v_floor = np.maximum(fl.sum(axis=0) - e.sum(axis=0), 0)
    r2 = np.linalg.norm(v_floor)
    U = np.array([-problem.sum_non_linear(x, args) for x in X])
    c = ["#2f6f4f", "#d1603d", "#3b6ea8", "#8a5aa8"]

    fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    ax = axs[0, 0]
    for l in range(2):
        ax.plot(lam, V[:, l], "o-", ms=3, color=c[l], label=f"v good {l+1}")
    ax.axhline(v_floor[0], color="r", ls="--", lw=1, label=f"analytic floor {v_floor[0]:.2f}")
    ax.set_xscale("log"); ax.set_xlabel("Penalty weight"); ax.set_ylabel("v = Z(x)+")
    ax.set_title("Residual components"); ax.legend(frameon=False)

    ax = axs[0, 1]
    ax.plot(lam, V.sum(1), "o-", ms=3, color=c[0], label="l1 norm")
    ax.plot(lam, np.linalg.norm(V, axis=1), "o-", ms=3, color=c[1], label="l2 norm")
    ax.plot(lam, V.max(1), "o-", ms=3, color=c[2], label="linf norm")
    ax.plot(lam, result_relaxed["v_trajectory"], "k--", lw=1, label="class v_trajectory (sum(x - x0))")
    for lim, col in ((v_floor.sum(), c[0]), (r2, c[1]), (v_floor.max(), c[2])):
        ax.axhline(lim, color=col, ls=":", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("Penalty weight")
    ax.set_title(f"Norms vs analytic floors ({v_floor.sum():.2f}, {r2:.4f}, {v_floor.max():.2f})")
    ax.legend(frameon=False)

    ax = axs[0, 2]
    k = 0
    for i in range(2):
        for j in range(2):
            ax.plot(lam, X[:, k], "o-", ms=3, color=c[k], label=f"x[agent {i}, good {j}]"); k += 1
    ax.axhline(fl[0, 0], color="k", ls="--", lw=0.8, label="survival floor")
    ax.set_xscale("log"); ax.set_xlabel("Penalty weight"); ax.set_ylabel("allocation")
    ax.set_title("Allocation path"); ax.legend(frameon=False, ncol=2)

    ax = axs[1, 0]
    sc = ax.scatter(np.linalg.norm(V, axis=1), U, c=np.log10(lam), cmap="viridis", s=28)
    ax.plot(np.linalg.norm(V, axis=1), U, color="gray", lw=0.6, zorder=0)
    fig.colorbar(sc, ax=ax, label="log10(lambda)")
    ax.set_xlabel("||v||_2"); ax.set_ylabel("U(x)")
    ax.set_title("Welfare vs feasibility")

    ax = axs[1, 1]
    ax.plot(lam, [r.nit for r in traj], "o-", ms=3, color=c[3], label="iterations (nit)")
    ax.set_xscale("log"); ax.set_xlabel("Penalty weight"); ax.set_ylabel("nit")
    ax2 = ax.twinx()
    ax2.plot(lam, [r.optimality for r in traj], "s-", ms=3, color=c[1], label="optimality")
    ax2.set_yscale("log"); ax2.set_ylabel("optimality")
    bad = [i for i, r in enumerate(traj) if not r.success]
    ax.plot(lam[bad], np.array([traj[i].nit for i in bad]), "rx", ms=9, label="success=False")
    ax.set_title("trust-constr diagnostics")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False)

    ax = axs[1, 2]
    step = np.linalg.norm(np.diff(X, axis=0), axis=1)
    ax.loglog(lam[1:], np.maximum(step, 1e-16), "o-", ms=3, color=c[0], label="||x_k - x_{k-1}||")
    ax.axhline(1e-7, color="r", ls="--", lw=1, label="stopping tol 1e-7")
    ax.set_xlabel("Penalty weight"); ax.set_title("Step between successive lambdas"); ax.legend(frameon=False)

    for a in axs.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("Relaxation: stressed economy", fontsize=14)
    fig.savefig("relaxation_dashboard.png", dpi=150); plt.close(fig)

    # Feasible case + feasibility ratio of the three scenarios
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    ax = axs[0]
    labels = [f"agent {i}\ngood {j}" for i in range(2) for j in range(2)]
    idx = np.arange(4); w = 0.27
    ax.bar(idx - w, [5, 2, 2, 5], w, label="endowment", color="#b0b7c3")
    ax.bar(idx, result_original.x, w, label="solver (res.x)", color=c[0])
    ax.bar(idx + w, ideal, w, label="ideal", color=c[1])
    ax.axhline(3, color="k", ls="--", lw=0.8, label="survival floor")
    ax.set_xticks(idx); ax.set_xticklabels(labels); ax.legend(frameon=False)
    ax.set_title("Feasible case")
    ax = axs[1]
    scen = {"feasible": ([5, 2, 2, 5], 3), "infeasible": ([5, 0.5, 0.5, 5], 3),
            "stressed": ([0.5, 0.05, 0.05, 0.5], 0.3)}
    ratios = [np.array(en).reshape(2, 2).sum(0)[0] / (2 * fl_) for en, fl_ in scen.values()]
    ax.bar(list(scen), ratios, color=[c[0] if r_ >= 1 else c[1] for r_ in ratios])
    ax.axhline(1, color="k", ls="--", lw=0.8)
    ax.set_ylabel("aggregate endowment / aggregate floor (per good)")
    ax.set_title("Feasible iff ratio >= 1")
    for a in axs:
        a.grid(axis="y", alpha=0.3)
    fig.savefig("feasibility_overview.png", dpi=150); plt.close(fig)
    print("Saved: plot.png, relaxation_dashboard.png, feasibility_overview.png")



if __name__ == "__main__":
    print("Starting script")

    # Each list carries [multiplier, exponent]
    args = ([[[1, 1], [-1/8, -8]], [[-1/8, -8], [1, 1]]])
    problem = WalrasianProblem(2, 2, args)
    survival_floor = 2*2*[3]
    satiation_ceiling = 2*2*[30]

    """ Solving feasible problem """
    endowments = [5, 2, 2, 5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    result_1 = problem.solve_problem()
    ideal = [4, 3, 3, 4]
    print("Obtained: ", problem.sum_non_linear(result_1.x, args))
    print("Best: ", problem.sum_non_linear(ideal, args))

    """ Reaching unfeasability due to initial endowments """
    endowments = [5, 0.5, 0.5, 5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.add_linear_constraints()
    problem.solve_problem()

    """ Finding v for increasing penalty weigths """
    args = ([[[1, 1], [-1/8, -8]], [[-1/8, -8], [1, 1]]])
    survival_floor = 2*2*[0.3]
    satiation_ceiling = 2*2*[3]
    problem = WalrasianProblem(2, 2, args)
    endowments = [0.5, 0.05, 0.05, 0.5]
    problem.set_variables(survival_floor, satiation_ceiling, endowments)
    problem.run_relaxation()
    result_2 = problem.result
    print(f"Em {result_2['iterations']}, {result_2['v_trajectory'][-1]} was obtained.")

    plt.plot(result_2["lambdas"][:result_2["iterations"]], result_2["v_trajectory"])
    plt.xlabel("Penalty weight")
    plt.ylabel("v")
    plt.xscale("log")
    plt.title("Penalty weight relaxation trajectory")
    plt.savefig("plot.png")
    plt.close()
    

    ######################
    various_plots(problem, result_2, result_1, endowments, survival_floor, args, ideal)


