import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import cvxpy as cp
import numpy as np
from src.utils import to_np
from src.probabilistic_model import ProbabilisticModel

class WeightsOptimizer:

    def __init__(self, data: dict, model: ProbabilisticModel, sample_adv):
        """
        Initialize the optimizer for given dataset and model

        Parameters
        ----------
        data : dict
            Dictionary format:
                - Keys : string, names of the observed variables
                - Values : torch.Tensor, observed samples, shape: (n_data_samples, variable_dim)

        model : ProbabilisticModel
            Instance of the ProbabilisticModel class representing the Defender's model.

        sample_adv : callable
            Function used to sample from the adversarial posterior.
            Signature: n_samples: int -> samples: dict with:
                        - Keys : string, names of the parameters
                        - Values : torch.Tensor, array of samples, shape: (n_param_samples, param_dim,)
        """
        
        self.model = model
        self.data = data
        self.n_data_samples = data[list(data.keys())[0]].shape[0]
        self.sample_adv = sample_adv


    def optimize(self, n_MC_samples_adv=1000,
                     max_iterations=100,
                     heuristic="FGSM", 
                     max_occurence=2, max_L1=20,
                     n_MC_samples_post=100,
                     warmup_steps=100,
                     warmup_steps_repeat=10,
                     make_plots=False,
                     eps=1e-3, lr=1.0,
                     betas=(0.9, 0.999),
                     stopping_ratio=1e-4,
                     rounding=True,
                     verbose=False,
                     L2_constraints=False,
                     decrease_lr=False):
        """
        Compute an approximate solution to the IPA problem.

        Parameters
        ----------
        heuristic: string, optional
            Name of the heuristic to use. Accepted values are "FGSM", "1O-ISCD", "2O-ISCD",
            "SGD-R2", "Adam-R2", "2O-R2", "2O-EMA-R2" and "2O-R2-m" (alias for "2O-EMA-R2").
            Default is "FGSM".

        max_iterations : int, optional
            Maximum number of iterations for R2 and ISCD heuristics.
            Default is 100.

        max_occurence : int, optional
            Constraint called "L", maximum number of repetition of a single datapoint in the tainted dataset.
            Default is 2.

        max_L1 : int, optional
            Constraint called "B", maximum L1 distance between the weight vector and the vector of ones of the same size.
            Default is 20.
        
        n_MC_samples_adv : int, optional
            Number of samples to compute Monte-Carlo approximation of the expectation 
            under the adversarial posterior.
            Default is 1000.

        n_MC_samples_post : int, optional
            Number of samples to compute Monte-Carlo approximations of the expectations 
            and covariances under the posterior.
            Default is 1000.

        warmup_steps : int, optional
            Number of warmup steps for the first MCMC sampling from the posterior. 
            Default is 100.

        warmup_steps_repeat : int, optional
            Number of warmup steps for MCMC sampling when initialized with previous samples. 
            Default is 10.

        eps: float, optional
            Regularization parameter epsilon for the 2O-R2 and 2O-EMA-R2 heuristics.
            Default is 1e-3.

        lr: float, optional
            Learning rate parameter for the SGD-R2 and Adam-R2 heuristics.
            Default is 1.0.

        decrease_lr: bool, optional
            Only used for SGD-R2, if True, the learning rate schedule is "gamma_t = gamma_0 / t", 
            where gamma_0 is the "lr" parameter. Otherwise, the learning rate is constant.
            Default is False.

        betas: tuple of float, optional
            Moving average parameters for Adam-R2 and 2O-EMA-R2.
            When using Adam-R2, correspond to the beta parameters of the Adam algorithm.
            When using 2O-EMA-R2, only the first parameter is used.
            Default is (0.9, 0.999).
        
        stopping_ratio: float, optional
            Ratio used in the stopping criterion of R2 heuristics.
            The algorithm stops whenever the ratio between the decrease of KL divergence
            estimated from Taylor expansion at the current iteration over the one estimated 
            during the first iteration is below this threshold.
            Default is 1e-4.

        make_plots: bool, optional
            When True, 2nd order Taylor expansions are computed at each iteration to estimate 
            the evolution of KL divergence and these estimations are plotted at the end. 
            Ignored when using FGSM.
            Default is False.
            
        rounding: bool, optional
            Allow to skip the rounding step in R2 heuristics if set to False, in this case a
            continuous weight vector is returned.
            Default is True.
        
        verbose: bool, optional
            Enables printing when stopping criterion is reached.
            Default is False.

        L2_constraints: bool, optional
            For R2 heuristics, if True and if max_occurences <= 2,
            replaces the L1 constraint for the continuous relaxation problem by an L2 constraint
            without changing the set of feasible integer weight vectors.
            Default is False.

        Returns
        -------
        weights : torch.Tensor, shape: (n_data_samples, )
            Vector associating a weight to each observed sample
        """

        self.initial_params = None
        adv_samples = self.sample_adv(n_MC_samples_adv)
        self.f_adv = self.model.log_likelihood_mean(self.data, adv_samples)
        if make_plots:
            # Estimate the covriance matrix of the f_adv estimate (covariance matrix of the empirical mean, not of f_A(theta))
            self.cov_f_adv_est = self.model.log_likelihood_covariance(self.data, adv_samples) / n_MC_samples_adv
        
        self.max_iterations = max_iterations
        self.max_occurence = max_occurence
        self.max_L1 = max_L1
        self.n_MC_samples_post = n_MC_samples_post
        self.warmup_steps = warmup_steps
        self.warmup_steps_repeat = warmup_steps_repeat
        self.make_plots = make_plots
        self.eps = eps
        self.lr = lr
        self.betas = betas
        self.stopping_ratio = stopping_ratio
        self.rounding = rounding
        self.verbose = verbose
        self.L2_constraints = False
        if L2_constraints and max_occurence <= 2:
            self.L2_constraints = True
        self.decrease_lr = decrease_lr


        if heuristic=="1O-ISCD":
            return self.ISCD(order=1)
        
        elif heuristic=="2O-ISCD":
            return self.ISCD(order=2)
        
        elif heuristic=="FGSM":
            return self.FGSM()
        
        elif heuristic == "SGD-R2":
            return self.rounded_relaxation(order=1, momentum=False)
        
        elif heuristic == "Adam-R2":
            return self.rounded_relaxation(order=1, momentum=True)
        
        elif heuristic == "2O-R2":
            return self.rounded_relaxation(order=2, momentum=False)
        
        elif heuristic == "2O-R2-m" or heuristic == "2O-EMA-R2":
            return self.rounded_relaxation(order=2, momentum=True)
        
        else:
            print("Heuristic", heuristic, "not implemented")       


    def FGSM(self):
        """
        Run the Fast-Gradient-Sign-Method heuristic.
        """
        weights = torch.ones(self.n_data_samples, dtype=int)
        grad = self.estimate_taylor_expansion(weights.float())
        
        # Maximum number of weights to modify
        n_points_modified = min(self.max_L1, self.n_data_samples)

        if self.max_occurence == 1:
            # Deletions only
            candidate_datapoints = torch.argsort(grad, descending=True)
            n_points_modified = min(n_points_modified, int(torch.sum(grad > 0)))
        
        else:
            # Deletions and duplications
            candidate_datapoints = torch.argsort(torch.abs(grad), descending=True)

        indices = candidate_datapoints[:n_points_modified]
        weights[indices] -= torch.sign(grad[indices]).int()

        return weights


    def ISCD(self, order=1):
        """
        Run the Integer-Steps Coordinate Descent heuristic.

        Parameters
        ----------
        order : int, optional
            Order of the Taylor expansion used in each iteration, either 1 or 2.
            Default is 1.
        """

        weights = torch.ones(self.n_data_samples, dtype=int)
        last_weights = weights.clone()
        if self.make_plots:
            # Lists of estimated relative KL evolutions
            # Convexity bounds using first order
            kl_diff_upper = [0.0]
            kl_diff_lower = [0.0]

            # Associated uncertainty
            kl_diff_upper_var = [0.0]
            kl_diff_lower_var = [0.0]

            # Second order estimates
            forward_O2_kl_diff = [0.0]
            backward_O2_kl_diff = [0.0]

        for i in range(self.max_iterations):
            self.iteration = i

            if order == 1 and not self.make_plots:
                grad = self.estimate_taylor_expansion(weights.float())
            
            else:
                grad, hessian_diag = self.estimate_taylor_expansion(weights.float(), 
                                                                    estimate_hessian=True, diagonal_hessian=True)

            # Update the decrease upper bound
            if self.make_plots and i >= 1:
                diff_est = - self.estimate_kl_diff(weights, last_weights, grad)
                kl_diff_upper.append(float(diff_est))
                backward_O2_kl_diff.append( -self.estimate_kl_diff(weights, last_weights, grad, hessian_diag))
                kl_diff_upper_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian_diag)))
    
            last_weights = weights.clone()

            # Estimate the best kl decrease for each potential coordinate update
            estimated_decrease = torch.abs(grad)
            if order == 2:
                estimated_decrease -= 0.5 * hessian_diag

            # Sort the candidate coordinates
            candidate_datapoints = torch.argsort(estimated_decrease, descending=True)

            # Iterate over the datapoints with the strongest gradient until we find one that can be
            # updated without violating the constraints
            for idx in candidate_datapoints:
                if estimated_decrease[idx] < 0:
                    # Updating this weight would increase the objective
                    continue

                if grad[idx] > 0 and weights[idx] == 0:
                    # Updating this weight would break the non-negativity constraint
                    continue

                if grad[idx] < 0 and weights[idx] >= self.max_occurence:
                    # Updating this weight would break the L-infty norm constraint
                    continue

                if torch.norm(weights - torch.ones(self.n_data_samples), p=1) >= self.max_L1:
                    # Updating this weight would break the L-1 norm constraint
                    if grad[idx] > 0 and weights[idx] >= 1:
                        continue

                    if grad[idx] < 0 and weights[idx] <= 1:
                        continue

                # Update the chosen weight
                weights.data[idx] -= int(torch.sign(grad[idx]))
                break
            
            # Estimate decrease of objective function
            diff_est = self.estimate_kl_diff(last_weights, weights, grad)

            # Update the lower bound
            if self.make_plots:
                kl_diff_lower.append(float(diff_est))
                kl_diff_lower_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian_diag)))
                forward_O2_kl_diff.append(self.estimate_kl_diff(last_weights, weights, grad, hessian_diag))


            if (last_weights == weights).all():
                # Reached the stopping criterion
                if self.verbose:
                    print("Stopping criterion reached after", i, "iterations")
                break


        if self.make_plots:

            # Update KL bounds
            grad, hessian_diag = self.estimate_taylor_expansion(weights.float(), estimate_hessian=True, diagonal_hessian=True)
            backward_O2_kl_diff.append( -self.estimate_kl_diff(weights, last_weights, grad, hessian_diag))
            kl_diff_upper_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian_diag)))

            diff_est = - self.estimate_kl_diff(weights, last_weights, grad)
            kl_diff_upper.append(float(diff_est))
            
            plt.figure(figsize=(8, 6))

            # Plot bounds 
            kl_diff_lower = np.array(kl_diff_lower)
            kl_diff_upper = np.array(kl_diff_upper)
            plt.plot(np.cumsum(kl_diff_lower), label="Lower bound", color="steelblue", alpha=1/order)
            plt.plot(np.cumsum(kl_diff_upper), label="Upper bound", color="orangered", alpha=1/order)

            # 95% confidence intervals on bounds
            kl_diff_lower_std = np.sqrt(np.cumsum(np.array(kl_diff_lower_var)))
            kl_diff_upper_std = np.sqrt(np.cumsum(np.array(kl_diff_upper_var)))
            plt.fill_between(np.arange(len(kl_diff_lower)), 
                                np.cumsum(kl_diff_lower) - 1.96 * kl_diff_lower_std, 
                                np.cumsum(kl_diff_lower) + 1.96 * kl_diff_lower_std, 
                                color="steelblue", alpha = 0.2)
            plt.fill_between(np.arange(len(kl_diff_lower)), 
                                np.cumsum(kl_diff_upper) - 1.96 * kl_diff_upper_std, 
                                np.cumsum(kl_diff_upper) + 1.96 * kl_diff_upper_std, color="orangered", alpha = 0.2)

            # Second order estimates
            forward_O2_kl_diff = np.array(forward_O2_kl_diff)
            backward_O2_kl_diff = np.array(backward_O2_kl_diff)

            plt.plot(np.cumsum(forward_O2_kl_diff), label="Forward 2O estimate", 
                        color="Blue")
            plt.plot(np.cumsum(backward_O2_kl_diff), label="Backward 2O estimate", 
                        color="Green")

            plt.legend(loc="upper right", fontsize=15)
            plt.grid(True)
            plt.xlabel("Iterations", fontsize=22)
            plt.ylabel("Relative KL divergence", fontsize=22)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()

        return weights


    def rounded_relaxation(self, order=1, momentum=False):
        """
        Run the Rounded Relaxation heuristic. The variation used depends on the parameters:

        - order=1, momentum=False: SGD-R2
        - order=1, momentum=True: Adam-R2
        - order=2, momentum=False: 2O-R2
        - order=2, momentum=True: 2O-EMA-R2

        Parameters
        ----------
        order : int, optional
            Order of the Taylor expansion used in each iteration, either 1 or 2.
            Default is 1.

        momentum : bool, optional
            Use exponential moving average of taylor expansions.
            Default is False.
        """

        weights = torch.ones(self.n_data_samples, dtype=torch.float)
        last_weights = weights.clone()
        if self.make_plots:
            # Lists of estimated relative KL evolutions
            # Convexity bounds using first order
            kl_diff_upper = [0.0]
            kl_diff_lower = [0.0]

            # Associated uncertainty
            kl_diff_upper_var = [0.0]
            kl_diff_lower_var = [0.0]

            # Second order estimates
            forward_O2_kl_diff = [0.0]
            backward_O2_kl_diff = [0.0]

        hessian = None
        first_kl_diff = 0.0

        for i in range(self.max_iterations):
            self.iteration = i

            if order == 1 and not self.make_plots:
                grad = self.estimate_taylor_expansion(weights.float())
            
            else:
                grad, hessian = self.estimate_taylor_expansion(weights.float(), 
                                                               estimate_hessian=True)

            # Update the decrease upper bound
            if self.make_plots and i >= 1:
                diff_est = - self.estimate_kl_diff(weights, last_weights, grad)
                kl_diff_upper.append(float(diff_est))
                backward_O2_kl_diff.append( -self.estimate_kl_diff(weights, last_weights, grad, hessian))
                kl_diff_upper_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian)))

    
            last_weights = weights.clone()

            # Update weights
            if order == 1:
                if momentum: # Adam-R2
                    self.step_adam(weights, grad)

                else: 
                    # SGD-R2
                    # Step size schedule
                    lr = self.lr / ((self.iteration + 1) if self.decrease_lr else 1)

                    # Projected SGD step
                    weights = self.project_continuous_constraint(weights - lr * grad)

            elif order == 2:
                if momentum:
                    # 2O-EMA-R2 step
                    self.step_2O_R2_m(weights, grad, hessian)

                else: 
                    # 2O-R2 step
                    weights = self.minimize_taylor(weights, grad, hessian)
                    
            # Estimate decrease of objective function
            diff_est = self.estimate_kl_diff(last_weights, weights, grad)

            # Update the lower bound
            if self.make_plots:
                kl_diff_lower.append(float(diff_est))
                kl_diff_lower_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian)))
                forward_O2_kl_diff.append(self.estimate_kl_diff(last_weights, weights, grad, hessian))

            if order == 2:
                diff_est = self.estimate_kl_diff(last_weights, weights, grad, hessian)

            if i == 0:
                first_kl_diff = diff_est

            # Stopping criterion
            if torch.abs(diff_est / first_kl_diff) < self.stopping_ratio:
                if self.verbose:
                    print("Stopping criterion reached after", i+1, "iterations")
                    print("first_kl_diff", to_np(first_kl_diff))
                    print("diff_est", to_np(diff_est))
                break
        
        # Update kl estimates
        if self.make_plots:
            grad, hessian = self.estimate_taylor_expansion(weights.float(), 
                                                        estimate_hessian=True)
            backward_O2_kl_diff.append( -self.estimate_kl_diff(weights, last_weights, grad, hessian))
            kl_diff_upper_var.append(float(self.estimate_kl_diff_var(weights, last_weights, hessian)))
            kl_diff_upper.append( -float(self.estimate_kl_diff(weights, last_weights, grad)))

        last_weights = weights.clone()
        if self.rounding:
            # Round the weights taking the constraints into account
            weights = self.constrained_rounding(weights)

        if self.make_plots:
            plt.figure(figsize=(8, 6))
            
            # Plot bounds 
            kl_diff_lower = np.array(kl_diff_lower)
            kl_diff_upper = np.array(kl_diff_upper)
            plt.plot(np.cumsum(kl_diff_lower), label="Lower bound", color="steelblue", alpha=1/order)
            plt.plot(np.cumsum(kl_diff_upper), label="Upper bound", color="orangered", alpha=1/order)

            rounding_kl_diff = self.estimate_kl_diff(last_weights, weights, grad, hessian)
            mean_final_kl_diff = (np.sum(kl_diff_lower) + np.sum(kl_diff_upper)) / 2
            print("Rounding kl difference:", to_np(rounding_kl_diff))
            print("L1 before rounding:", to_np(torch.norm(last_weights - torch.ones(self.n_data_samples), p=1)))
            print("L1 after rounding:", to_np(torch.norm(weights - torch.ones(self.n_data_samples), p=1)))

            # 95% confidence intervals on bounds
            kl_diff_lower_std = np.sqrt(np.cumsum(np.array(kl_diff_lower_var)))
            kl_diff_upper_std = np.sqrt(np.cumsum(np.array(kl_diff_upper_var)))
            plt.fill_between(np.arange(self.iteration + 2), 
                                np.cumsum(kl_diff_lower) - 1.96 * kl_diff_lower_std, 
                                np.cumsum(kl_diff_lower) + 1.96 * kl_diff_lower_std, 
                                color="steelblue", alpha = 0.2)
            plt.fill_between(np.arange(self.iteration + 2), 
                                np.cumsum(kl_diff_upper) - 1.96 * kl_diff_upper_std, 
                                np.cumsum(kl_diff_upper) + 1.96 * kl_diff_upper_std, color="orangered", alpha = 0.2)


            # Second order estimates
            forward_O2_kl_diff = np.array(forward_O2_kl_diff)
            backward_O2_kl_diff = np.array(backward_O2_kl_diff)

            mean_final_kl_diff = (np.sum(forward_O2_kl_diff) + np.sum(backward_O2_kl_diff)) / 2

            plt.plot(np.cumsum(forward_O2_kl_diff), label="Forward 2O estimate", 
                        color="Blue")
            plt.plot(np.cumsum(backward_O2_kl_diff), label="Backward 2O estimate", 
                        color="Green")


            plt.scatter([len(kl_diff_lower) - 1], [mean_final_kl_diff], 
                        marker="x", label="Mean before rounding", color="darkslategrey")
            plt.scatter([len(kl_diff_lower) - 1], [mean_final_kl_diff + rounding_kl_diff], 
                        marker="P", label="Mean after rounding", color="red")
            plt.legend(loc="upper right", fontsize=15)
            plt.grid(True)
            plt.xlabel("Iterations", fontsize=22)
            plt.ylabel("Relative KL divergence", fontsize=22)
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.show()

        return weights


    def step_adam(self, weights: torch.Tensor, grad: torch.Tensor):
        """
        Perform one step of the Adam-R2 heuristic.

        Parameters
        ----------
        weights : torch.Tensor, shape: (n_data_samples, )
            Current weight vector, will be modified inplace.

        grad : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation.
        """
        if self.iteration == 0:
            # Initialize optimizer
            self.adam = torch.optim.Adam([weights], lr=self.lr, betas=self.betas)

        self.adam.zero_grad()

        # Projected gradient trick
        with torch.no_grad():
            # Simple projected SGD 
            projected_point = self.project_continuous_constraint(weights - self.lr * grad)
            # Projected version of the gradient
            weights.grad = weights - projected_point

        self.adam.step()
        
        with torch.no_grad():
            # Inplace update of the weight vector
            weights.data = self.project_continuous_constraint(weights)

    
    def step_2O_R2_m(self, weights: torch.Tensor, grad: torch.Tensor, hessian: torch.Tensor):
        """
        Perform one step of the 2O-EMA-R2 heuristic.

        Parameters
        ----------
        weights : torch.Tensor, shape: (n_data_samples, )
            Current weight vector, will be modified inplace.

        grad : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation.

        hessian : torch.Tensor, shape: (n_data_samples, n_data_samples)
            Hessian estimation.
        """
                
        if self.iteration == 0:
            # Initialize moving average quantities
            self.Q = torch.zeros((self.n_data_samples, self.n_data_samples), dtype=torch.float)
            self.p = torch.zeros(self.n_data_samples, dtype=torch.float)
            self.last_weights = weights.clone()
        
        # Update the moving average quantities taking into
        # account the shift of reference point for Taylor expansion
        w_shift = weights - self.last_weights
        self.last_weights = weights.clone()
        self.p = (1 - self.betas[0])*grad + self.betas[0]*(self.Q @ w_shift + self.p)
        self.Q = (1 - self.betas[0])*hessian + self.betas[0]*self.Q

        # Solve the approximate intermediate problem, inplace update of the weights
        weights.data = self.minimize_taylor(weights, self.p, self.Q)
    
   
    def minimize_taylor(self, w0: torch.Tensor, g: torch.Tensor, H: torch.Tensor):
        """
        Solve the constrained quadratic program corresponding to an 
        approximation of the continuous relaxation problem, where
        the objective function is replaced by a second order Taylor expansion.
        This is solved using the cvxpy library.

        Parameters
        ----------
        w0 : torch.Tensor, shape: (n_data_samples, )
            Reference point for the Taylor expansion.

        g : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation.

        H : torch.Tensor, shape: (n_data_samples, n_data_samples)
            Hessian estimation.

        Returns
        -------
        weights : torch.Tensor, shape: (n_data_samples, )
            Approximate solution to the problem.
        """
        w_var = cp.Variable(self.n_data_samples)

        w0_np = to_np(w0)
        g_np = to_np(g)
        H_np = to_np(H) + np.eye(self.n_data_samples) * self.eps
        H_np = cp.psd_wrap(H_np) # Wrapper to avoid numerical issue in cvxpy when trying to find the eigenvalues
        
        # Constraints
        if self.L2_constraints:
            constraints = [
                w_var >= 0,
                w_var <= self.max_occurence,
                cp.norm2(w_var - np.ones(self.n_data_samples)) <= np.sqrt(self.max_L1)
            ]
        else:
            constraints = [
                w_var >= 0,
                w_var <= self.max_occurence,
                cp.norm1(w_var - np.ones(self.n_data_samples)) <= self.max_L1
            ]
        
        # Objective
        objective = cp.Minimize((1/2)*cp.quad_form(w_var - w0_np, H_np) + g_np.T @ (w_var - w0_np))

        problem = cp.Problem(objective, constraints)
        
        # Solve
        problem.solve(warm_start=False)
        
        return torch.tensor(w_var.value, dtype=torch.float)
    

    def project_continuous_constraint(self, w: torch.Tensor):
        """
        Project a weight vector on the feasible set of the continuous relaxation,
        using cvxpy to solve the euclidian projection problem.

        Parameters
        ----------
        w : torch.Tensor, shape: (n_data_samples, )
            Weight vector to project.
            
        Returns
        -------
        weights : torch.Tensor, shape: (n_data_samples, )
            Projected weight vector.
        """
        w_proj = cp.Variable(self.n_data_samples)
        
        # Constraints
        if self.L2_constraints:
            constraints = [
                w_proj >= 0,
                w_proj <= self.max_occurence,
                cp.norm2(w_proj - np.ones(self.n_data_samples)) <= np.sqrt(self.max_L1)
            ]
        else:
            constraints = [
                w_proj >= 0,
                w_proj <= self.max_occurence,
                cp.norm1(w_proj - np.ones(self.n_data_samples)) <= self.max_L1
            ]

        # Objective
        objective = cp.Minimize(cp.norm2(w_proj - w.data.numpy()))
        
        problem = cp.Problem(objective, constraints)
        
        # Solve
        problem.solve()
        
        return torch.tensor(w_proj.value, dtype=torch.float)


    def estimate_taylor_expansion(self, weights: torch.Tensor, estimate_hessian=False, diagonal_hessian=False):
        """
        Estimate parameters of Taylor expansion from posterior samples.

        Parameters
        ----------
        weights : torch.Tensor, shape: (n_data_samples, )
            Current weight vector.

        estimate_hessian : bool, optional
            If True, estimate second order Taylor expansion.
            Default is False.
            
        diagonal_hessian : bool, optional
            If True, only the diagonal coefficients of the Hessian matrix are estimated, 
            the other ones are set to 0.
            
        Returns
        -------
        grad : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation.

        hessian : torch.Tensor, shape: (n_data_samples, n_data_samples)
            Hessian estimation.
        """

        warmup_steps = self.warmup_steps 
        if self.initial_params is not None:
            # Reduce the number of warmup steps if initial parameters are given
            warmup_steps = self.warmup_steps_repeat
        
        # Sample from the posterior
        posterior_samples = self.model.sample_posterior(self.n_MC_samples_post, self.data, 
                                                weights=weights, warmup_steps=warmup_steps,
                                                initial_params=self.initial_params)
        
        # Estimate the mean log likelihood vector under the posterior distribution with weights
        f_post = self.model.log_likelihood_mean(self.data, posterior_samples)

        # In case of numerical issues during sampling (some samples are inf)
        while torch.isinf(f_post).any():
            # Re-sample with a new warmup, ignoring the initial parameters
            print("inf params at iteration", self.iteration)
            posterior_samples = self.model.sample_posterior(self.n_MC_samples_post, self.data, 
                                                    weights=weights, warmup_steps=self.warmup_steps,
                                                    initial_params=None)
            f_post = self.model.log_likelihood_mean(self.data, posterior_samples)
        
        # Define the initial parameters for the next iteration
        self.initial_params = {k: v[-1].reshape(-1) for k, v in posterior_samples.items()}

        # Estimate the gradient
        grad = f_post - self.f_adv

        if not estimate_hessian:
            return grad
        
        if diagonal_hessian:
            ll_mat = self.model.log_likelihood_matrix(self.data, posterior_samples)
            centered_ll_mat = ll_mat - f_post # Shape: (n_param_samples, n_data_samples)
            ll_cov_diag = torch.mean(centered_ll_mat**2, dim=0)
            return grad, ll_cov_diag

        # Estimate the Hessian
        hessian = self.model.log_likelihood_covariance(self.data, posterior_samples)
        return grad, hessian


    def estimate_kl_diff(self, w_ref: torch.Tensor, w_test: torch.Tensor, grad: torch.Tensor, hessian=None):
        """
        Estimate difference in objective function at two weight vetors using Taylor expansion.

        Parameters
        ----------
        w_ref : torch.Tensor, shape: (n_data_samples, )
            Reference weight vector.

        w_test : torch.Tensor, shape: (n_data_samples, )
            Test weight vector.
            
        grad : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation at w_ref.

        hessian : torch.Tensor, shape: (n_data_samples, n_data_samples) or (n_data_samples, ), optional
            Hessian estimation at w_ref. If a tensor of shape (n_data_samples, ) is given, it will be 
            interpreted as the diagonal of the matrix.
            Default is None.
            
        Returns
        -------
        kl_diff : torch.Tensor, shape: (, )
            Estimation of the difference between the objective function at w_test and w_ref.
        """
        weights_diff = (w_test - w_ref).float()

        kl_diff = torch.dot(grad, weights_diff) # First order contribution

        if hessian is not None:
             # Second order contribution
            if len(hessian.shape) == 1:
                # If only the diagonal was estimated
                hessian = torch.diag(hessian)
            kl_diff += 0.5 * torch.dot(weights_diff, hessian @ weights_diff)

        return kl_diff

        
    def estimate_kl_diff_var(self, w_ref: torch.Tensor, w_test: torch.Tensor, hessian: torch.Tensor):
        """
        Estimate the variance of the estimation of difference in objective function
        when first order Taylor expansion was used.

        Parameters
        ----------
        w_ref : torch.Tensor, shape: (n_data_samples, )
            Reference weight vector.

        w_test : torch.Tensor, shape: (n_data_samples, )
            Test weight vector.
            
        grad : torch.Tensor, shape: (n_data_samples, )
            Gradient estimation at w_ref.

        hessian : torch.Tensor, shape: (n_data_samples, n_data_samples) or (n_data_samples, )
            Hessian estimation at w_ref. If a tensor of shape (n_data_samples, ) is given, it will be 
            interpreted as the diagonal of the matrix.
            
        Returns
        -------
        kl_diff_var : torch.Tensor, shape: (, )
            Estimation of the difference between the objective function at w_test and w_ref.
        """
        weights_diff = (w_test - w_ref).float()
        if len(hessian.shape) == 1:
            # If only the diagonal was estimated
            hessian = torch.diag(hessian)

        cov_grad = hessian / self.n_MC_samples_post + self.cov_f_adv_est # Covariance matrix of gradient estimate
        kl_diff_var = torch.dot(weights_diff, cov_grad @ weights_diff)
        return kl_diff_var
        

    def constrained_rounding(self, weights: torch.Tensor):
        """
        Project a continuous weight vector on the closest feasible weight vector 
        with integer coordinates.

        Parameters
        ----------
        w : torch.Tensor, shape: (n_data_samples, )
            Weight vector to project. 
            Must verify:
                0 <= w_i <= max_occurences, for all i, 
                ||v - 1||_1 <= max_L1, where v is obtained by rounding 
                                        each coordinate of w towards 1.
            
        Returns
        -------
        weights : torch.Tensor, shape: (n_data_samples, )
            Projected weight vector.
        """
        ones = torch.ones(self.n_data_samples)

        delta = torch.abs(weights - ones)
        floor = torch.floor(delta).int()

        # Compute the budget on the ||alpha||_1
        N_max = self.max_L1 - torch.sum(floor)

        eps = delta - floor
        alpha = (eps > 0.5).int()

        # Respect the budget
        indices = torch.argsort(eps*alpha, descending=True)
        alpha[indices[N_max:]] = 0

        return ones + torch.sign(weights - ones) * (floor + alpha)

