import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
import matplotlib.pyplot as plt

from src.probabilistic_model import ProbabilisticModel


class NIGLinearRegression(ProbabilisticModel):
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown coefficients. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(mu, σ²v)

    The implementation is made with torch.
        
    Observed data dictionary format:
        {
        "X": torch.Tensor, shape: (n_data_samples, dim + 1), design matrix.
        "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
        }
        
    Model parameters dictionary format:
        {
        "beta": torch.Tensor, shape: (n_param_samples, dim + 1), Columns are 
            vectors of intercept and regression coefficients,
        "sigma2": torch.Tensor, shape: (n_param_samples, ), Noise variances.
        }
    """
    
    def __init__(self, mu: torch.Tensor, v: torch.Tensor, a: float, b:float):
        """
        Initialize the model with normal inverse-gamma prior.

        Parameters
        ----------
        mu : torch.Tensor, shape: (n_data_samples, )
            Mean of the prior distribution for β

        v : torch.Tensor, shape: (n_data_samples, n_data_samples)
            Covariance matrix used to define the prior distribution for β

        a : float
            Shape parameter of the Gamma prior for 1/σ²

        b : float
            Rate parameter of the Gamma prior for 1/σ²
        """
        self.mu = mu
        self.v = v
        self.a = a
        self.b = b


    def sample_posterior(self, n_param_samples: int, data: dict, weights=None,
                         initial_params=None, warmup_steps=1) -> dict:
        """
        Sample the posterior distribution given observed data.

        Parameters
        ----------
        n_param_samples : int
            Number of samples to generate.

        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, dim + 1), design matrix.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }
        
        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        initial_params : not used, Default is None.

        warmup_steps : not used, Default is 1.

        Returns
        -------
        samples: dict
            Samples from the posterior distribution
            Dictionary format:
                - Keys : string, names of the parameter
                - Values : torch.Tensor, array of posterior samples, shape: (n_param_samples, param_dim,)
        """
    
        mu_ast, v_ast, a_ast, b_ast = self.posterior_parameters(data, weights=weights)

        # Sample the posterior
        inv_sigma2 = Gamma(a_ast, b_ast)
        sigma2_sample = 1.0 / ( inv_sigma2.sample(torch.tensor([n_param_samples])) )
        centered_beta_dist = MultivariateNormal(torch.zeros_like(mu_ast), v_ast)
        beta_sample = centered_beta_dist.sample((n_param_samples, ))

        beta_sample = mu_ast + (torch.sqrt(sigma2_sample) * beta_sample.T).T

        return {'beta': beta_sample, 'sigma2': sigma2_sample}


    def log_likelihood_matrix(self, data: dict, params: dict) -> torch.Tensor:
        """
        Compute the log-likelihood matrix of the data samples given model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, dim + 1), design matrix.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }
        
        params : dict, 
            Contains n_param_samples sets of model parameters, the j-th set is represented by the
            set of all j-th rows of the tensors in the dictionary
            Dictionary format:
                {
                "beta": torch.Tensor, shape: (n_param_samples, dim + 1), Columns are 
                    vectors of intercept and regression coefficients,
                "sigma2": torch.Tensor, shape: (n_param_samples, ), Noise variances.
                }            

        Returns
        -------
        ll_matrix : torch.Tensor, shape: (n_param_samples, n_data_samples)
            Log-likelihood matrix, each coefficient [i,j] is the log-likelihood of the 
            i-th data sample under the j-th set of model parameters
        """
        X = data["X"]
        y = data["y"]

        betas, sig2s = params["beta"], params["sigma2"]

        residuals = betas @ X.T - y # shape: (n_betas, n_datapoints)
        squared_residuals_T = residuals.T ** 2 # shape: (n_datapoints, n_betas)
        log_Z =  -0.5 * torch.log(sig2s * 2 * torch.pi)
        ll_matrix = log_Z - 0.5 * squared_residuals_T / sig2s
        
        return ll_matrix.T # shape: (n_betas, n_datapoints)
    

    def posterior_parameters(self, data, weights=None):
        """
        Compute the parameters of the posterior NIG distribution, given observed data
        
        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, dim + 1), design matrix.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }
        
        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        Returns
        -------
        mu_ast: torch.Tensor, shape: (dim + 1)
            Posterior for N(mu, σ²v) on the model β

        v_ast:  torch.Tensor, shape: (dim + 1, dim + 1)
            Posterior for N(mu, σ²v) on the model β

        a_ast:  torch.Tensor, shape: (, )
            Posterior for Γ(a, b) on the inverse sigma2 of the distribution

        b_ast:  torch.Tensor, shape: (, )
            Posterior for Γ(a, b) on the inverse sigma2 of the distribution
        """

        X = data["X"]
        y = data["y"]
        
        m, _ = X.shape

        v_inv = torch.linalg.inv(self.v)

        if weights is None:
            weights = torch.ones(m)

        X_t = (X.clone().mT*weights)

        Lamb_ast = v_inv + X_t @ X
        v_ast = torch.linalg.inv(Lamb_ast)
        v_ast = (v_ast + v_ast.mT) / 2 # Make sure that v is symmetric (correct numerical errors due to bad condition)

        mu_ast = v_ast @ (v_inv @ self.mu + X_t @ y)

        a_ast = self.a + 0.5 * torch.sum(weights)

        b_ast = self.b + 0.5 * (torch.sum(weights * y**2) + torch.dot(self.mu, torch.linalg.inv(self.v) @ self.mu) - torch.dot(mu_ast, Lamb_ast @ mu_ast))

        return mu_ast, v_ast, a_ast, b_ast
    

    def plot_posterior_betas(self, n_param_samples: int, data: dict, weights=None, title=None, covariate_names=None):
        """Plot histograms of the posterior marginal distributions given observed data.

        Parameters
        ----------
        n_param_samples : int
            Number of samples to generate the histograms.

        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, dim + 1), design matrix.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }
        
        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        title : string, optional
            Title of the plot, Default is None.

        covariate_names : list of string, optional
            Name of the covariates to display, Default is None.
        """

        beta_sample = self.sample_posterior(n_param_samples, data, weights=weights)["beta"]

        beta_size = beta_sample.shape[1]
        n_cols = 5
        n_rows = 1 + (beta_size-1) // n_cols 

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*2, n_rows*3)) 
        axs = axs.ravel() 
        cmap = plt.colormaps["Pastel2"]  
        c = cmap.colors
        for i in range(beta_sample.shape[1]):

            # Plot histogram for the first sample
            axs[i].hist(beta_sample[:,i], bins=30, alpha=0.7, color=c[i//len(c)], edgecolor='black')
            if covariate_names is not None:
                if i == 0:
                    axs[i].set_title(f'Intercept')
                else:
                    axs[i].set_title(covariate_names[i-1])
            else:
                axs[i].set_title(f'Beta {i} Coefficient')
            axs[i].set_xlabel('Value')
            axs[i].set_ylabel('Target variable')

        if title is not None:
            plt.suptitle(title)
            
        # Adjust layout for better spacing
        plt.tight_layout()

        # Show plot
        plt.show()
