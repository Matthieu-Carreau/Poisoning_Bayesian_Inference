import numpy as np
import torch
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import MCMC, NUTS
from src.probabilistic_model import ProbabilisticModel
from numpyro.infer.util import log_likelihood
from numpyro.infer.autoguide import AutoLaplaceApproximation

from src.utils import remove_zeros, to_np

class LogisticRegression(ProbabilisticModel):
    """
    Linear Logistic Regression module implemented with numpyro, with a multivariate gaussian prior.
    Specifically,
        β ~ N(mu_0, Sigma_0)        (vector of regression coefficient)

    Data likelihood:
        y_i | β, X_i ~ B(σ(β^T X_i)), σ: logistic function
    
    Implemented with numpyro.

    Parameters
    ----------
    mu_0: mean of the prior over β.
    Sigma_0: covariance of the prior over β.
    
    Observed data dictionary format:
        {
        "X": torch.Tensor, shape: (n_samples, dim), array of features.
        "y": torch.Tensor, shape: (n_samples, ), vector of output binary values.
        }
        
    Model parameters dictionary format:
        {
        "beta": torch.Tensor, shape: (n_samples, dim), Columns are 
            vectors of regression coefficients.
        }
    """
    
    def __init__(self, prior_params, seed=0) -> None:
        """
        Initialize the model with prior parameters.

        Parameters
        ----------
        prior_params : dict
            Dictionary format:
                {
                "mu": torch.Tensor, shape: (dim), Prior mean.
                "Sigma": torch.Tensor, shape: (dim, dim), Prior covariance matrix.
                }

        seed : int, optional
            Initial random seed used for random generators.
            This seed will be automatically incremented after each random sampling.
            Default is 0.
        """
        self.mu_0 = jnp.array(to_np(prior_params["mu"]))
        self.Sigma_0 = jnp.array(to_np(prior_params["Sigma"]))

        self.prior_dist = dist.MultivariateNormal(self.mu_0, self.Sigma_0)
        
        def model(X, y, weights):
            """
            Numpyro probabilistic model
            """
            # Prior
            betas = numpyro.sample("beta", self.prior_dist)
            
            # Observed data
            with numpyro.plate('data', weights.shape[0]):
                with numpyro.handlers.scale(scale=weights):      
                    logits = jnp.dot(X, betas)
                    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)

        self.model = model
        self.seed = seed

        # Initial parameters for SVI with warm start
        self.init_params_svi = None



    def sample_posterior(self, n_param_samples: int, data: dict, weights=None,
                         initial_params=None, warmup_steps=1):
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
                "X": torch.Tensor, shape: (n_samples, dim), array of features.
                "y": torch.Tensor, shape: (n_samples, ), vector of output binaryvalues.
                }

        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        initial_params : dict, optional
            In the case of MCMC sampling, specifies the initial sample to be used.
                {
                "beta": torch.Tensor, shape: (dim, ), Vector of regression coefficients.
                }
            Default is None.

        warmup_steps : int, optional
            Number of warmup steps for MCMC sampling. Default is 1.

        Returns
        -------
        samples: dict
            Samples from the posterior distribution
            Dictionary format:
                {
                "beta": torch.Tensor, shape: (n_samples, dim), Columns are 
                    vectors of regression coefficients.
                }
        """
        X = to_np(data["X"])
        y = to_np(data["y"])

        if weights is None:
            weights = torch.ones(y.shape)

        if initial_params is not None:
            initial_params = {k: to_np(v) for k, v in initial_params.items()}
    
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=n_param_samples, num_warmup=warmup_steps, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(self.get_seed()), X, y, to_np(remove_zeros(weights)),
                 init_params=initial_params)
        return {k: torch.tensor(np.array(v)) for k, v in mcmc.get_samples().items()} #mcmc.get_samples()
    
    

    def log_likelihood_matrix(self, data: dict, params: dict) -> torch.Tensor:
        """
        Compute the log-likelihood matrix of the data samples given model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_samples, dim), array of features.
                "y": torch.Tensor, shape: (n_samples, ), vector of output binaryvalues.
                }

        params : dict, 
            Contains n_param_samples sets of model parameters.
            Dictionary format:
                {
                "beta": torch.Tensor, shape: (n_samples, dim), Columns are 
                    vectors of regression coefficients.
                }

        Returns
        -------
        ll_matrix : torch.Tensor, shape: (n_param_samples, n_data_samples)
            Log-likelihood matrix, each coefficient [i,j] is the log-likelihood of the 
            i-th data sample under the j-th set of model parameters
        """
        X = to_np(data["X"])
        y = to_np(data["y"])

        weights = np.ones(y.shape)
        params_np = {k: to_np(v) for k, v in params.items()}
        jax_ll_array = log_likelihood(self.model, params_np, X, y, weights)["obs"]
        return torch.tensor(np.array(jax_ll_array), dtype=torch.float)
    

    def normal_posterior_approximation(self, data: dict, weights=None, method="Laplace", optim_steps=500, lr=1e-2, n_samples_MCMC=200, initial_loc=None, verbose=False):
        """
        Compute the parameters for a multivariate gaussian that approximates the posterior distribution

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_samples, dim), array of features.
                "y": torch.Tensor, shape: (n_samples, ), vector of output binaryvalues.
                }

        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        method: string, only "Laplace" is implemented
            Method used to obtain a normal approximation of the posterior.
            - "Laplace": Laplace approximation.

        Returns
        -------
        mean : torch.Tensor, shape: (dim, )
            Mean vector for the normal posterior approximation.

        covariance : torch.Tensor, shape: (dim, dim)
            Covariance matrix for the normal posterior approximation.
        """
        X = to_np(data["X"])
        y = to_np(data["y"])
    
        if weights is None:
            weights = torch.ones(y.shape)

        weights = to_np(remove_zeros(weights))

        if method=="Laplace":
            guide_laplace = AutoLaplaceApproximation(self.model)
            optimizer = optim.Adam(step_size=lr)
            svi_laplace = SVI(self.model, guide_laplace, optimizer, loss=Trace_ELBO(), X=X, y=y, weights=weights)
            svi_laplace_result = svi_laplace.run(jax.random.PRNGKey(0), optim_steps, progress_bar=verbose)
            posterior = guide_laplace.get_posterior(svi_laplace.get_params(svi_laplace_result.state))

            mean_vi = torch.tensor(np.array(posterior.loc))
            covariance_vi = torch.tensor(np.array(posterior.covariance_matrix))

            return mean_vi, covariance_vi
        else:
            print("Method \"{}\" not implemented".format(method))


    def get_seed(self):
        """Return and increment the model's seed"""
        self.seed += 1
        return self.seed

