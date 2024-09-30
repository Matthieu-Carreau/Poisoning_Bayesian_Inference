import torch
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.optim as optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import MCMC, NUTS
from src.probabilistic_model import ProbabilisticModel
from numpyro.infer.util import log_likelihood
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoDiagonalNormal


from src.utils import remove_zeros, to_np



class StudentTLinReg(ProbabilisticModel):
    """
    Linear regression with a single predictor and a StudentT prior on the 
    coefficients. Used to replicate the mexico microcredit study of 
    Angelucci et al. (2015). Specifically,
        β_0, β_1, log(σ) ∼ t(3, 0, 1000)

    Data likelihood:
        y_i | β, X_i, σ ~ N(β_0 + β_1 X_i, σ²), 1 <= i <= n

    The implementation is made with numpyro.
    
    Observed data dictionary format:
        {
        "X": torch.Tensor, shape: (n_data_samples, ), predictor vector.
        "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
        }
        
    Model parameters dictionary format:
        {
        "beta_0": torch.Tensor, shape: (n_param_samples, ), intercept samples.
        "beta_1": torch.Tensor, shape: (n_param_samples, ), slope samples.
        "sigma": torch.Tensor, shape: (n_param_samples, ), Noise level samples.
        }
    """
    
    def __init__(self, seed=0):
        """
        Initialize the model.

        Parameters
        ----------
        seed : int, optional
            Initial random seed used for random generators.
            This seed will be automatically incremented after each random sampling.
            Default is 0.
        """

        # Define the numpyro model
        def model(X, y, weights):
            # Priors for the parameters
            beta_0 = numpyro.sample("beta_0", dist.StudentT(df=3.0, loc=0.0, scale=1000.0))  # Baseline profit
            beta_1 = numpyro.sample("beta_1", dist.StudentT(df=3.0, loc=0.0, scale=1000.0))  # Treatment effect
            sigma = numpyro.sample("sigma", dist.TransformedDistribution(
                dist.StudentT(df=3.0, loc=0.0, scale=1000.0), dist.transforms.ExpTransform()))  # Noise scale (positive)

            # Linear model 
            mean_profit = beta_1 * X + beta_0
            
            # Likelihood for observed data
            with numpyro.plate("data", X.shape[0]):
                    with numpyro.handlers.scale(scale=weights):
                        numpyro.sample("obs", dist.Normal(mean_profit, sigma), obs=y)
        
        self.model = model

        self.init_params_svi = None

        self.seed = seed


    def sample_posterior(self, n_param_samples: int, data: dict, weights=None,
                         initial_params=None, warmup_steps=100):        
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
                "X": torch.Tensor, shape: (n_data_samples, ), predictor vector.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }

        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        initial_params : dict, optional
            In the case of MCMC sampling, specifies the initial sample to be used.
            Dictionary format:
                {
                "beta_0": torch.Tensor, shape: (n_param_samples, ), intercept samples.
                "beta_1": torch.Tensor, shape: (n_param_samples, ), slope samples.
                "sigma": torch.Tensor, shape: (n_param_samples, ), Noise level samples.
                }
            Default is None.

        warmup_steps : int, optional
            Number of warmup steps for MCMC sampling. 
            Default is 100.

        Returns
        -------
        samples: dict
            Samples from the posterior distribution
            Dictionary format:
                {
                "beta_0": torch.Tensor, shape: (n_param_samples, ), intercept samples.
                "beta_1": torch.Tensor, shape: (n_param_samples, ), slope samples.
                "sigma": torch.Tensor, shape: (n_param_samples, ), Noise level samples.
                }
        """
        X = data["X"].data.numpy()
        y = data["y"].data.numpy()

        if weights is None:
            weights = torch.ones(y.shape)

        if initial_params is not None:
            initial_params = {k: v.data.numpy() for k, v in initial_params.items()}
    
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=n_param_samples, num_warmup=warmup_steps, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(self.get_seed()), X, y, remove_zeros(weights).data.numpy(),
                 init_params=initial_params)
        return {k: torch.tensor(np.array(v)) for k, v in mcmc.get_samples().items()} #mcmc.get_samples()
    
    
    def log_likelihood_matrix(self, data: dict, params: dict):     
        """
        Compute the log-likelihood matrix of the data samples given model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, ), predictor vector.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }
        
        params : dict, 
            Contains n_param_samples sets of model parameters, the j-th set is represented by the
            set of all j-th rows of the tensors in the dictionary
            Dictionary format:
                {
                "beta_0": torch.Tensor, shape: (n_param_samples, ), intercept samples.
                "beta_1": torch.Tensor, shape: (n_param_samples, ), slope samples.
                "sigma": torch.Tensor, shape: (n_param_samples, ), Noise level samples.
                }

        Returns
        -------
        ll_matrix : torch.Tensor, shape: (n_param_samples, n_data_samples)
            Log-likelihood matrix, each coefficient [i,j] is the log-likelihood of the 
            i-th data sample under the j-th set of model parameters
        """


        X = data["X"].data.numpy()
        y = data["y"].data.numpy()

        weights = np.ones(y.shape)
        params_np = {k: v.data.numpy() for k, v in params.items()}
        jax_ll_array = log_likelihood(self.model, params_np, X, y, weights)["obs"]
        return torch.tensor(np.array(jax_ll_array), dtype=torch.float)
    

    def normal_posterior_approximation(self, data: dict, weights=None, method="MCMC", 
                                       optim_steps=500, lr=1e-2, 
                                       n_samples_MCMC=200, warmup_steps=100, 
                                       warm_start_svi=False, diagonal=False,
                                       verbose=False):
        """
        Compute the parameters for a multivariate gaussian that approximates the posterior distribution

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                {
                "X": torch.Tensor, shape: (n_data_samples, ), predictor vector.
                "y": torch.Tensor, shape: (n_data_samples, ), vector of output values.
                }

        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        method: string, "MCMC" or "VI"
            Method used to obtain a normal approximation of the posterior.
            - "MCMC": Computes the empirical mean and covariance matrix of posterior
            samples, used as parameters of a multivariate gaussian.
            - "VI": Variational Inference method.

        Returns
        -------
        mean : torch.Tensor, shape: (dim, )
            Mean vector for the normal posterior approximation.

        covariance : torch.Tensor, shape: (dim, dim)
            Covariance matrix for the normal posterior approximation.
        """

        if method == "MCMC":
            posterior_samples = self.sample_posterior(n_samples_MCMC, data, weights=weights,
                                                        warmup_steps=warmup_steps)
            shapes = [0]
            keys = []

            for key in posterior_samples.keys():
                keys.append(key)
                shape = posterior_samples[key].shape
                if len(shape) == 1:
                    shapes.append(1)
                elif len(shape) == 2:
                    shapes.append(shape[-1])
                else:
                    print("Unexpected shape for", key, shape)

            shapes = torch.tensor(shapes)
            idx = torch.cumsum(shapes, 0)
            samples_tensor = torch.zeros((n_samples_MCMC, shapes.sum()))

            for i, key in enumerate(keys):
                shape = posterior_samples[key].shape
                if len(shape) == 1:
                    samples_tensor[:, idx[i]] = posterior_samples[key]
                elif len(shape) == 2:
                    samples_tensor[:, idx[i]: idx[i+1]] = posterior_samples[key]

            mean_mcmc = samples_tensor.mean(dim=0)
            covariance_mcmc = (samples_tensor - mean_mcmc).T @ (samples_tensor - mean_mcmc) / n_samples_MCMC

            return mean_mcmc, covariance_mcmc
        
        if method == "VI":
            X = to_np(data["X"])
            y = to_np(data["y"])
            
            if weights is None:
                weights = torch.ones(y.shape)

            weights = to_np(remove_zeros(weights))

            guide = AutoDiagonalNormal(self.model) if diagonal else AutoMultivariateNormal(self.model)
            optimizer = optim.Adam(step_size=lr)
            svi = SVI(self.model, guide, optimizer, loss=Trace_ELBO(), X=X, y=y, weights=weights)
                
                
            init_params = None
            if warm_start_svi:
                init_params = self.init_params_svi
            svi_result = svi.run(jax.random.PRNGKey(self.get_seed()), optim_steps, init_params=init_params)
            if warm_start_svi:
                self.init_params_svi = svi.get_params(svi_result.state)

            if verbose:
                plt.plot(svi_result.losses)
                plt.ylabel("ELBO")
                plt.xlabel("Iterations")
                plt.show()

            posterior = guide.get_posterior(svi.get_params(svi_result.state))
            mean_vi = torch.tensor(np.array(posterior.loc))
            if diagonal:
                covariance_vi = torch.tensor(np.diag(np.array(posterior.scale)**2))
            else:
                covariance_vi = torch.tensor(np.array(posterior.covariance_matrix))

            return mean_vi, covariance_vi
        
        else:
            print("Method \"{}\" not implemented".format(method))

    def get_seed(self):
        self.seed += 1
        return self.seed