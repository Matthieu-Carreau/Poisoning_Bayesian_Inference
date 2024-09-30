from abc import ABC, abstractmethod
from torch import Tensor


class ProbabilisticModel(ABC):
    """
    Abstract class for the probabilistic models
    """
    
    @abstractmethod
    def __init__(self, prior_params: dict, seed=0) -> None:
        """
        Initialize the probabilistic models with prior parameters.

        Parameters
        ----------
        prior_params : dict
            Dictionary format:
                - Keys : string, names of the parameters
                - Values : torch.Tensor, prior values for the parameters, shape: (param_dim,)

        seed : int, optional
            Initial random seed used for random generators if numpyro is used.
            This seed will be automatically incremented after each random sampling.
            Default is 0.
        """
        pass
    

    @abstractmethod
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
                - Keys : string, names of the observed variables
                - Values : torch.Tensor, observed samples, shape: (n_data_samples, variable_dim)

        weights : torch.Tensor, shape: (n_data_samples, ), optional
            Vector associating a weight to each observed sample, Default is None.

        initial_params : dict, optional
            In the case of MCMC sampling, specifies the initial sample to be used.
            Dictionary format:
                - Keys : string, names of the parameter
                - Values : torch.Tensor, initial samples for the parameter, shape: (param_dim,)
            Default is None.

        warmup_steps : int, optional
            Number of warmup steps for MCMC sampling. Default is 1.

        Returns
        -------
        samples: dict
            Samples from the posterior distribution
            Dictionary format:
                - Keys : string, names of the parameter
                - Values : torch.Tensor, array of posterior samples, shape: (n_param_samples, param_dim,)
        """
        pass


    @abstractmethod
    def log_likelihood_matrix(self, data: dict, params: dict) -> Tensor:
        """
        Compute the log-likelihood matrix of the data samples given model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples
            Dictionary format:
                - Keys : string, names of the observed variables
                - Values : torch.Tensor, observed samples, shape: (n_data_samples, variable_dim)

        params : dict, 
            Contains n_param_samples sets of model parameters, the j-th set is represented by the
            set of all j-th rows of the tensors in the dictionary
            Dictionary format:
                - Keys : string, names of the parameter
                - Values : torch.Tensor, values of model parameters, shape: (n_param_samples, param_dim)

        Returns
        -------
        ll_matrix : torch.Tensor, shape: (n_param_samples, n_data_samples)
            Log-likelihood matrix, each coefficient [i,j] is the log-likelihood of the 
            i-th data sample under the j-th set of model parameters
        """
        pass
    

    def log_likelihood_mean(self, data: dict, params: dict) -> Tensor:
        """
        Compute the mean log-likelihood for each data sample given model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples.
            Dictionary format:
                - Keys : string, names of the observed variables
                - Values : torch.Tensor, observed samples, shape: (n_data_samples, variable_dim)

        params : dict
            Model parameters.
            Dictionary format:
                - Keys : string, names of the parameters
                - Values : torch.Tensor, parameter values, shape: (n_param_samples, param_dim)

        Returns
        -------
        ll_mean : torch.Tensor, shape: (n_data_samples,)
            Mean log-likelihood for each data sample.
        """
        ll_mat = self.log_likelihood_matrix(data, params)
        ll_mean = ll_mat.mean(0)
        
        return ll_mean


    def log_likelihood_covariance(self, data: dict, params: dict) -> Tensor:
        """
        Compute the empirical covariance of the log-likelihood vector for the 
        data samples given a set of model parameters.

        Parameters
        ----------
        data : dict
            Observed data samples.
            Dictionary format:
                - Keys : string, names of the observed variables
                - Values : torch.Tensor, observed samples, shape: (n_data_samples, variable_dim)

        params : dict
            Model parameters.
            Dictionary format:
                - Keys : string, names of the parameters
                - Values : torch.Tensor, parameter values, shape: (n_param_samples, param_dim)

        Returns
        -------
        ll_cov : torch.Tensor, shape: (n_data_samples, n_data_samples)
            Covariance matrix of the log-likelihood vector.
        """
        ll_mat = self.log_likelihood_matrix(data, params)
        ll_mean = ll_mat.mean(0)

        centered_ll_mat = ll_mat - ll_mean
        ll_cov = centered_ll_mat.T @ centered_ll_mat / (ll_mat.shape[0] - 1)

        return ll_cov