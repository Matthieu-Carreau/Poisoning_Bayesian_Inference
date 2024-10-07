# Code for Poisoning Bayesian Inference via Data Deletion and Replication

This repository contains code for reproducing all the experiments in the paper "Poisoning Bayesian Inference via Data Deletion and Replication"

## Installation and dependencies

Install all required dependencies using

```bash
conda env create -f poisoning_bayesian_inference.yml
```

Finally, install package using

```bash
pip install -e .
```

## Replication of the simulation study for linear regression

The results of the simulation study of Section 6.1 can be replicated with the notebook ```simulation_study_NIG.ipynb```.

The additional results obtained with the same model presented in the supplementary materials can be replicated with:

- ```simulation_study_NIG_roudning_effect.ipynb``` for the continuous relaxation problem (Section 5.2 of supplementary materials).

- ```simulation_study_NIG_noise_level_effect.ipynb``` for the noise level experiment (Section 5.3 of supplementary materials).

- ```simulation_study_NIG_decrease_uncertainty.ipynb``` and ```simulation_study_NIG_increase_uncertainty.ipynb``` for the uncertainty experiment (Section 5.4 of supplementary materials).


## Replication of the linear regression experiments on real data

The attack on the house prices dataset from Section 6.2 can be replicated with the notebook ```house_prices.ipynb```, and the one against the Mexico microcredit dataset with ```microcredit.ipynb``` (Section 6.2 of supplementary materials).

## Replication of the logistic regression experiments for spam classification

The attack on the spam classifier from Section 6.3 of supplementary materials can be replicated with the notebooks ```spam_classification_send.ipynb``` and  ```spam_classification_your.ipynb```.

