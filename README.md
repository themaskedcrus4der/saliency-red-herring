activmask
---------

**installation**:

`conda env create -f environment.yml`.


**data**:

An easy and hard version of the synthetic dataset are included. All experiments
were done using `synth_hard`.

**training**:

A config file needs to be defined to run the experiments, e.g.:

```
# A single run.
activmask train --config activmask/config/mnist.yml

# A hyperparamater search using skopt.
activmask train-skopt --config activmask/config/mnist-search.yml
```

**skopt**

Steps to launch bayesian hyperparameters search:
1. In your `.yml` config file, choose the parameters you want to optimize
   (i.e. learning rate).
2. Replace the value with  the search parameters. For example:
    ```
    # Optimizer
    optimizer:
      Adam:
        lr: "Real(10**-4, 10**-2, 'log-uniform')"
    ```
    search the learning rate in the range (0.01, 0.0001), on a log scale.
    [Examples.](https://scikit-optimize.github.io/#skopt.BayesSearchCV).
