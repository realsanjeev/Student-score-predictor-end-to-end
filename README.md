# ML project

## Set Up environment
```
virtualenv env
source ./venv/Scripts/activate
```
> To shorten path in terminal: PROMPT_DIRTRIM=1

To install dependencies in package: `python setup.py install`. This method is decrecated.

#### Recomendation
```
pip install .
```
This command assumes that you are in the root directory of the project where the `setup.py` file is located. The `.` specifies the current directory as the source for installation.

### Dataclass
A `dataclass` is a class that is designed to only hold data values. They aren't different from regular classes, but they usually don't have any other methods. They are typically used to store information that will be passed between different parts of a program or a system.

### Project Structure
```
Ml_project\
    src\
        components\
            data_ingestion.py
            data_transformation.py
        pipeline\
            pipeline.py
        __init__.py
        exception.py
        logger.py
        utils.py
setup.py
```

## Ensemble Techinique
### `AdaBoostRegressor` and `GradientBoostingRegressor`
1. **Algorithm:** `AdaBoostRegressor` uses the AdaBoost algorithm, which combines multiple weak learners in a sequential manner. It assigns higher weights to the misclassified instances in each iteration to focus on those instances and improve the model's performance. On the other hand, `GradientBoostingRegressor` uses the gradient boosting algorithm, which also combines multiple weak learners, but it fits subsequent models to the residuals (errors) of the previous models, attempting to minimize the loss function through gradient descent.

2. **Loss Function:** `AdaBoostRegressor` minimizes the exponential loss function by iteratively adjusting the weights of the training instances. `GradientBoostingRegressor`, on the other hand, can work with different loss functions, such as least squares loss (default), least absolute deviation (L1 loss), or huber loss, allowing more flexibility in capturing different types of relationships in the data.

3. **Weighting of Weak Learners:** In `AdaBoostRegressor`, each weak learner is assigned a weight based on its performance. Weak learners with lower errors are given higher weights. In contrast, `GradientBoostingRegressor` assigns weights to the weak learners based on the gradients of the loss function, indicating the direction and magnitude of the error reduction.

4. **Model Complexity:** `AdaBoostRegressor` typically uses simple weak learners, such as decision stumps (a decision tree with a single split). It focuses on iteratively correcting the mistakes made by these weak learners. `GradientBoostingRegressor` can use more complex weak learners, usually decision trees with multiple levels, to better capture complex relationships in the data.

5. **Hyperparameters:** Both algorithms have their specific set of hyperparameters. While they share some common hyperparameters (e.g., number of estimators), they may have different hyperparameters related to their respective algorithms. For example, `AdaBoostRegressor` has parameters like learning rate and loss function, whereas `GradientBoostingRegressor` has parameters like learning rate, maximum tree depth, and subsampling rate.


### doctest