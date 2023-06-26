import numpy as np
from skopt import Optimizer
from skopt.utils import use_named_args
from skopt.space import Real, Integer, Categorical
from vcausal.causaldiscovery.utils.evaluation import Evaluation


class HyperparameterOptimizer:
    @staticmethod
    def optimize_hyperparameters(method_obj, method_name, dataset, true_dag, true_adj_matrix, n_calls=50):
        # Define the hyperparameter search space
        search_space = []
        for param, value in method_obj.get_hyperparameters().items():
            if isinstance(value, float):
                search_space.append(Real(value * 0.1, value * 10, name=param))
            elif isinstance(value, int):
                search_space.append(Integer(max(value - 10, 1), value + 10, name=param))
            elif isinstance(value, str):
                search_space.append(Categorical([value], name=param))

        # Define the objective function
        @use_named_args(search_space)
        def objective(**params):
            method_obj.set_hyperparameters(params)
            result = method_obj.run_method()
            eval_obj = Evaluation()
            score = eval_obj.evaluate_shd(true_dag, result['dag'])  # Use SHD as the optimization metric
            return score

        # Run the optimization
        optimizer = Optimizer(search_space)
        for _ in range(n_calls):
            suggested_params = optimizer.ask()
            score = objective(suggested_params)
            optimizer.tell(suggested_params, score)

        # Set the best hyperparameters
        best_params = optimizer.Xi[np.argmin(optimizer.yi)]
        method_obj.set_hyperparameters(dict(zip([param.name for param in search_space], best_params)))

        print(f"Optimized hyperparameters for {method_name}:")
        print(method_obj.get_hyperparameters())