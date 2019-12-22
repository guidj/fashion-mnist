import argparse
import os.path
import time
from typing import Dict, Any, Tuple

import hyperopt
import numpy as np
from hyperopt import hp
from hyperopt import tpe
from hyperopt.pyll import scope

from fmnist import logger, constants
from fmnist.hps import core

APP_NAME = 'fmnist-hps'

PARAM_SPACE = {
    'batch_size': hp.choice('batch_size', options=[2 ** x for x in range(3, 7 + 1)]),
    'learning_rate': hp.loguniform('learning_rate', low=np.log(0.0001), high=np.log(1)),
    'dropout_rate': hp.quniform('dropout_rate', low=0.05, high=0.5, q=0.05),
    'activation': hp.choice('activation', options=['relu', 'selu', 'tanh']),
    'num_layers': scope.int(hp.quniform('num_layers', low=1, high=16, q=2)),
    'layer_size': hp.choice('layer_size', options=[2 ** x for x in range(6, 10 + 1)])
}


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST HyperParameter Search')
    arg_parser.add_argument('--num-epochs', type=int, default=2, help='Num training epochs for each experiment run')
    arg_parser.add_argument('--buffer-size', type=int, default=1024, help='Capacity for the reading queue')
    arg_parser.add_argument('--num-threads', type=int, default=1, help='Number of threads for processing data')
    arg_parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    arg_parser.add_argument('--job-dir', required=True, help='Path to job dir')
    arg_parser.add_argument('--model-dir', required=True, help='Path to model dir')
    arg_parser.add_argument('--train-data', required=True, help='Path to input data path')
    arg_parser.add_argument('--max-evaluations', type=int, required=False, default=2, help='Max number of experiments')
    arg_parser.add_argument('--spark-host', type=str, required=False, default=None,
                            help='Hostname of spark server to use Apache Spark for parallel tuning.')
    arg_parser.set_defaults(shuffle=True)

    args = arg_parser.parse_args()
    logger.info('Running with args:')
    for arg in vars(args):
        logger.info('\t%s: %s', arg, getattr(args, arg))

    return args


def tune(param_space: Dict[str, Any], max_evaluations: int, base_data_dir: str, num_threads: int, buffer_size: int,
         num_epochs: int, shuffle: bool, job_dir: str, model_dir: str,
         spark_host: str) -> Tuple[Dict[str, Any], hyperopt.Trials]:
    start = time.time()

    if spark_host:
        import pyspark
        spark_session = pyspark.sql.SparkSession(pyspark.SparkContext(master=spark_host, appName=APP_NAME))
        trials = hyperopt.SparkTrials(spark_session=spark_session)
    else:
        trials = hyperopt.Trials()

    objective_fn = core.create_train_fn(base_data_dir, num_threads=num_threads, buffer_size=buffer_size,
                                        num_epochs=num_epochs, shuffle=shuffle, job_dir=job_dir, model_dir=model_dir)

    best_params = hyperopt.fmin(objective_fn,
                                param_space,
                                algo=tpe.suggest,
                                max_evals=max_evaluations,
                                trials=trials,
                                rstate=np.random.RandomState(1777))
    evaluated_best_params = hyperopt.space_eval(param_space, best_params)
    losses = [x['result']['loss'] for x in trials.trials]

    logger.info('Score best parameters: %f', min(losses) * -1)
    logger.info('Best parameters: %s', evaluated_best_params)
    logger.info('Time elapsed: %s', time.strftime("%H:%M:%S", time.gmtime(time.time() - start)))
    logger.info('Parameter combinations evaluated: %d', max_evaluations)

    return evaluated_best_params, trials


def main():
    args = parse_args()

    base_data_dir = os.path.join(args.train_data, constants.DataPaths.INTERIM)

    evaluated_best_params, trials = tune(PARAM_SPACE, max_evaluations=args.max_evaluations, base_data_dir=base_data_dir,
                                         num_threads=args.num_threads, buffer_size=args.buffer_size,
                                         num_epochs=args.num_epochs,
                                         shuffle=args.shuffle, job_dir=args.job_dir, model_dir=args.model_dir,
                                         spark_host=args.spark_host)
    core.export_parameters(evaluated_best_params, args.job_dir)
    core.export_trials(trials, args.job_dir)


if __name__ == '__main__':
    main()
