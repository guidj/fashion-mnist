import argparse
import enum
import math
import os.path
import time
from typing import Dict, Any, Tuple, Callable

import cloudpickle
import hyperopt
import numpy as np
import pyspark.serializers
from hyperopt import hp
from hyperopt import tpe
from hyperopt.pyll import scope

from fmnist import logger, constants
from fmnist.hps import core

# Patch for: https://github.com/cloudpipe/cloudpickle/issues/305
pyspark.serializers.cloudpickle = cloudpickle

APP_NAME = 'fmnist-hps'


def parse_args() -> argparse.Namespace:
    """
    Parse cmd arguments
    :return: :class:`ArgumentParser` instance
    """
    arg_parser = argparse.ArgumentParser(description='FMNIST HyperParameter Search')
    arg_parser.add_argument('--spec', type=str, choices=[Spec.FCNN.name, Spec.CVNN.name],
                            help='Model to tune.')
    arg_parser.add_argument('--num-epochs', type=int, default=2, help='Num training epochs for each experiment run')
    arg_parser.add_argument('--buffer-size', type=int, default=256, help='Capacity for the reading queue')
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


class Spec(enum.Enum):
    FCNN = 'FCNN'
    CVNN = 'CVNN'


class FCNNTuner(core.SpecTuner):
    def param_space(self) -> Dict[str, Any]:
        return {
            'batch_size': hp.choice('batch_size', options=[2 ** x for x in range(4, 7 + 1)]),
            'learning_rate': hp.loguniform('learning_rate', low=np.log(0.0001), high=np.log(1)),
            'dropout_rate': hp.quniform('dropout_rate', low=0.05, high=0.5, q=0.05),
            'activation': hp.choice('activation', options=['relu', 'selu', 'tanh']),
            'num_layers': scope.int(hp.quniform('num_layers', low=1, high=16, q=2)),
            'layer_size': hp.choice('layer_size', options=[512, 768, 1024, 1536]),
            'optimizer': hp.choice('optimizer', options=['adam', 'adamax', 'nadam', 'rms-prop'])
        }

    def create_wrapper_fn(self, base_data_dir: str, num_threads: int, buffer_size: int, num_epochs: int, shuffle: bool,
                          job_dir: str, model_dir: str):
        def wrapper_fn(params: Dict[str, Any]):
            signature = core.create_signature(
                params={**params, 'tuner': self.__class__.__name__}
            )
            task_job_dir = os.path.join(job_dir, signature)
            task_model_dir = os.path.join(model_dir, signature)

            logger.info('Running with config: %s', params)

            def train_fn(batch_size: int, learning_rate: float, dropout_rate: float, activation: str, num_layers: int,
                         layer_size: int, optimizer: str) -> Dict[str, Any]:
                from fmnist.learning.arch.fcnn import train
                hps_loss, status = math.nan, hyperopt.STATUS_FAIL

                try:
                    metrics, export_path = train.train(base_data_dir, num_threads=num_threads,
                                                       buffer_size=buffer_size,
                                                       batch_size=batch_size, num_epochs=num_epochs,
                                                       shuffle=shuffle,
                                                       job_dir=task_job_dir, model_dir=task_model_dir,
                                                       learning_rate=learning_rate,
                                                       dropout_rate=dropout_rate, activation=activation,
                                                       num_layers=num_layers,
                                                       layer_size=layer_size, optimizer_name=optimizer)
                    if math.isnan(metrics['sparse_categorical_accuracy']) or math.isnan(metrics['loss']):
                        status = hyperopt.STATUS_FAIL
                    else:
                        status = hyperopt.STATUS_OK
                    hps_loss = -math.pow(metrics['sparse_categorical_accuracy'], 2.0)
                except RuntimeError:
                    pass
                finally:
                    return {'loss': hps_loss, 'status': status,
                            'job_dir': task_job_dir, 'model_dir': task_model_dir,
                            'params': {**params, 'num_epochs': num_epochs, 'tuner': self.__class__.__name__}}

            return train_fn(batch_size=params['batch_size'], learning_rate=params['learning_rate'],
                            dropout_rate=params['dropout_rate'],
                            activation=params['activation'], num_layers=params['num_layers'],
                            layer_size=params['layer_size'], optimizer=params['optimizer'])

        return wrapper_fn


class CVNNTuner(core.SpecTuner):
    def param_space(self) -> Dict[str, Any]:
        return {
            'batch_size': hp.choice('batch_size', options=[2 ** x for x in range(4, 6 + 1)]),
            'learning_rate': hp.loguniform('learning_rate', low=np.log(0.0001), high=np.log(1)),
            'num_blocks': scope.int(hp.quniform('num_blocks', low=2, high=6, q=1)),
            'block_size': scope.int(hp.quniform('block_size', low=1, high=3, q=1)),
            'fcl_num_layers': scope.int(hp.quniform('fcl_num_layers', low=1, high=4, q=1)),
            'fcl_layer_size': hp.choice('fcl_layer_size', options=[512, 768, 1024, 1536]),
            'fcl_dropout_rate': hp.quniform('fcl_dropout_rate', low=0.05, high=0.5, q=0.05),
            'activation': hp.choice('activation', options=['relu', 'selu', 'tanh']),
            'optimizer': hp.choice('optimizer', options=['adam', 'adamax', 'nadam', 'rms-prop'])
        }

    def create_wrapper_fn(self, base_data_dir: str, num_threads: int, buffer_size: int, num_epochs: int, shuffle: bool,
                          job_dir: str, model_dir: str):
        def wrapper_fn(params: Dict[str, Any]):
            signature = core.create_signature(
                params={**params, 'class': self.__class__.__name__}
            )
            task_job_dir = os.path.join(job_dir, signature)
            task_model_dir = os.path.join(model_dir, signature)

            logger.info('Running with config: %s', params)

            def train_fn(batch_size: int, learning_rate: float, fcl_dropout_rate: float, activation: str,
                         num_blocks: int, block_size: int,
                         fcl_num_layers: int, fcl_layer_size: int,
                         optimizer: str) -> Dict[str, Any]:
                from fmnist.learning.arch.cvnn import train
                hps_loss, status = math.nan, hyperopt.STATUS_FAIL

                try:
                    metrics, export_path = train.train(base_data_dir, num_threads=num_threads,
                                                       buffer_size=buffer_size,
                                                       batch_size=batch_size, num_epochs=num_epochs,
                                                       shuffle=shuffle,
                                                       job_dir=task_job_dir, model_dir=task_model_dir,
                                                       learning_rate=learning_rate,
                                                       num_blocks=num_blocks, block_size=block_size,
                                                       fcl_dropout_rate=fcl_dropout_rate, activation=activation,
                                                       fcl_num_layers=fcl_num_layers,
                                                       fcl_layer_size=fcl_layer_size, optimizer_name=optimizer)
                    if math.isnan(metrics['sparse_categorical_accuracy']) or math.isnan(metrics['loss']):
                        status = hyperopt.STATUS_FAIL
                    else:
                        status = hyperopt.STATUS_OK
                    hps_loss = -math.pow(metrics['sparse_categorical_accuracy'], 2.0)
                except Exception as err:
                    logger.error(err)
                finally:
                    return {'loss': hps_loss, 'status': status,
                            'job_dir': task_job_dir, 'model_dir': task_model_dir,
                            'params': {**params, 'num_epochs': num_epochs, 'tuner': self.__class__.__name__}}

            return train_fn(batch_size=params['batch_size'], learning_rate=params['learning_rate'],
                            fcl_dropout_rate=params['fcl_dropout_rate'],
                            activation=params['activation'],
                            num_blocks=params['num_blocks'], block_size=params['block_size'],
                            fcl_num_layers=params['fcl_num_layers'],
                            fcl_layer_size=params['fcl_layer_size'], optimizer=params['optimizer'])

        return wrapper_fn


def create_spec_tuner(spec: Spec) -> core.SpecTuner:
    if spec == Spec.FCNN:
        return FCNNTuner()
    elif spec == Spec.CVNN:
        return CVNNTuner()


def tune(param_space: Dict[str, Any], objective_fn: Callable[[Dict[str, Any]], Dict[str, Any]], max_evaluations: int,
         spark_host: str) -> Tuple[Dict[str, Any], hyperopt.Trials]:
    start = time.time()

    if spark_host:
        import pyspark
        spark_session = pyspark.sql.SparkSession(pyspark.SparkContext(master=spark_host, appName=APP_NAME))
        trials = hyperopt.SparkTrials(spark_session=spark_session)
    else:
        trials = hyperopt.Trials()

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
    spec = Spec(args.spec)

    spec_tuner = create_spec_tuner(spec)
    param_space = spec_tuner.param_space()
    objective_fn = spec_tuner.create_wrapper_fn(base_data_dir=base_data_dir,
                                                num_threads=args.num_threads,
                                                buffer_size=args.buffer_size,
                                                num_epochs=args.num_epochs, shuffle=args.shuffle, job_dir=args.job_dir,
                                                model_dir=args.model_dir)

    evaluated_best_params, trials = tune(param_space=param_space,
                                         objective_fn=objective_fn, max_evaluations=args.max_evaluations,
                                         spark_host=args.spark_host)
    core.export_parameters(evaluated_best_params, args.job_dir)
    core.export_trials(trials, args.job_dir)


if __name__ == '__main__':
    main()
