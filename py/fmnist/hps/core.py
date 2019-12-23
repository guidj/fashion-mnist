import base64
import hashlib
import json
import math
import numbers
import os.path
import pickle
from typing import Dict, Any, Callable, Optional, List

import hyperopt
import pandas as pd
import tensorflow as tf

from fmnist import logger, xpath
from fmnist.learning import train

EVALUATED_PARAMS_FILE = 'best-params.json'
TRIALS_PICKLE_FILE = 'trials.pickle'
TRIALS_TABLE_FILE = 'trials.csv'


def create_signature(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True)
    hsh = hashlib.sha224(payload.encode('utf-8')).hexdigest()
    return base64.b64encode(hsh.encode('utf-8')).decode('utf-8')


def create_train_fn(base_data_dir: str, num_threads: int, buffer_size: int, num_epochs: int, shuffle: bool,
                    job_dir: str, model_dir: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    def wrapper_fn(params: Dict[str, Any]):
        signature = create_signature(
            params=params
        )
        task_job_dir = os.path.join(job_dir, signature)
        task_model_dir = os.path.join(model_dir, signature)

        logger.info('Running with config: %s', params)

        def train_fn(batch_size: int, learning_rate: float, dropout_rate: float, activation: str, num_layers: int,
                     layer_size: int, optimizer: str) -> Dict[str, Any]:

            hps_loss, status = math.nan, hyperopt.STATUS_FAIL

            try:
                metrics, export_path = train.train(base_data_dir, num_threads=num_threads, buffer_size=buffer_size,
                                                   batch_size=batch_size, num_epochs=num_epochs, shuffle=shuffle,
                                                   job_dir=task_job_dir, model_dir=task_model_dir,
                                                   learning_rate=learning_rate,
                                                   dropout_rate=dropout_rate, activation=activation, num_layers=num_layers,
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
                        'params': {**params, 'num_epochs': num_epochs}}

        return train_fn(batch_size=params['batch_size'], learning_rate=params['learning_rate'],
                        dropout_rate=params['dropout_rate'],
                        activation=params['activation'], num_layers=params['num_layers'],
                        layer_size=params['layer_size'], optimizer=params['optimizer'])

    return wrapper_fn


def convert_trials_to_data_frame(trials: hyperopt.Trials) -> Optional[pd.DataFrame]:
    def column_repr(source: List[Dict[str, Any]]):
        data = {}
        if source:
            sample = source[0]
            single_value_keys = [k for k, v in sample.items() if isinstance(v, (str, numbers.Integral, numbers.Real))]
            for k in single_value_keys:
                data[k] = []
            for i, rs in enumerate(source):
                for k in single_value_keys:
                    data[k].append(rs[k])
        return data

    if trials.results:
        result_data = column_repr(trials.results)
        params_data = {}

        if 'params' in trials.results[0].keys():
            params = [rs['params'] for rs in trials.results]
            params_data = column_repr(params)

        return pd.DataFrame({**result_data, **params_data})
    else:
        return None


def export_parameters(params: Dict[str, Any], path: str) -> None:
    evaluated_params_path = os.path.join(path, EVALUATED_PARAMS_FILE)
    logger.info('Exporting best params to %s', evaluated_params_path)
    xpath.prepare_path(evaluated_params_path)
    with tf.io.gfile.GFile(evaluated_params_path, 'w') as fp:
        json.dump(params, fp=fp, sort_keys=True)


def export_trials(trials: hyperopt.Trials, path: str) -> None:
    def slim(source: hyperopt.Trials) -> hyperopt.Trials:
        """
        Strips trials to the basic values in order to pickle them
        """
        trials = hyperopt.Trials()
        for tid, trial in enumerate(source.trials):
            docs = hyperopt.Trials().new_trial_docs(
                tids=[trial['tid']],
                specs=[trial['spec']],
                results=[trial['result']],
                miscs=[trial['misc']]
            )

            trials.insert_trial_docs(docs)
            trials.refresh()
        return trials

    trials_pickle_path = os.path.join(path, TRIALS_PICKLE_FILE)
    trials_table_path = os.path.join(path, TRIALS_TABLE_FILE)

    xpath.prepare_path(trials_pickle_path)
    xpath.prepare_path(trials_table_path)

    logger.info('Exporting trials (pickled) to %s', trials_pickle_path)
    with tf.io.gfile.GFile(trials_pickle_path, 'wb') as fp:
        st = slim(trials)
        pickle.dump(st, file=fp)

    logger.info('Exporting trials table (csv) to %s', trials_table_path)
    df = convert_trials_to_data_frame(trials)
    with tf.io.gfile.GFile(trials_table_path, 'w') as fp:
        df.to_csv(fp, header=True, index=False)
