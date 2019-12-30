import abc
import base64
import hashlib
import json
import numbers
import os.path
import pickle
from typing import Dict, Any, Optional, List

import hyperopt
import pandas as pd
import tensorflow as tf

from fmnist import logger, xpath

EVALUATED_PARAMS_FILE = 'best-params.json'
TRIALS_PICKLE_FILE = 'trials.pickle'
TRIALS_TABLE_FILE = 'trials.csv'


class SpecTuner(abc.ABC):
    @abc.abstractmethod
    def param_space(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def create_wrapper_fn(self, base_data_dir: str, num_threads: int, buffer_size: int, num_epochs: int, shuffle: bool,
                          job_dir: str, model_dir: str):
        raise NotImplementedError


def create_signature(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True)
    hsh = hashlib.sha224(payload.encode('utf-8')).hexdigest()
    return base64.b64encode(hsh.encode('utf-8')).decode('utf-8')


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
        _trials = hyperopt.Trials()
        for tid, trial in enumerate(source.trials):
            docs = hyperopt.Trials().new_trial_docs(
                tids=[trial['tid']],
                specs=[trial['spec']],
                results=[trial['result']],
                miscs=[trial['misc']]
            )

            _trials.insert_trial_docs(docs)
            _trials.refresh()
        return _trials

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
