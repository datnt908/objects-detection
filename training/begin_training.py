import os
import sys
import json

import functools
import tensorflow as tf

from object_detection.builders import dataset_builder
from object_detection.builders import graph_rewriter_builder
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from object_detection.utils import config_util

tf.logging.set_verbosity(tf.logging.INFO)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

TRAINING_DIR = os.getcwd() + '/training/model_training_dir'
CONFIG_PATH = os.getcwd() + '/training/ssd_inception_v2.config'
NUM_CLONE = 2
CLONE_ON_CPU = True

if not tf.gfile.IsDirectory(TRAINING_DIR):
    tf.gfile.MakeDirs(TRAINING_DIR)

@tf.contrib.framework.deprecated(
    None, 'Use object_detection/model_main.py.')

def main():
    configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                 job_name=task_info.type,
                                 task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        NUM_CLONE,
        worker_replicas,
        CLONE_ON_CPU,
        ps_tasks,
        worker_job_name,
        is_chief,
        TRAINING_DIR,
        graph_hook_fn=graph_rewriter_fn)

main()
