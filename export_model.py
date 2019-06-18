import os
import sys
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2

CONFIG_PATH = os.getcwd() + '/training/ssd_inception_v2.config'
TRAINED_CHECKPOINT = os.getcwd() + '/training/model_training_dir/' + sys.argv[1]
OUTPUT_DIR = os.getcwd() + '/Exported_Model/'

def main():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(CONFIG_PATH, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    input_shape = None
    exporter.export_inference_graph(
        'image_tensor', 
        pipeline_config, 
        TRAINED_CHECKPOINT,
        OUTPUT_DIR, 
        input_shape=input_shape,
        write_inference_graph=False)

main()