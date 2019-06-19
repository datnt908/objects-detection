import os
import glob
import statistics
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from utils import label_map_util
from pascal_voc_tools import XmlReader

THRESHOLD = 0.5
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
MODEL_NAME = 'Exported_Model'
PATH_TO_FROZEN = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABEL_MAP = 'annotations/label_map.pbtxt'
TEST_IMAGES_DIR = 'images/test'
TEST_IMAGES_PATHS = glob.glob(TEST_IMAGES_DIR + '/*.jpg')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABEL_MAP, use_display_name=True)

def read_annotate_xml(xml_file: str):

    reader = XmlReader(xml_file)
    ann_dict = reader.load()
    groundTrues = []
    filename = ann_dict['filename']
    size = [int(ann_dict['size']['width']), int(ann_dict['size']['height'])]

    for obj in ann_dict['object']:
        ymin, xmin, ymax, xmax, label = None, None, None, None, None
        label = obj['name']
        
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])

        box = [xmin, ymin, xmax, ymax]
        groundTrues.append([label, box])

    return size, groundTrues

def bb_intersection_over_union(boxA, boxB):
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	return iou

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections',
                        'detection_boxes',
                        'detection_scores', 
                        'detection_classes', 
                        'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
                
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.int64)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def object_detected_evaluate(size, groundTrues, output_dict):
    prediction = []
    matchedObjs = []

    for i in range(0, output_dict['num_detections']):
        if output_dict['detection_scores'][i] > THRESHOLD:
            boxB = [
                int(float(output_dict['detection_boxes'][i][1]) * int(size[0])),
                int(float(output_dict['detection_boxes'][i][0]) * int(size[1])),
                int(float(output_dict['detection_boxes'][i][3]) * int(size[0])),
                int(float(output_dict['detection_boxes'][i][2]) * int(size[1]))
            ]
            prediction.append(
                [category_index[output_dict['detection_classes'][i]]['name'],
                boxB]
            )

    for groundTrue in groundTrues:
        max_IoU = 0
        max_index = -1
        j = 0
        while j < len(prediction):
            if groundTrue[0] == prediction[j][0]:
                tempIoU = bb_intersection_over_union(groundTrue[1], prediction[j][1])
                if tempIoU > max_IoU:
                    max_IoU = tempIoU
                    max_index = j
            j += 1
        if max_index != -1:
            del prediction[max_index]
        if max_IoU > 0.5:
            matchedObjs.append(1)
        else:
            matchedObjs.append(0)

    for i in prediction:
        matchedObjs.append(0)
    return statistics.mean(matchedObjs)

def load_model_into_memory():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def main_evaluation():
    detection_graph = load_model_into_memory()
    accuracy = 0.

    for imagePath in TEST_IMAGES_PATHS:
        image = Image.open(imagePath)
        image_np = load_image_into_numpy_array(image)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        size, groundTrues = read_annotate_xml(imagePath.replace('.jpg', '.xml'))
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        imageAccuracy = object_detected_evaluate(size, groundTrues, output_dict)
        print("{} has accuracy: {}".format(imagePath.replace('.xml', '.jpg'), imageAccuracy))
        accuracy += (imageAccuracy / len(TEST_IMAGES_PATHS))

    print ("Average Accuracy: {}".format(accuracy))