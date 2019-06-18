import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt 
import glob
import os
import numpy as np
import data_aug
import bbox_util
from pascal_voc_tools import XmlReader
from pascal_voc_tools import XmlWriter

INPUT_DIR = os.getcwd() + '/images/augment/'

def read_content(xml_file: str):

    reader = XmlReader(xml_file)
    ann_dict = reader.load()
    list_with_all_boxes = []
    
    filename = ann_dict['filename']

    for obj in ann_dict['object']:
        ymin, xmin, ymax, xmax, label = None, None, None, None, None
        if obj['name'] == 'cat':
            label = 1.
        else:
            label = 2.
        
        xmin = float(obj['bndbox']['xmin'])
        ymin = float(obj['bndbox']['ymin'])
        xmax = float(obj['bndbox']['xmax'])
        ymax = float(obj['bndbox']['ymax'])

        list_with_single_boxes = [xmin, ymin, xmax, ymax, label]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, np.array(list_with_all_boxes)

def replace_content(xml_file: str, bboxes):
    reader = XmlReader(xml_file)
    ann_dict = reader.load()
    list_with_all_boxes = []
    
    ann_dict['filename'] = ann_dict['filename'].replace('.jpg', '-flip.jpg')
    index = 0
    for obj in ann_dict['object']:
        if bboxes[index][4] == 1.:
            obj['name'] = 'cat'
        else:
            obj['name'] = 'dog'
        obj['bndbox']['xmin'] = int(bboxes[index][0])
        obj['bndbox']['ymin'] = int(bboxes[index][1]) 
        obj['bndbox']['xmax'] = int(bboxes[index][2])
        obj['bndbox']['ymax'] = int(bboxes[index][3])

    writer = XmlWriter('a',1,1)
    writer.save(xml_file.replace('.xml', '-flip.xml'), ann_dict)

def main():
    xml_list = []
    for xml_file in glob.glob(INPUT_DIR + '/*.xml'):
        imgName, bboxes = read_content(xml_file)
        imagepath = INPUT_DIR + imgName
        img = cv2.imread(imagepath)
        # Flip
        img_, bboxes_ = data_aug.RandomHorizontalFlip(1)(img.copy(), bboxes.copy())
        replace_content(xml_file, bboxes_)
        cv2.imwrite(imagepath.replace('.jpg', '-flip.jpg'), img_)

main()