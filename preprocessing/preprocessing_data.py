import os
import xml_to_csv
import pandas as pd
import tensorflow as tf
import generate_tfrecords as tfrecord

INPUT_DIR = os.getcwd() + '/images/'
OUTPUT_DIR = os.getcwd() + '/annotations/'

def create_csv_from_xml_train():
    #preprocessing training data
    xml_df_train = xml_to_csv.xml_to_csv(INPUT_DIR + 'train')
    xml_df_train.to_csv(OUTPUT_DIR + 'train_anno.csv', index=None)
    output_path = os.path.join(OUTPUT_DIR + 'train_anno.csv')
    print('Successfully created the TFRecords: {}'.format(output_path))

def create_csv_from_xml_test():
    #preprocessing testing data
    xml_df_test = xml_to_csv.xml_to_csv(INPUT_DIR + 'test')
    xml_df_test.to_csv(OUTPUT_DIR + 'test_anno.csv', index=None)
    output_path = os.path.join(OUTPUT_DIR + 'test_anno.csv')
    print('Successfully created the TFRecords: {}'.format(output_path))

create_csv_from_xml_train()
create_csv_from_xml_test()

def create_train_record_from_csv():
    writer = tf.python_io.TFRecordWriter(OUTPUT_DIR + 'train.record')
    path = os.path.join(INPUT_DIR + 'train')
    examples = pd.read_csv(OUTPUT_DIR + 'train_anno.csv')
    grouped = tfrecord.split(examples, 'filename')
    for group in grouped:
        tf_example = tfrecord.create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(OUTPUT_DIR + 'train.record')
    print('Successfully created the TFRecords: {}'.format(output_path))

def create_test_record_from_csv():
    writer = tf.python_io.TFRecordWriter(OUTPUT_DIR + 'test.record')
    path = os.path.join(INPUT_DIR + 'test')
    examples = pd.read_csv(OUTPUT_DIR + 'test_anno.csv')
    grouped = tfrecord.split(examples, 'filename')
    for group in grouped:
        tf_example = tfrecord.create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(OUTPUT_DIR + 'test.record')
    print('Successfully created the TFRecords: {}'.format(output_path))

create_train_record_from_csv()
create_test_record_from_csv()
