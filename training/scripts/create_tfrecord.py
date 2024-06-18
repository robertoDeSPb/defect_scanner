import os
import tensorflow as tf
from object_detection.utils import dataset_util
from lxml import etree
import io
from PIL import Image
import hashlib

def create_tf_example(xml_path, img_path):
    with tf.io.gfile.GFile(xml_path, 'rb') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = data['filename'].encode('utf8')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for obj in data['object']:
        xmin = float(obj['bndbox']['xmin']) / width
        xmax = float(obj['bndbox']['xmax']) / width
        ymin = float(obj['bndbox']['ymin']) / height
        ymax = float(obj['bndbox']['ymax']) / height
        class_text = obj['name'].encode('utf8')
        class_id = int(obj['class_id'])

        xmins.append(xmin)
        xmaxs.append(xmax)
        ymins.append(ymin)
        ymaxs.append(ymax)
        classes_text.append(class_text)
        classes.append(class_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpeg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    writer = tf.io.TFRecordWriter('D:/Cpp Projects/cv/cvStudy/training/records/train.record')
    path = 'D:/Cpp Projects/cv/cvStudy/dataset'
    annotations = [f for f in os.listdir(path + '/annotations') if f.endswith('.xml')]
    for annotation in annotations:
        img_path = os.path.join(path, 'images', annotation.replace('.xml', '.jpg'))
        xml_path = os.path.join(path, 'annotations', annotation)
        tf_example = create_tf_example(xml_path, img_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    main()