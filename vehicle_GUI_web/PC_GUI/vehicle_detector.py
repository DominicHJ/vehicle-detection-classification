# coding: utf-8
import numpy as np
import os
import sys
import tensorflow as tf

from PIL import Image, ImageDraw, ImageFont

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat

from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util

import logging

def setlog(LOG_PATH):
    LOG_FORMAT = "%(asctime)-15s %(levelname)s %(filename)s %(lineno)d %(process)d %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename= LOG_PATH, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def input_for_detecion(input_image):
    image = Image.open(input_image)                      # 打开图片
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return (im_width, im_height), np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def crop_with_bbox(boxs, image_np, image_size):
    (img_width, img_height) = image_size 
    image_preprocessed_list = []
    for cb in boxs:
        ymin = cb[0]
        xmin = cb[1]
        ymax = cb[2]
        xmax = cb[3]

        left = int(xmin * img_width)
        right = int(xmax * img_width)
        top = int(ymin * img_height)
        bottom = int(ymax * img_height)

        #image_crop = tf.convert_to_tensor(image_np[top:bottom,left:right,:])

        image_crop = tf.image.crop_to_bounding_box(image_np, 
                                                    offset_height=top, offset_width=left, 
                                                    target_height=bottom-top, target_width=right-left)
        image_preprocessed_list.append(preprocess_for_eval(image_crop, 299, 299))
    
    return image_preprocessed_list

def preprocess_for_eval(image, height, width,central_fraction=0.875, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)            
        if central_fraction:
            image = tf.image.central_crop(image, central_fraction=central_fraction)
        if height and width:
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [height, width],align_corners=False)
            image = tf.squeeze(image, [0])            
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)      
        return image

def draw_predictions(image_np, image_size, vehicle_box, vehicle_predict_lab):
    img_width, img_height = image_size
    font_size = int(img_width / 200) + 12
    image_result = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_result) 
    use_normalized_coordinates = True
    for i,cb in enumerate(vehicle_box):
        ymin = cb[0]
        xmin = cb[1]
        ymax = cb[2]
        xmax = cb[3]
        
        if use_normalized_coordinates:
            (left,right,top,bottom) = (xmin * img_width, xmax * img_width,
                                    ymin * img_height, ymax * img_height)
        else:
            (left,right,top,bottom) = (xmin,xmax,ymin,ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),(right, top),(left,top)], width=8,fill='cyan')

        try:
            font = ImageFont.truetype('simhei.ttf',font_size,encoding='utf-8') 
        except IOError:
            font = ImageFont.load_default()
            
        text_width, text_height = font.getsize(vehicle_predict_lab[i] )  
        text_bottom = top

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,text_bottom)],fill='cyan')
        draw.text( (left + margin, text_bottom - text_height - margin),
                    vehicle_predict_lab[i],
                    fill='black',
                    font=font)    
    
    return image_result

class VehileDetectionModel(object):
    def __init__(self, mode, detect_config= None, classify_config = None):
        self._mode = mode
        if not detect_config:
            raise ValueError('detection configurations should not be none')

        if mode == 'twostep':
            if not classify_config:
                raise ValueError('classification configurations should not be none')
            
            self.init_twostep(detect_config, classify_config)

    def init_twostep(self, detect_config, classify_config):
        self._vehicle_id_detect, self._detection_graph, self._detection_ops, self._detection_input = self.init_detection(detect_config)
        self._vehicle_lable, self._classify_graph = self.init_classify(classify_config)

    def init_detection(self, detect_config):
        # process lables
        label_map = label_map_util.load_labelmap(detect_config['labels'])
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=detect_config['num_class'],                                                            use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        
        # 找出"car"的编号
        vehicle_id_detect = []
        for cate in category_index.values():
            if cate['name'] == 'car' or cate['name'] == 'truck':
                vehicle_id_detect.append(cate['id'])
        
        # init graph
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(detect_config['pb'], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                logging.debug('Load detection pb success')
            
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')       # 图片张量
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') # 被检测到的物体的边框
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')  # 每一个物体的置信度分值
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')# 类别标签
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            detection_ops = [detection_boxes,detection_scores,detection_classes,num_detections]
        '''
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            #od_graph_def = saved_model_pb2.SavedModel()   
            with tf.gfile.GFile(detect_config['pb'], 'rb') as fid:
                serialized_graph = compat.as_bytes(fid.read())
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def.meta_graphs[0].graph_def, name='')        
        '''
        return vehicle_id_detect, detection_graph, detection_ops, image_tensor
    
    def init_classify(self, classify_config):
        # process labels
        vehicle_lable = {} 
        with open(classify_config['labels'],encoding='utf-8') as f:
            for item in f.readlines():
                item = item.strip().split(':')
                vehicle_lable[int(item[0])] = {'id':int(item[0]),'name':item[1]} 

        # init graph
        classify_graph = tf.Graph()
        with classify_graph.as_default():        
            with open(classify_config['pb'], 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                logging.debug('Load classification pb success')


        return vehicle_lable, classify_graph
     
    def inference(self, image_np, img_size):
        image_np_expanded = np.expand_dims(image_np, axis=0)     # 增加维度，转化为4为张量：[1, None, None, 3]

        with tf.Session(graph=self._detection_graph) as sess:
            (boxes, scores, classes, num) = sess.run(self._detection_ops,
                feed_dict={self._detection_input: image_np_expanded})  
        
        # 找出检测到的车辆的边框
        # 首先查看一下检测的输出参数
        logging.debug('Detection:shape of boxes in detection mode:\n{}'.format(np.squeeze(boxes).shape))
        logging.debug('Detection:labels of the first 10 boxes:\n{}'.format(np.squeeze(classes).astype(np.int32)[:10]))
        logging.debug('Detection:scores of the first 10 boxes:\n{}'.format(np.squeeze(scores)[:10]))
        logging.debug('Detection:total number of valid boxes:%d' % num)
        
        
        # 找出汽车的数组编号及准确度
        all_class = np.squeeze(classes).astype(np.int32)# 3是汽车类别，确定汽车对应的编号
        scores = np.squeeze(scores)
        vehicle_index = []
        vehicle_score = []
        for i,id in enumerate(all_class):
            if id in self._vehicle_id_detect:
                vehicle_index.append(i)
                vehicle_score.append(scores[i])
        if not vehicle_index:
            logging.debug('No car in picture')
            return None,None
        
        logging.debug('Detection:vehicle_index:\n{}'.format(vehicle_index))
        logging.debug('Detection:vehicle_score:\n{}'.format(vehicle_score))
        # 找出汽车对应的边框
        all_boxes = np.squeeze(boxes) 
        logging.debug('Detection:all_boxes:\n{}'.format(all_boxes))
        vehicle_box = []
        for idx, vehicle in enumerate(vehicle_index):
            # 过滤 score较小的值,删除score 小于0.2的box
            if vehicle_score[idx] > 0.2:
                vehicle_box.append(all_boxes[vehicle].tolist())
        if not vehicle_box:
            logging.debug('No valid car in picture')
            return None,None
        logging.debug('Detection:vehicle_box:\n{}'.format(np.array(vehicle_box)))

        # classify
        with self._classify_graph.as_default():
            image_preprocessed_list = crop_with_bbox(vehicle_box, image_np, img_size)
            image_input_tensor = tf.stack(image_preprocessed_list)
            with tf.Session(graph=self._classify_graph) as sess:
                image_input = sess.run(image_input_tensor)
                softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
                predictions = sess.run(softmax_tensor, feed_dict={'input:0': image_input})
        
        vehicle_acc = np.max(predictions,1)
        vehicle_predict_idx = np.argmax(predictions,1)
        vehicle_predict_lab = []
        vehicle_predict_box = []
        logging.debug('Classify:vehicle_acc=\n{}'.format(vehicle_acc))
        for i,idx in enumerate(vehicle_predict_idx):
            if vehicle_acc[i] > 0.1:
                vehicle_predict_lab.append('%s-%.2f' % (self._vehicle_lable[idx]['name'], vehicle_acc[i]))
                vehicle_predict_box.append(vehicle_box[i])
        
        return vehicle_predict_box, vehicle_predict_lab
