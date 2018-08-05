import tensorflow as tf
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from tensorflow.python.util import compat

from object_detection.utils import label_map_util

import logging

# 检测模型 : ssd
PATH_TO_DETECTION_PB = './object_detection/detection_models/ssd_mobilenet_v2_coco_2018_03_29.pb'    
PATH_TO_DETECTION_LABELS = './object_detection/detection_models/mscoco_label_map.pbtxt'   
DETECTION_NUM_CLASSES = 90

# 分类模型 : inception_v4
PATH_TO_CLASSIFY_PB = './object_detection/detection_models/vehicle_inception_v4_freeze.pb'    
PATH_TO_CLASSIFY_LABELS = './object_detection/detection_models/labels.txt'   
CLASSIFY_NUM_CLASSES = 764


def image_to_numpy(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def detection_box_accuracy(boxes, scores, classes):
    label_map = label_map_util.load_labelmap(PATH_TO_DETECTION_LABELS)    
    categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=DETECTION_NUM_CLASSES,use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
       
    vehicle_id_detect = []
    for cate in category_index.values():
        if cate['name'] == 'car' or cate['name'] == 'truck':
            vehicle_id_detect.append(cate['id'])

    all_class = np.squeeze(classes).astype(np.int32) 
    scores = np.squeeze(scores)
    vehicle_index = [] 
    vehicle_score = []
    vehicle_name = []
    for i, id in enumerate(all_class):
        if i < 10:
            if id in vehicle_id_detect:
                vehicle_index.append(i)
                vehicle_score.append(scores[i])
                vehicle_name.append(category_index[id]['name'])

    all_boxes = np.squeeze(boxes)       
    vehicle_box = [] 
    for idx,vehicle in enumerate(vehicle_index):
        if vehicle_score[idx] > 0.3:                       
            vehicle_box.append(all_boxes[vehicle].tolist())
    return vehicle_name,vehicle_box
    

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
    
def crop_box(boxs,image,image_np):
    (img_width, img_height) = image.size
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

        image_crop = tf.image.crop_to_bounding_box(image_np, 
                                                    offset_height=top, offset_width=left, 
                                                    target_height=bottom-top, target_width=right-left)
        image_preprocessed_list.append(preprocess_for_eval(image_crop, 299, 299))   
    return image_preprocessed_list


def classification_box_accuracy(predictions,vehicle_box):
    vehicle_label = {} 
    with open(PATH_TO_CLASSIFY_LABELS,encoding='utf-8') as f:
        for item in f.readlines():
            item = item.strip().split(':')
            vehicle_label[int(item[0])] = {'id':int(item[0]),'name':item[1]}

    vehicle_accuracy = np.max(predictions,axis=1)
    vehicle_idx = np.argmax(predictions,axis=1)  
    vehicle_predict_name = []
    vehicle_predict_box = []
    for i,idx in enumerate(vehicle_idx):
        if vehicle_accuracy[i] > 0.3:
            vehicle_predict_name.append((str(vehicle_label[idx]['name']) + ': ' + '%.1f'%(vehicle_accuracy[i]*100) + '%'))
        else:
            vehicle_predict_name.append((str('未知车辆') + ': ' + '%.1f'%(vehicle_accuracy[i]*100) + '%'))           
        vehicle_predict_box.append(vehicle_box[i]) 
            
    return vehicle_predict_name,vehicle_predict_box

def classification(image,image_np,vehicle_predict_name,vehicle_predict_box):
    img_width,img_height = image.size
    test_image = Image.fromarray(image_np)                   
    draw = ImageDraw.Draw(test_image)                            
    use_normalized_coordinates=True
   
    for i in range(len(vehicle_predict_box)):
        ymin = vehicle_predict_box[i][0]
        xmin = vehicle_predict_box[i][1]
        ymax = vehicle_predict_box[i][2]
        xmax = vehicle_predict_box[i][3]

        if use_normalized_coordinates:
            (left,right,top,bottom) = (xmin * img_width, xmax * img_width,
                                       ymin * img_height, ymax * img_height)
        else:
            (left,right,top,bottom) = (xmin,xmax,ymin,ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),(right, top),(left,top)], width=8,fill='cyan')

        try:
            font = ImageFont.truetype('simhei.ttf',35,encoding='utf-8')   
        except IOError:
            font = ImageFont.load_default()

        text_width, text_height = font.getsize(vehicle_predict_name[i])  
        text_bottom = top

        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin), (left + text_width,text_bottom)],fill='cyan')
        draw.text( (left + margin, text_bottom - text_height - margin),
                    vehicle_predict_name[i],
                    fill='black',
                    font=font)

    img = np.array(test_image)
    plt.imsave('./out_image/test_label.jpg',img)
    return test_image