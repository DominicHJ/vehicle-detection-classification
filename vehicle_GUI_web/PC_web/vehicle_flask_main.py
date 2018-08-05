import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import uuid

from vehicle_detection import *
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from PIL import Image,ImageDraw, ImageFont
from shutil import copy2

from flask import Flask, request, redirect, send_from_directory, url_for


app = Flask(__name__)

# 检测模型 : ssd
PATH_TO_DETECTION_PB = './object_detection/detection_models/ssd_mobilenet_v2_coco_2018_03_29.pb'    
PATH_TO_DETECTION_LABELS = './object_detection/detection_models/mscoco_label_map.pbtxt'   
DETECTION_NUM_CLASSES = 90

# 分类模型 : inception_v4
PATH_TO_CLASSIFY_PB = './object_detection/detection_models/vehicle_inception_v4_freeze.pb'    
PATH_TO_CLASSIFY_LABELS = './object_detection/detection_models/labels.txt'   
CLASSIFY_NUM_CLASSES = 764

app._static_folder = './static'
UPLOAD_FOLDER = './static/test_image'
OUTPUT_FOLDER = './static/out_image'
ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])


def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_filename):
    basename = os.path.basename(old_filename)
    name, ext = os.path.splitext(basename)
    new_filename = str(uuid.uuid1()) + ext
    return new_filename  

def vehicle_detection_classify(image_name, detection_config, classify_config):    
    # 加载检测模型
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = saved_model_pb2.SavedModel()
        with tf.gfile.GFile(detection_config, 'rb') as fid:
            serialized_graph = compat.as_bytes(fid.read())
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def.meta_graphs[0].graph_def, name='')
            
        # 检测    
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')        
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')  
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')   
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image = Image.open(image_name)  
            img_width,img_height = image.size
            image_np = image_to_numpy(image)            
            image_np_expanded = np.expand_dims(image_np, axis=0)     

            (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],\
                                                      feed_dict={image_tensor: image_np_expanded})       
    vehicle_name,vehicle_box = detection_box_accuracy(boxes, scores, classes)

    # 如果检测到车辆，对车辆分类，识别型号
    if vehicle_name and vehicle_box:
        classify_graph = tf.Graph()
        with classify_graph.as_default():        
            with open(classify_config, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

           # 分类
            image_preprocessed_list = crop_box(vehicle_box, image,image_np)
            image_input_tensor = tf.stack(image_preprocessed_list)
            with tf.Session(graph=classify_graph) as sess:
                image_input = sess.run(image_input_tensor)
                softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
                predictions = sess.run(softmax_tensor, feed_dict={'input:0': image_input})

        vehicle_predict_name,vehicle_predict_box = classification_box_accuracy(predictions,vehicle_box)

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

        im = np.array(test_image)    
        plt.imsave(os.path.join(OUTPUT_FOLDER, os.path.basename(image_name)), im)

        image_height = int(img_height/2)
        image_width = int(img_width/2)

        image_detection = OUTPUT_FOLDER + '/%s' % os.path.basename(os.path.join(OUTPUT_FOLDER, image_name))
        image_tag = '<img src="%s" height="%d" width="%d"></img><p>'
        image_detection_tag = image_tag % (image_detection,image_height,image_width) 

        show_result = '<b>检测到的车辆型号如下：</b><br/>'
        for name in vehicle_predict_name:
            show_result += name + '<br>'
        show_all_result  = image_detection_tag + show_result + '<br>'
        return show_all_result
    
    # 如果没有检测到车辆，显示原图
    elif not vehicle_name:
        plt.imsave(os.path.join(OUTPUT_FOLDER, os.path.basename(image_name)),image_np)
        
        image_height = int(img_height/2)
        image_width = int(img_width/2)

        image_detection = OUTPUT_FOLDER + '/%s' % os.path.basename(os.path.join(OUTPUT_FOLDER, image_name))
        image_tag = '<img src="%s" height="%d" width="%d"></img><p>'
        image_detection_tag = image_tag % (image_detection,image_height,image_width) 

        show_result = '<b>图片中没有汽车</b><br/>'
        show_all_result  = image_detection_tag + show_result + '<br>'
        return show_all_result
        
    
@app.route("/", methods=['GET','POST'])
def root(): 
    vehicle_result = """
        <!doctype html>
        <body bgcolor="Plum"> 
        <title>车辆检测及型号识别</title>
        <font size=5 color=black> 车辆检测及型号识别 </font><br><br> 
        <font size=4 color=blue> 上传图片，检测图片中车辆位置、并识别其型号 </font> <br>               
        <form action="" method=post enctype=multipart/form-data>
        <p><input type=file name=file value='选择图片' style='font-size:20px'> 
            <input type=submit value='上传图片' style='font-size:20px'> 
        </form>
        <p>%s</p>
        """ % "<br>"  
    
    if request.method == 'POST':
        file = request.files['file']
        old_filename = file.filename
        if file and allowed_files(old_filename):
            filename = rename_filename(old_filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)
            type_name = 'N/A' 
            out_image = vehicle_detection_classify(image_path,detection_config=PATH_TO_DETECTION_PB,classify_config=PATH_TO_CLASSIFY_PB)     
            return vehicle_result + out_image 
    return vehicle_result
       
    
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80, debug=True, threaded=True)
    