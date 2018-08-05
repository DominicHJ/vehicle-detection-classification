from flask import Flask, request, jsonify, url_for, send_from_directory
from werkzeug import secure_filename
from datetime import datetime 
import json, uuid, os

from vehicle_detector import VehileDetectionModel, input_for_detecion, draw_predictions, setlog
from vehicle_detector_config import get_models_config, get_log_dir, get_detector_mode
#import logging

setlog(get_log_dir())

print('begain to load models')     
(DETECT_CONFIG, CLASSIFY_CONFIG) = get_models_config()
vehicle_detector = VehileDetectionModel(get_detector_mode(), detect_config= DETECT_CONFIG, classify_config = CLASSIFY_CONFIG)        
print('load models done.')


#导入Flask
app = Flask(__name__)
#创建一个Flask实例

UPLOAD_FOLDER = 'upload'
app.config['IMG_UPLOAD'] = 'images_upload'  # 设置文件上传的目标文件夹
app.config['IMG_RESULT'] = 'images_result'
if not os.path.exists('./' + app.config['IMG_UPLOAD']):
    os.mkdir('./' + app.config['IMG_UPLOAD'])
if not os.path.exists('./' + app.config['IMG_RESULT']):
    os.mkdir('./' + app.config['IMG_RESULT'])
basedir = os.path.abspath(os.path.dirname(__file__))  # 获取当前项目的绝对路径
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'JPEG', 'jpeg'])  # 允许上传的文件后缀

# 判断文件是否合法
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#设置路由，即url
@app.route('/')
#url对应的函数
def hello_world():
    #返回的页面
    return 'Hello World!'

@app.route('/images_result/<path:filename>')
def images_result(filename):
    dirpath = os.path.join(app.root_path, 'images_result')
    print('send file done')
    return send_from_directory(dirpath, filename, as_attachment=True) 

@app.route('/images_upload',methods=['GET', 'POST'])
def images_upload():
    if request.method=='POST':
        print('method POST,run here')
        file_data = request.files['img']
        # check file recieved
        if file_data and allowed_file(file_data.filename):
            print(file_data.filename)
            filename = secure_filename(file_data.filename)
            file_uuid = str(uuid.uuid4().hex)
            time_now = datetime.now()
            filename = time_now.strftime("%Y%m%d%H%M%S")+"_"+file_uuid+"_"+filename
            filepth = './images_upload/' + filename
            file_data.save(filepth)
            # start detection model
            print('file save done,begin to run detection model!')
            img_size, image_np = input_for_detecion(filepth)
            file_type = filepth.split('/')[-1].split('.')[-1]
            vehicle_box, vehicle_predict_lab = vehicle_detector.inference(image_np, img_size)
            # return result image url and detection result
            if vehicle_box:
                image_result = draw_predictions(image_np, img_size, vehicle_box, vehicle_predict_lab)
                image_result.save('./images_result/' + filename.split(file_type)[0] + '_result.jpg')
                img_url = '/images_result/' + filename.split(file_type)[0] + '_result.jpg'
                vh_prob = vehicle_predict_lab[0].split('-')[-1]
                vh_label = vehicle_predict_lab[0].replace('-'+vh_prob,'')
                code  = 0
            else:
                img_url = 'None'
                code = 1
                vh_label = 'None'
                vh_prob = 'None'
            '''
            img_url = '/images_result/' + 'b1.jpg'
            code = 1
            '''
            print('return file')
            print(img_url)
            return jsonify({"img_url":img_url , "code": code, 'vh_label':vh_label, 'vh_prob':vh_prob})
            
        else:
            print('recive data is none')
            return jsonify({"img_url":'None' , "code": 2})
    elif request.method=='GET':
        print('method GET,run here')
        return jsonify({"img_url":'None' , "code": 2})


#这个不是作为模块导入的时候运行，比如这个文件为aa.py，当python aa.py就执行这个代码。如果是在其他文件import的话，不执行这个文件。（这个属于python的基础知识）
if __name__ == '__main__':
    #app.run(host='172.20.95.122',port=8730,debug=True)
    app.run(host='0.0.0.0',port=8081,debug=True)
