import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image

from vehicle_detector_config import get_models_config, get_log_dir, get_detector_mode
from vehicle_detector import VehileDetectionModel, input_for_detecion, draw_predictions, setlog

class inference_thread(QThread):
    # inference 线程
    statusSignal = pyqtSignal(list)
    def __init__(self, filenames, vehicle_detector, parent=None):
        super(inference_thread, self).__init__(parent)
        self._filenames = filenames
        self.vehicle_detector = vehicle_detector

    def run(self):
        for idx, filename in enumerate(self._filenames):
            img_size, image_np = input_for_detecion(filename)
            file_type = filename.split('/')[-1].split('.')[-1]
            vehicle_box, vehicle_predict_lab = self.vehicle_detector.inference(image_np, img_size)
            if vehicle_box:
                image_result = draw_predictions(image_np, img_size, vehicle_box, vehicle_predict_lab)
                image_result.save(filename.split(file_type)[0] + '_result.jpg')
                # signal runing:car detected
                self.statusSignal.emit([1, filename, idx, image_result])
            else:
                # signal runing:no car detected
                self.statusSignal.emit([-1, filename, idx])
        # signal finished
        self.statusSignal.emit([0])

class load_models_thread(QThread):
    # 导入model线程
    finishSignal = pyqtSignal(VehileDetectionModel)
    def __init__(self, parent=None):
        super(load_models_thread, self).__init__(parent)

    def run(self):
         #get models config
        (DETECT_CONFIG, CLASSIFY_CONFIG) = get_models_config()
        vehicle_detector = VehileDetectionModel(get_detector_mode(), detect_config= DETECT_CONFIG, classify_config = CLASSIFY_CONFIG)
        self.finishSignal.emit(vehicle_detector)

class VehicleGui(QWidget):
    def __init__(self):
        super().__init__()
        self.show_wind_list = []
        self.path = './'
        self.setUI()
    
    def setUI(self):
        # set main window information
        self.setWindowIcon(QIcon("./ico/total.ico"))
        self.setWindowTitle(u"车辆识别")
        self.setToolTip("<b>车辆识别</b>")
        
        # add button to GUI
        btn_load = QPushButton("导入照片",self)
        btn_load.setToolTip("点击导入需要检测的图片")
        btn_load.setDisabled(True)
        btn_inference = QPushButton("检测",self)
        btn_inference.setToolTip("点击开始检测导入的图片")
        btn_inference.setDisabled(True)

        # add checkbox to GUI
        cbx_auto_open = QCheckBox("自动打开", self)
        cbx_auto_open.setToolTip("自动显示检测结果图片")
        cbx_auto_open.setChecked(True)

        # add list
        self.debug_info = QListWidget()
        self.debug_info.setToolTip("日志窗口")

        # horizon
        h_box = QHBoxLayout()
        h_box.addWidget(btn_load)
        h_box.addWidget(btn_inference)
        h_box.addWidget(cbx_auto_open)

        # vertical
        v_box = QVBoxLayout()
        v_box.addLayout(h_box)
        v_box.addWidget(self.debug_info)

        self.setLayout(v_box)
        self.resize(QSize(500,400))

        btn_load.clicked.connect(self.load_image)
        btn_inference.clicked.connect(self.inference)
        self.btn_inference = btn_inference
        self.btn_load = btn_load
        self.cbx_auto_open = cbx_auto_open
        self.center()
        self.show()
        setlog(get_log_dir())
        self.debug_info.addItem('Status: BUSY : Set log dir:{}'.format(get_log_dir().split('/')[-1]))
        self.debug_info.addItem('Status: BUSY : Loading detection model, please wait...')
        
        # load model pb
        self.loadThread = load_models_thread()
        self.loadThread.finishSignal.connect(self.load_models_end)
        self.loadThread.start()

    def load_models_end(self, result):
        self.vehicle_detector = result
        self.btn_load.setDisabled(False)
        self.debug_info.addItem('Status: IDLE : Load detection model done!')

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
        #self.move(cp)

    def load_image(self):
        filenames, _ = QFileDialog.getOpenFileNames(self,
                                    "选取文件",
                                    self.path,
                                    "Images (*.jpeg;*.jpg;*.png);;All Files (*)")
        
        if not filenames:
            return
        
        self._filenames = []
        self.path = filenames[0].split(filenames[0].split('/')[-1])[0]
        self.im_num = len(filenames)
        for filename in filenames:
            file_type = filename.split('/')[-1].split('.')[-1]
            if file_type in ['jpg','jpeg','png']:
                self.debug_info.addItem('Status: BUSY :Load image: ' + filename.split('/')[-1])
                self.btn_inference.setDisabled(False)
                self._filenames.append(filename)
            elif file_type:
                #QMessageBox.information(self, "提示", "不支持的格式")
                self.debug_info.addItem('Not support file type of loaded file:' +  filename.split('/')[-1])

        self.debug_info.addItem('Status: IDLE :Load images done!')

    def inference(self):
        # update gui
        self.btn_inference.setDisabled(True)
        self.debug_info.addItem('Status: BUSY :The program is processing images,please wait... ')
        # create thread
        self.infThread = inference_thread(self._filenames, self.vehicle_detector)
        self.infThread.statusSignal.connect(self.inference_process)
        self.infThread.start()
    
    def inference_process(self, result):
        if result[0] == 1:
            self.debug_info.addItem('Status:BUSY : {}/{},{} proceed done' .format(result[2]+1, self.im_num, result[1].split('/')[-1]))
            if self.cbx_auto_open.isChecked():
                result[3].show()
        elif result[0] == -1:
            self.debug_info.addItem('Status:BUSY : {}/{},{} proceed done. No car in this image ' .format(result[2]+1, self.im_num, result[1].split('/')[-1]))
        elif result[0] == 0:
            self.btn_inference.setDisabled(False)
            self.debug_info.addItem('Status:IDLE : All images proceed ')


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建一个应用
    vehicle = VehicleGui()
    sys.exit(app.exec_())  # 将app的退出信号传给程序进程，让程序进程退出