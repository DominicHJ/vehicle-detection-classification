# vehicle-detection-classification

## 1 项目简介  
车辆检测及型号识别广泛应用于物业，交通等的管理场景中。通过在停车场出入口，路口，高速卡口等位置采集的图片数据，对车辆的数量型号等进行识别，可以以较高的效率对车型，数量信息等进行采集。通过采集的数据，在不同的场景中可以辅助不同的业务开展，如商场停车位的规划，路况规划，或者公安系统追踪肇事车辆等等。  
本项目主要是实现对给定车辆的图片进行车辆检测和型号的分类，并对系统进行整合，以GUI界面形式对输入图片进行检测分类后输出结果。  
## 2 环境  
环境：TensorFlow 1.5 / Python 3.6  
框架：Slim  
## 3 数据集  
 数据集为TFRecord格式，共48856张图片包含车辆的图片，其中43971张作为训练集，4885张作为验证集，每张图片只有一辆汽车和一个分类标签，无检测边框。
使用TFRecord2img_vehicle.ipynb中代码可对验证集的图片进行decode，对图片样式进行预览  
![](./ImageforReadme/1.jpg '验证集图片1')  
![](./ImageforReadme/2.jpg '验证集图片2')  
![](./ImageforReadme/3.jpg '验证集图片3')  
## 4 方案选择  
**4.1检测与分类分离方案**  
由于提供的是单分类单标签数据集，没有检测边框，所以检测部分无法进行网络训练或finetune。该方案思路是先使用检测效果较好的预训练模型直接进行图片中车辆的检测，然后把图片传入已使用数据集训练好的分类模型中进行分类，最后将预测边框和分类结果写回原图，完成图片的分类和检测  

**4.2基于CAM的分类和弱监督检测方案**  
该方案适用于无检测标签的检测分类问题，论文提出在进行卷积分类时，卷积网络除了进行特征抽提，还保留了丰富的语义和位置信息，但此功能在进入分类器前经过全连接层后丧失。使用GAP（全局平均池化）代替全连接层，可以使位置信息延续到网络最后，然后再使用热图和上采样等方法将物体的区域还原到原图，在分类的同时也得到了位置检测  

**4.3基于Grad-CAM的分类和弱监督检测方案**  
Grad-CAM是CAM的泛化版本，相对于CAM来说，Grad-CAM不需要改变网络结构，只需在梯度处理上做一些特殊处理，最大程度保留原网络的分类性能，论文中得到的效果比CAM稍微好点  

## 5 方案实现  
**5.1检测与分类分离方案**  
1）分类模型  
使用Inception V4作为BaseNet，使用现有数据集进行finetune，经过两次不同学习率的训练（0.01和0.0001，衰减比例0.94），最终得到在验证集上的准确率为88.4%。将训练好的模型导出后，生成带权重的图协议文件（训练代码文件vehicle_classifiction）。  
![](./ImageforReadme/4.png '训练结果1')  
![](./ImageforReadme/5.png '训练结果2')  
![](./ImageforReadme/6.png '训练结果3')  
![](./ImageforReadme/7.png '训练结果4')  

2）检测模型  
使用[ssd_mobilenet_v2_coco_2018_03_29](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)检测预训练模型，直接对输入图片进行边框检测（该预训练模型使用的数据集是coco，汽车Car分类对应ID为3）  
3）模型结合   

**5.2基于CAM的分类和弱监督检测方案**  
1）参考论文[《Learning Deep Features for Discriminative Localization》](https://arxiv.org/abs/1512.04150)  
2）论文主体思想：  
CNN网络中各卷积核除了提取特征外，其实本身已经具有物体检测功能，即使没有单独对物体的位置检测进行监督学习，而这种能力在使用全连接层进行分类的时候会丧失。通过使用GAP（global average pooling）替代全连接层，可以保持网络定位物体的能力，且相对于全连接网络而言参数更少。论文中提出一种CAM（Class Activation Mapping）方法，可以将CNN在分类时使用的分类依据（图中对应的类别特征）在原图的位置进行可视化，并绘制成热图，以此作为物体定位的依据，具体请见[论文的解读](https://blog.csdn.net/dominic_s/article/details/81209887)。    
![](./ImageforReadme/11.png 'CAM')  


**5.3基于Grad-CAM的分类和弱监督检测方案**  
1）参考论文 [《Visual Explanations from Deep Networks via Gradient-based Localization》](https://arxiv.org/abs/1610.02391)  
2）论文主体思想：  
CAM的实现需要将网络的全连接层替换为GAP（即卷积特征映射→全局平均池化→softmax层），一定程度上改变了原卷积网络的结构，所以适用性受到一些影响。在CAM的基础上有人也提出了更泛化的版本Grad-CAM，这是一种使用梯度信号组合特征映射的方法，该方法不需要对网络架构进行任何修改，所以基本适用于所有的CNN结构得到网络。  
![](./ImageforReadme/12.png '输出结果3')  
![](./ImageforReadme/13.png '输出结果3')  
3）demo实现  
A. 使用VGG作为分类BaseNet，在第5层池化后，抽取特征图，然后对特征图进行反向梯度计算，生成新的特征图。对新的特征图画图，得到被检测到的车辆的热图  
![](./ImageforReadme/14.png 'Grad-CAM结果1')   
![](./ImageforReadme/15.png 'Grad-CAM结果2')   
B. 对热图进行画框和分类，并将分类结果写入边框顶端  
![](./ImageforReadme/16.png 'Grad-CAM结果3')   
C. 生成新的图片并保存   
![](./ImageforReadme/17.png 'Grad-CAM结果4')   

## 6 模型与界面系统  
**6.1 GUI界面**    
1）“导入照片”：点击导入照片  
2）“检测”：导入照片后，点击开始识别图片中的车辆  
3）“自动打开”： 使能后，会自动打开显示运行结果的图片  
![](./ImageforReadme/18.png 'GUI结果1')     
![](./ImageforReadme/19.png 'GUI结果2')   
![](./ImageforReadme/20.jpg 'GUI结果3')   

**6.2 web界面**    
用flask搭建一个web系统，在web页上上传文件，在web页上显示车辆检测及型号识别结果   
1）单辆车检测  
![](./ImageforReadme/21.png 'web结果1')   
2）无车辆检测  
![](./ImageforReadme/22.png 'web结果1')   
3）多车辆检测  
![](./ImageforReadme/23.png 'web结果1')   









