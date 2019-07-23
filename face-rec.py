# -*- coding: UTF-8 -*-
import sys, os, dlib, glob, numpy
from skimage import io

if len(sys.argv) != 5:
    print
    "请检查参数是否正确"
    exit()

# 1.人脸关键点检测器
predictor_path = sys.argv[1]
# 2.人脸识别模型
face_rec_model_path = sys.argv[2]
#print (face_rec_model_path)
# 3.候选人脸文件夹
faces_folder_path = sys.argv[3]
#print("人脸文件夹",faces_folder_path)
# 4.需识别的人脸
img_path = sys.argv[4]
#print("需要是别的人脸",img_path)
# 1.加载正脸检测器
detector = dlib.get_frontal_face_detector()
# 2.加载人脸关键点检测器
sp = dlib.shape_predictor(predictor_path)
# 3. 加载人脸识别模型
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

#win = dlib.image_window()

# 候选人脸描述子list
descriptors = []

# 对文件夹下的每一个人脸进行:
# 1.人脸检测
# 2.关键点检测
# 3.描述子提取
#print("1")
for f in glob.glob('candidate-faces/*.jpg'):
    #print("1")
    print("Processing file: {}".format(f))
    img = io.imread(f)
    # win.clear_overlay()
    # win.set_image(img)

    # 1.人脸检测
    dets = detector(img, 1)
    #print(dets)
    #print("Number of faces detected: {}".format(len(dets)))

    # 2.关键点检测
    for k, d in enumerate(dets):
        shape = sp(img, d)
    # 画出人脸区域和和关键点
    # win.clear_overlay()
    # win.add_overlay(d)
    # win.add_overlay(shape)

    # 3.描述子提取，128D向量
        face_descriptor = facerec.compute_face_descriptor(img, shape)

    # 转换为numpy array
        v = numpy.array(face_descriptor)
        #print(v)
        descriptors.append(v)
#print(descriptors)
# 对需识别人脸进行同样处理
# 提取描述子，不再注释
img = io.imread(img_path)
dets = detector(img, 1)

dist = []
for k, d in enumerate(dets):
    shape = sp(img, d)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    d_test = numpy.array(face_descriptor)

    # 计算欧式距离

    for i in descriptors:
        dist_ = numpy.linalg.norm(i - d_test)
        #print(dist_)
        dist.append(dist_)

# 候选人名单
candidate = ['海涛','刘德华', '刘诗诗', '刘亦菲',  '孙笑川', '唐嫣', '杨幂']

# 候选人和距离组成一个dict
c_d = dict(zip(candidate, dist))

cd_sorted = sorted(c_d.items(), key=lambda d: d[1])
#print(cd_sorted)
#print(c_d)
print ("\n图片", img_path, " 中的人是: ", cd_sorted[0][0])
#dlib.hit_enter_to_continue()

#python face-rec.py shape_predictor_68_face_landmarks.dat dlib_face_recognition_resnet_model_v1.dat ./candidate-faecs ./picturetest/test5.jpg
