import numpy as np
import cv2
import tensorflow as tf

# global face_detection_model, trained_model, labels

face_detection_model = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt.txt',
                                                './models/res10_300x300_ssd_iter_140000_fp16.caffemodel')
trained_modelS = tf.keras.models.load_model('./models/model_S.h5')
trained_modelX = tf.keras.models.load_model('./models/model_X.h5')
target_sizes = [(300, 300), (224, 224)]
labels = ['face', 'incorrect', 'mask']
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2
def getColor(label):
    if label == "mask":
        color = (0,255,0)

    elif label == 'incorrect':
        color = (0,0,255)
    else:
        color = (255,255,0)
        
    return color


def data_preprocess(image, model, flag=False):
    img = image.copy()
    h,w,_ = img.shape
    # Blob
    # 是指任何被认为是大的物体或在黑暗背景中的任何明亮的东西，
    # 在图像中，可以从其背景中分辨出来。

    # OpenCV中认为我们的图片通道顺序是BGR，
    # 但是我平均值假设的顺序是RGB，所以如果需要交换R和G，那么就要使swapRB=true
    blob = cv2.dnn.blobFromImage(img, 1, (300, 300), (104,117,123))
    # 得到检测目标
    face_detection_model.setInput(blob)
    detections = face_detection_model.forward() # 神经网络进行前向传播得到结果
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2] # 获得 confidence score
        if confidence > 0.5: # confidence score 至少要大于 50%
            box = detections[0,0,i,3:7]*np.array([w,h,w,h]) 
            box = box.astype(np.int16)
            #根据index获得box 的真正的坐标
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            # 图像切片只保留脸部部分，为今后的口罩识别做铺垫
            face = image[box[1]:box[3], box[0]:box[2]]
            if model == 'S':
                blob = cv2.dnn.blobFromImage(face, 1, target_sizes[0], (104,117,123))
            else:
                blob = cv2.dnn.blobFromImage(face, 1, target_sizes[1], (104,117,123))
            img_process = cv2.flip(cv2.rotate(np.squeeze(blob).T, cv2.ROTATE_90_CLOCKWISE), 1)
            face = np.maximum(img_process, 0)
            if flag:
                face = (face * 1.0)/255
                faces.append((face, pt1, pt2))
            else:
                faces.append((face))
    return faces

def prediction(img, model='X'):
    image = img.copy()
    faces = data_preprocess(img, model, flag=True)
    for tup in faces:
        processed_img, pt1, pt2 = tup
        y = np.round((pt1[0] + pt2[0])/2 * 0.9)
        x = np.round(pt2[1]*1.05)
        pos = (y.astype(np.int16), x.astype(np.int16))
        if model == 'S':
            shape_y, shape_x = target_sizes[0]
            img_input = processed_img.reshape(1,shape_y, shape_x,3)
            result = trained_modelS.predict(img_input)
        else:
            shape_y, shape_x = target_sizes[1]
            img_input = processed_img.reshape(1,shape_y, shape_x,3)
            result = trained_modelX.predict(img_input)
        confidence_index = result.argmax()
        label = labels[confidence_index]
        color = getColor(label)
        cv2.rectangle(image,pt1,pt2,color,3)
        cv2.putText(image, label, pos, font, fontScale, color, thickness, cv2.LINE_AA)
    return image


