import numpy as np
import pandas as pd
import re
from tqdm import tqdm
import os
import cv2
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Flatten, Dropout, Conv2D, Input, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from mtcnn.mtcnn import MTCNN
from scipy.spatial import distance as dist

from keras import backend as K
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.layers import concatenate
from keras.models import load_model


def init():
    global ROOT_PATH, TRAIN_IMG_DIR, CROPPED_IMG_DIR, VAL_IMG_DIR, FRAME_COUNTER, YAW_STACK, SMOOTH

    SMOOTH = 3
    FRAME_COUNTER = 0

    YAW_STACK = np.zeros((SMOOTH))

    ROOT_PATH = r'/'

    TRAIN_IMG_DIR = r'train_img'

    CROPPED_IMG_DIR = r'cropped_faces'

    VAL_IMG_DIR = r'val_img'

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    detector = MTCNN()

    return camera, detector

def filename_parser(fn : str):

    arr = np.array(re.findall(r"[+-]?\d+(?:\.\d+)?", fn)).astype("float32")

    return arr[1:]

def image_reading(s : int):
    X = []
    y = []

    if s == 0:
        dirct = TRAIN_IMG_DIR
    elif s == 1:
        dirct = VAL_IMG_DIR
    else:
        print('Wrong key for the image_reading function')
        exit()

    for directory, _, files in os.walk(os.path.join(ROOT_PATH, dirct)):
        for file in tqdm(files):
            # print(_)
            if file.endswith('jpg') or file.endswith('png'):

                image = cv2.imread(os.path.join(directory,file))

                X.append(image)
                y.append(filename_parser(file))

    return np.array(X), np.array(y)

def create_mlp(X_train):

    model = Sequential()
    model.add(Dense(8, input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, kernel_initializer='normal', activation='relu'))

    return model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def get_landmarks(face):

    pts = []

    landmarks = face[0]['keypoints']

    for k, point in landmarks.items():
        pts.append(point)

    return pts

def predict_and_show(cam, loaded_model, detector):
    global FRAME_COUNTER, YAW_STACK, SMOOTH

    ret, frame = cam.read()

    font = cv2.FONT_HERSHEY_SIMPLEX
    scaleFactor = 0.4
    thickness = 1

    faces = detector.detect_faces(frame)

    if len(faces) > 0:
        for face in zip(faces):
            (x, y, w, h) = face[0]['box']

            m = max(w, h)

            y_min =  y - int(0.25*m) if y - int(0.25*m) > 0 else 0
            y_max =  y+m + int(0.25*m) if y+m + int(0.25*m) > 0 else frame.shape[0]
            x_min = x - int(0.25*m) if x - int(0.25*m) > 0 else 0
            x_max = x+m + int(0.25*m) if x+m + int(0.25*m) > 0 else frame.shape[1]

            cropped = frame[y_min : y_max, x_min : x_max]

            cropped = cv2.resize(cropped, (128, 128))

            if face[0]['confidence'] >= 0.7:

                cropped_face = detector.detect_faces(cropped)

                if len(cropped_face) > 0:
                    p = get_proportions(cropped_face)

                    lmks = get_landmarks(cropped_face)
                    angle, rotation_centre, dist = get_angle(lmks)

                    M = cv2.getRotationMatrix2D(rotation_centre, angle, 1)

                    (w, h) = (128, 128)

                    cropped_h, cropped_w = cropped.shape[:2]

                    cropped = cv2.warpAffine(cropped, M, (w, h), flags=cv2.INTER_CUBIC)

                    frame[:cropped_h, :cropped_w, :] = cropped[:,:,:]

                    num_features = get_features_from_dict(p)

                    cropped = cropped / 255

                    cropped = cropped.reshape((1, cropped.shape[0], cropped.shape[1], 3))

                    predicted = loaded_model.predict([num_features, cropped])

                    yaw = predicted[0][0]

                    # This block smoothes prediction (3 frames)
                    YAW_STACK[FRAME_COUNTER] = yaw
                    yaw = YAW_STACK.mean()
                    FRAME_COUNTER = FRAME_COUNTER + 1 if FRAME_COUNTER < SMOOTH - 1 else 0

                    cv2.putText(frame, 'Head yaw: %.1f' % (yaw), (frame.shape[1] - 150, 20), font, scaleFactor, (0, 255, 0), thickness)

            return frame

    return frame

def crop_save_create(trainset_x, trainset_y):

    X = pd.DataFrame()
    a = []
    img_arr = []

    for id, (sample, aim) in enumerate(zip(trainset_x, trainset_y)):
        if id == 279:
            continue
        fd = detector.detect_faces(sample)

        if len(fd) > 0:
            for face in zip(fd):
                (x, y_, w, h) = face[0]['box']

                m = max(w, h)

                cropped = sample[y_ - int(0.25*m) : y_+m + int(0.25*m), x - int(0.25*m) : x+m + int(0.25*m)]
                cropped = cv2.resize(cropped, (128, 128))
                cv2.imwrite(ROOT_PATH + '\\cropped_faces\\' + 'id' + str(id) + 'yaw' + str(aim[0]) + 'pitch' + str(aim[1]) + 'roll' + str(aim[2]) + '.png', cropped)

                p = get_proportions(face)

            if face[0]['confidence'] >= 0.7:
                X = X.append(get_features_from_dict(p))
                a.append(aim)

                cropped = cropped.reshape((cropped.shape[0], cropped.shape[1], 3))
                img_arr.append(cropped)


    return X.reset_index(drop=True), np.array(a), np.array(img_arr)

def get_angle(pts):

    leftEye = pts[0]
    rightEye = pts[1]

    h_d = leftEye[0] - rightEye[0]
    v_d =  leftEye[1] - rightEye[1]

    tan = v_d / h_d

    r_centre = ((leftEye[0] + rightEye[0]) // 2, (leftEye[1] + rightEye[1]) // 2)

    return math.degrees(math.atan(tan)), r_centre, np.sqrt((h_d ** 2) + (v_d ** 2))

def get_proportions(face):
    (x, y_, w, h) = face[0]['box']

    features = {}
    dst = {}
    proportions = {}

    for k in range(len(list(face[0]['keypoints'].values()))):
        for m in range(k+1, len(list(face[0]['keypoints'].values()))):
            d = dist.euclidean(list(face[0]['keypoints'].values())[k], list(face[0]['keypoints'].values())[m])
            d_name = 'from_' + str(list(face[0]['keypoints'].keys())[k]) + '_to_' + str(list(face[0]['keypoints'].keys())[m])
            dst.update({d_name : d})

    factor=1.0/sum(dst.values())
    for k in dst:
       dst[k] = dst[k]*factor

    distances = {
    'd1' : dist.euclidean(face[0]['keypoints'].get('left_eye'), face[0]['keypoints'].get('right_eye')),
    'd2' : dist.euclidean(face[0]['keypoints'].get('mouth_left'), face[0]['keypoints'].get('mouth_right')),
    'd3' : dist.euclidean(face[0]['keypoints'].get('left_eye'), face[0]['keypoints'].get('mouth_left')),
    'd4' : dist.euclidean(face[0]['keypoints'].get('right_eye'), face[0]['keypoints'].get('mouth_right'))
    }

    for i in range(len(distances.keys())):
        for j in range(i+1, len(distances.keys())):

            prop = distances.get(list(distances.keys())[i]) / distances.get(list(distances.keys())[j]) if distances.get(list(distances.keys())[j]) != 0 else 0
            prop_name = str(list(distances.keys())[i]) + '_to_' + str(list(distances.keys())[j])
            proportions.update({prop_name : prop})

    factor=1.0/sum(proportions.values())
    for k in proportions:
       proportions[k] = proportions[k]*factor

    nose_position_x, nose_position_y = face[0]['keypoints'].get('nose')

    nose_position = (nose_position_x - (x + w // 2)) / w

    features.update({'nose_pos' : nose_position})
    features.update(dst)
    features.update(proportions)

    return features

def get_features_from_dict(d : dict):

    return pd.DataFrame(d, index=[0])

if __name__ == "__main__":

    camera, detector = init()

    filename = 'final_model1.h5'

    training = False

    if training:

        X_init, y_init = image_reading(0)

        x_val_init, y_val_init = image_reading(1)

        X, y, img = crop_save_create(X_init, y_init)

        x_val, y_val, img_val = crop_save_create(x_val_init, y_val_init)

        model_cnn = create_cnn()
        model_mlp = create_mlp(X)
        combined_nn = mixed_nn(model_mlp, model_cnn)

        training_process, trained_model = compile_and_train_combined(combined_nn, X, img, y)

        evaluate_model(trained_model, x_val, y_val, img_val, 10)

        trained_model.save(os.path.join(ROOT_PATH, filename))

        print('\nNeural Network is trained')
        print('Program exit...')

        exit()

    loaded_model = load_model(os.path.join(ROOT_PATH, filename), compile=False)

    while camera.isOpened():
        slide = predict_and_show(camera, loaded_model, detector)

        cv2.imshow('Yaw Detection', slide)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()

    cv2.destroyAllWindows()

