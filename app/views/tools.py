import os
import numpy as np
import cv2, dlib
import imutils
from imutils import face_utils
import tensorflow as tf
from ..models.user import User
from ..models.caracteristica import Caracteristica
from numpy.linalg import norm as L2
from copy import deepcopy
import pickle
from sklearn.svm import SVC
from app import db
from collections import OrderedDict


FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

desired_parts = [
    'mouth', 'left_eyebrow', 'right_eyebrow'
]

ABSPATH = os.getenv('PYTHONPATH') + "app"
DEPLOY_TXT = os.path.join(ABSPATH, 'files', 'deploy.prototxt.txt')
EMBEDDER_FILE = os.path.join(ABSPATH, 'files', 'inception-resnet-v1-facenet.h5')
LANDMARK_FILE = os.path.join(ABSPATH, 'files', 'shape_predictor_68_face_landmarks.dat')
FACE_DETECTION = os.path.join(ABSPATH, 'files', 'res10_300x300_ssd_iter_140000.caffemodel')
SAVE_CROPS = os.path.join(ABSPATH, 'saved_crops') + "/"

embedder = tf.keras.models.load_model(EMBEDDER_FILE)
landMarksPredictor = dlib.shape_predictor(LANDMARK_FILE)
faceAligner = imutils.face_utils.FaceAligner(landMarksPredictor, desiredFaceWidth=150, desiredLeftEye=(0.3, 0.3))
detector = dlib.get_frontal_face_detector()
detectorCNN = cv2.dnn.readNetFromCaffe(DEPLOY_TXT, FACE_DETECTION)
i = 0
def crop_face_ellipse(image):
    (h, w) = image.shape[:2]
    w_adjusted = w//2
    if w == h:
        w_adjusted = w//2 - int(w * 0.05)
    center = (w // 2, h // 2)
    mask = np.zeros((h, w), np.uint8)
    circle_img = cv2.ellipse(mask, center, (w_adjusted, h // 2), 0, 0, 360, (255, 255, 255), thickness=-1)
    masked_data = cv2.bitwise_and(image, image, mask=circle_img)
    return masked_data

def preproces(crop, method='norm'):
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop / 255
    # if method == 'norm':
    #     for i in range(3):
    #         crop[:,:,i] = (crop[:,:,i] - np.mean(crop[:,:,i]))/np.std(crop[:,:,i])
    #     return crop


def detection(frame):
    global detectorCNN
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    detectorCNN.setInput(blob)
    detections = detectorCNN.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.7:
            return False, frame

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        return True, (startX, startY, endX, endY)

def show_bbox(frame, bbox):
    (startX, startY, endX, endY) = bbox
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
    cv2.imwrite('test.jpeg', frame)

def crop_face_roi(frame, k):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = detector(gray, 1)
    for (i, rect) in enumerate(rect):
        shape = landMarksPredictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        desired_shape = []
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name in desired_parts:
                desired_shape.append(shape[i:j])
        desired_shape = np.concatenate((desired_shape))
        desired_shape = np.concatenate((desired_shape, [shape[5], shape[11]]))
        (x, y, w, h) = cv2.boundingRect(np.array([desired_shape]))
        H, W = frame.shape[:2]
        delta_y = int(0.35*(shape[39][1] - shape[20][1]))
        x1, y1, x2, y2 = x, y-delta_y, x + w, y + h
        masked = frame[y1:y2, x1:x2]
        cv2.imwrite(SAVE_CROPS+"test_{}.jpeg".format(k), masked)
        return masked, ((w * h) / (H * W))

def crop_face_roi_distant(frame, k):
    detected, box = detection(frame)
    # show_bbox(frame, box)
    if detected:
        (startX, startY, endX, endY) = box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(startX, startY, endX, endY)
        alignedFace = faceAligner.align(frame, gray, rect)
        H,W = alignedFace.shape[:2]
        alignedRect = dlib.rectangle(0,0,W,H)
        alignedGray = cv2.cvtColor(alignedFace, cv2.COLOR_BGR2GRAY)
        shape = landMarksPredictor(alignedGray, alignedRect)
        shape = face_utils.shape_to_np(shape)
        desired_shape = []
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            if name in desired_parts:
                desired_shape.append(shape[i:j])
        desired_shape = np.concatenate((desired_shape))
        desired_shape = np.concatenate((desired_shape, [shape[5], shape[11]]))
        (x, y, w, h) = cv2.boundingRect(np.array([desired_shape]))
        H, W = frame.shape[:2]
        delta_y = int(0.35 * np.abs(shape[39][1] - shape[20][1]))
        x1, y1, x2, y2 = x, y - delta_y, x + w, y + h
        masked = alignedFace[y1:y2, x1:x2]
        cv2.imwrite(SAVE_CROPS + "test_{}.jpeg".format(k), masked)
        proportion = ((w * h) / (H * W))
        print(proportion)
        return masked, proportion

    return [], 0.0


def align_face(frame, j):
    H, W = frame.shape[:2]
    # flag, box = detection(frame)
    # (startX, startY, endX, endY) = box
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = dlib.rectangle(0, 0, W, H)
    # cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 3)
    # alignedFace = faceAligner.align(frame, gray, rect)
    elipsedFace = crop_face_ellipse(alignedFace)
    elipsedFace = cv2.resize(elipsedFace, (150, 150), cv2.INTER_AREA)
    cv2.imwrite(SAVE_CROPS+'test_{}.jpg'.format(j), elipsedFace)

    return elipsedFace

def extract_features(crop):
    global embedder
    crop = cv2.resize(crop, (150, 150), cv2.INTER_AREA)
    crop = preproces(crop, method='norm')
    img = np.expand_dims(crop, axis=0)
    return np.array(embedder(img))

def format_data(features):
    new_features = []
    for idf, f in enumerate(features):
        new_f = f[2:-2].split(",")
        new_f = np.array(list(map(float, new_f)))
        new_features.append(new_f)
    return new_features

def compute_positives(user_features):
    positives = []
    for idu, feature in enumerate(user_features):
        anchor = user_features[idu]
        pos = deepcopy(user_features)
        del pos[idu]
        positives.append(
            [L2(anchor - p) for p in pos]
        )
    return np.array(positives)

def compute_negatives(user_features, non_user_features):
    random_vec = np.arange(len(non_user_features))
    np.random.shuffle(random_vec)
    random_vec = random_vec[:3 * len(user_features)]
    new_non_user_features = []
    for index in random_vec:
        new_non_user_features.append(non_user_features[index])
    non_user_features = format_data(new_non_user_features)
    negatives = []
    for idn, neg in enumerate(non_user_features):
        random_vec = np.arange(len(user_features))
        np.random.shuffle(random_vec)
        index = random_vec[0]
        vectors = deepcopy(user_features)
        del vectors[index]
        negatives.append(
            [L2(neg-vector) for vector in vectors]
        )
    return np.array(negatives)

def fit_svm_to_user(cpf):
    features = Caracteristica.query.all()
    user_features = []
    non_user_features = []
    for f in features:
        if f.user_cpf == cpf:
            user_features.append(f.vetor_entrada)
        else:
            non_user_features.append(f.vetor_entrada)
    user_features = format_data(user_features)
    score = 0.0
    while score < 1.0:
        positives = compute_positives(user_features)
        negatives = compute_negatives(user_features, non_user_features)
        X = np.concatenate((positives, negatives))
        Y = np.concatenate((
            np.ones(len(positives)),
            np.zeros(len(negatives))
        ))
        #gamma=0.0625 if use rbf kernel
        svm = SVC(C=0.25, gamma=0.0625, kernel='linear', probability=True)
        svm.fit(X, Y)
        score = svm.score(X, Y)
        print("Score: {}".format(score))

    user = User.query.filter_by(cpf=cpf).first()
    user.classifier = pickle.dumps(svm)
    db.session.add(user)
    db.session.commit()