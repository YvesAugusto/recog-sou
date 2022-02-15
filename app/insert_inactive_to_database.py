import os, time
import cv2 as cv
import numpy as np
from app import db
import string, random
import tensorflow as tf
from models.user import User
from models.caracteristica import Caracteristica

def preproces(crop, shape=(150, 150)):
    return crop/255

def extract_features(embedder, crop):
    crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
    crop = cv.resize(crop, (150, 150), cv.INTER_AREA)
    crop = preproces(crop, shape=(150, 150))
    img = np.expand_dims(crop, axis=0)
    return np.array(embedder(img))

def map_paths_from_username_folder_(filepaths_folder):
    files = os.listdir(filepaths_folder)
    user_maps = []
    for file in files:
        filepath = filepaths_folder + file
        user_maps.append(filepath)

    return user_maps

def map_paths_from_folder_containing_users_(directory):
    names = os.listdir(directory)
    maps = {}
    for name in names:
        filepaths_folder = directory + "/" + name + "/"
        user_maps = map_paths_from_username_folder_(filepaths_folder=filepaths_folder)
        maps.update({name: user_maps})

    return maps

def make_feature_maps(embedder, folder):
    maps = map_paths_from_folder_containing_users_(folder)
    feature_maps = {}
    i = 0
    tam = sum([len(values) for key, values in maps.items()])
    start = time.time()
    for name, files in maps.items():
        if name not in list(feature_maps.keys()):
            feature_maps.update({name: []})
        for idf, file in enumerate(files):
            feature_maps[name].append(
                extract_features(embedder, cv.imread(file))
            )
            i+=1
            print(f'Extracted features: {i}/{tam}')
    print(f'Tempo de extração total: {(time.time() - start)}s')

    return feature_maps

def id_generator(size=11, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def insert_to_database(embedder, folder):
    feature_maps = make_feature_maps(embedder, folder)
    i = 0
    for name, vectors in feature_maps.items():
        user = User(name=name, active=False,
                    cpf=id_generator(), classifier=None)
        db.session.add(user)
        db.session.commit()
        for vector in vectors:
            f = Caracteristica(user_cpf=user.cpf,
                               vetor_entrada=vector.tolist(),
                               root_model='inception-resnet-v1')
            db.session.add(f)
        i+=1
        print(f'Usuário inativo {name}, {i}/{len(list(feature_maps.keys()))}')
        db.session.commit()

if __name__ == '__main__':
    ABSPATH = os.getenv('PYTHONPATH') + "app"
    EMBEDDER_FILE = os.path.join(ABSPATH, 'files', 'inception-resnet-v1-facenet.h5')
    embedder = tf.keras.models.load_model(EMBEDDER_FILE)
    insert_to_database(embedder, '/media/yves/HD/inactive-users/')