import json
from flask import jsonify, request
from ..views.tools import *

MIN_AREA_PROPORTION = 0.05
THRESHOLD = 0.6
TIMEOUT_DOWNLOAD_URLS = 5
MAXIMUM_NUMBER = 7
MINIMUM_NUMBER = 3
FACE_CROP_FUNCTION = crop_face_roi_distant

def handler(signum, frame):
    print("Done!")
    raise Exception("[INFO] Timeout exceeded...")

def read_from_disk(URL):
    return cv2.imread(URL)

def read_from_url(URL):
    return imutils.url_to_image(URL)

def format_data(features):
    new_features = []
    for idf, f in enumerate(features):
        new_f = f[2:-2].split(",")
        new_f = np.array(list(map(float, new_f)))
        new_features.append(new_f)
    return new_features

def delete_feature(data):
    user = User.query.filter_by(cpf=data['cpf']).all()
    if len(user) < 1:
        return jsonify(
            {"message": "user wit cpf {} does not exist".format(data['cpf']), "data": {}}
        ), 401
    features = Caracteristica.query.filter_by(user_cpf=data['cpf']).all()
    if len(features) < 0:
        return jsonify(
            {"message": "user wit cpf {} has no features".format(data['cpf']), "data": {}}
        ), 401
    for idf, feature in enumerate(features):
        db.session.delete(features[idf])
    try:
        db.session.commit()
    except:
        return jsonify(
            {"message": "unable to delete", "data": {}}
        ), 401
    return jsonify(
        {"message": "deleted {} features from cpf {}".format(len(features), data['cpf']), "data": {}}
    ), 201

def enroll_feature(data):
    URL_LIST = data['urls']
    CPF = data['cpf']
    try:
        actual_features = Caracteristica.query.filter_by(user_cpf=CPF).all()
    except:
        return jsonify(
            {"message": "unable to select from database",
             "data": {}}
        ), 401

    if len(actual_features) >= MAXIMUM_NUMBER:
        return jsonify(
            {"message": "this user has reached the maximum number of features",
             "data": {}}
        ), 201

    actual_len = len(actual_features) + len(URL_LIST)
    if actual_len > MAXIMUM_NUMBER:
        return jsonify(
            {
                "message": "user cannot have more than {} images enrolled".format(MAXIMUM_NUMBER),
                 "data": {
                     "number_of_images_enrolled": len(actual_features),
                     "you_tried_to_insert": len(URL_LIST)}
             }
        ), 201

    try:
        frames = [read_from_url(URL) for URL in URL_LIST]
    except Exception as exc:
        return jsonify(
            {"message": 'could not download images', "data": {}}
        ), 401
    j = 0
    k = -1
    for frame in frames:
        try:
            k+=1
            face, proportion = FACE_CROP_FUNCTION(frame, k)
        except Exception as exp:
            print(exp)
            continue
        j+=1
        embedding = extract_features(face)
        feature = Caracteristica(
                user_cpf=CPF,
                vetor_entrada=embedding.tolist(),
                root_model='inception-resnet-v1'
        )
        db.session.add(feature)
    try:
        db.session.commit()
    except Exception as excep:
        print(excep)
        return jsonify(
            {"message": "enroll failed", "data": {}}
        ), 401

    actual_len = len(actual_features) + j
    response = {"cpf": CPF, "actual_number_of_images": actual_len}
    if actual_len == MAXIMUM_NUMBER:
        try:
            fit_svm_to_user(cpf=data['cpf'])
            response["train"] = "sucess"
        except Exception as exp:
            print(exp)
            response["train"] = "fail"

    return jsonify(
        {
            "message": "{} image(s) successfully enrolled".format(j),
            "data": json.dumps(response)
         }
    ), 201

def recog(data):
    URL = data['urls'][0]
    CPF = data['cpf']
    actual_features = Caracteristica.query.filter_by(user_cpf=CPF).all()

    if len(actual_features) < MINIMUM_NUMBER:
        return jsonify(
            {"message": 'user has less than {} features enrolled'.format(MINIMUM_NUMBER), "data": {}}
        ), 401

    try:
        svm = pickle.loads(User.query.filter_by(cpf=data['cpf']).first().classifier)
    except:
        return jsonify(
            {"message": 'unable to load classifier', "data": {}}
        ), 401

    try:
        face = read_from_url(URL)
        # H, W = face.shape[:2]
        #         # known_face_locations = [(0, 0, W, H)]
        #         # encoding = face_recognition.face_encodings(face, known_face_locations=known_face_locations)
        face, proportion = FACE_CROP_FUNCTION(face, 0)
    except Exception as exc:
        return jsonify(
            {"message": 'unable to process image', "data": {}}
        ), 401

    actual_features = format_data([f.vetor_entrada for f in actual_features])
    feature = extract_features(face)
    random_vec = np.arange(len(actual_features))
    np.random.shuffle(random_vec)
    index = random_vec[0]
    del actual_features[index]
    distances = [L2(feature - f) for f in actual_features]
    result = svm.predict_proba([distances])[0][1]
    return jsonify(
        {"message": "verification applied",
         "data": {"label": (result)}}
    ), 201

    # return jsonify(
    #     {"message": "face ROI too small, i.e, less than {}% of image area".format(100*MIN_AREA_PROPORTION),
    #      "data": {}}
    # ), 201

def get_users():
    users = User.query.filter_by(active=True).all()
    data = []
    for user in users:
        features = len(Caracteristica.query.filter_by(user_cpf=user.cpf).all())
        data.append(
            {'cpf': user.cpf,
             'number_of_features': features}
        )

    return jsonify({'message': 'successfully fetched', 'data': data}), 201

def get_user_by_cpf(cpf):
    if cpf:
        users = User.query.filter_by(cpf=cpf).all()
    else:
        users = User.query.all()
    if users:
        data = []
        for user in users:
            features = len(Caracteristica.query.filter_by(user_cpf=user.cpf).all())
            data.append(
                {'cpf': user.cpf,
                 'number_of_features': features}
            )
        return jsonify({'message': 'successfully fetched', 'data': data})

    return jsonify({'message': 'nothing found', 'data': {}})


