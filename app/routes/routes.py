from app import app
from ..views import users, caracteristicas, helper

@app.route("/auth", methods=['POST'])
def authenticate():
    return helper.auth()

@app.route('/viewUsers', methods=['GET'])
@helper.validate_token
def get_users():
    return caracteristicas.get_users()

@app.route('/viewUsers/<id>', methods=['GET'])
@helper.validate_token
def get_user(id):
    return caracteristicas.get_user_by_cpf(id)

@app.route('/deleteUser', methods=['POST'])
@helper.validate_token
@helper.validate_cpf
def delete_user(data):
    return users.delete_user(data)

@app.route('/deleteFeature', methods=['POST'])
@helper.validate_token
@helper.validate_cpf
def delete_feature(data):
    return caracteristicas.delete_feature(data)

@app.route('/enroll', methods=['POST'])
@helper.validate_token
@helper.validate_cpf
def enroll(data):
    return users.enroll(data)

@app.route('/feature', methods=['POST'])
@helper.validate_token
@helper.validate_url
@helper.validate_cpf
def feature(data):
    return caracteristicas.enroll_feature(data)

@app.route('/recog', methods=['POST'])
@helper.validate_token
@helper.validate_url
@helper.validate_cpf
def recog(data):
    return caracteristicas.recog(data)