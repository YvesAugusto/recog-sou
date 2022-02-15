import json

from flask import request, jsonify
from ..models.user import User, users_schema
from app import db

def delete_user(data):
    user = User.query.filter_by(cpf=data['cpf']).all()

    if len(user) > 0:
        try:
            db.session.delete(user[-1])
            db.session.commit()
        except:
            return jsonify(
                {"message": "unable to delete", "data": {}}
            ), 401

        return jsonify(
            {"message": "sucessfully deleted", "data": data}
        ), 201
    return jsonify(
        {"message": "user wit cpf {} does not exist".format(data['cpf']), "data": {}}
    ), 401



    return jsonify(
                {"message": "user does not exist", "data": {}}
            ), 401

def enroll(data):
    user = User.query.filter_by(cpf=data['cpf']).all()
    if len(user) == 0:
        try:
            user = User(
                name=data['name'],
                cpf=data['cpf'],
                active=True
            )
            db.session.add(user)
            db.session.commit()

        except:
            return jsonify(
                {"message": "enroll failed", "data": {}}
            ), 401
        return jsonify(
            {"message": "successfully enrolled", "data": data}
        ), 201

    return jsonify(
        {"message": "user with cpf {} already exists".format(data['cpf']), "data": {}}
    ), 401


