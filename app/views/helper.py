import json
from flask import request, jsonify
import os, jwt
from datetime import datetime
import datetime as dt
from functools import wraps
from app import app, db

USERNAME = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')

def auth():

    auth = request.authorization
    if not auth or not auth.username or not auth.password:
        return jsonify(
            {"message": "verification failed",
             "WWW-Autenticate": "Login required"}
        ), 401

    if auth.username == USERNAME and auth.password == PASSWORD:
        now = datetime.now()
        token = jwt.encode(
            {'username': USERNAME, 'exp': now + dt.timedelta(hours=12)},
            app.config['SECRET_KEY']
        )

        return jsonify(
            {
                "message": "Success",
                "token": token.decode('UTF-8'),
                "exp": now + dt.timedelta(hours=12)
            }
        )

    return jsonify(
        {
            "message": "verification failed",
            "WWW-Autenticate": "Login required"
        }
    )

def validate_token(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        token = request.args.get('token')
        if not token:
            return jsonify(
                {"message": "missing token", "data": {}}, 401
            )
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
        except:
            return jsonify(
                {"message": "invalid token or expired", "data": {}}, 401
            )
        return func(*args, **kwargs)

    return decorator

def validate_cpf(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            data = json.loads(request.data)
        except:
            return jsonify(
                {"message": "invalid data", "data": {}}, 401
            )
        cpf = data['cpf']
        if (len(cpf) == 11) and (sum(c.isdigit() for c in cpf) == 11):
            return func(data)
        else:
            return jsonify(
                {"message": "invalid cpf", "data": {}}, 401
            )

    return decorator

def validate_url(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        try:
            data = json.loads(request.data)
        except:
            return jsonify(
                {"message": "invalid data", "data": {}}, 401
            )
        URL_LIST = data['urls']
        if len(URL_LIST) > 0:
            return func(data)
        else:
            return jsonify(
                {"message": "missing urls", "data": {}}, 401
            )
    return decorator