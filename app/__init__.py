from flask import Flask, jsonify
from flask_marshmallow import Marshmallow
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate


app = Flask(__name__)
app.config.from_object('config')
db = SQLAlchemy(app)
migrate = Migrate(app, db)
ma = Marshmallow(app)

from .models import user, caracteristica
db.create_all()
from .views import users, helper
from .routes import routes