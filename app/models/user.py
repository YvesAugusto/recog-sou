import datetime
from app import db, ma

class User(db.Model):
    __tablename__ = 'users'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(60), nullable=False)
    cpf = db.Column(db.String(11), unique=True, nullable=False)
    active = db.Column(db.Boolean, default=False)
    classifier = db.Column(db.PickleType(), nullable=True)

    def __init__(self, name, cpf, active, classifier=None):
        self.name = name
        self.active = active
        self.cpf = cpf
        self.classifier = classifier

class UserSchema(ma.Schema):
    class Meta:
        fields = ('id', 'name', 'active', 'classifier')

user_schema = UserSchema()
users_schema = UserSchema(many=True)