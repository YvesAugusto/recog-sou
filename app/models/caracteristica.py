import datetime
from app import db, ma

class Caracteristica(db.Model):
    __tablename__ = 'caracteristica'
    __table_args__ = {'extend_existing': True}
    id = db.Column(db.Integer, primary_key=True)
    user_cpf = db.Column(db.String(11), db.ForeignKey('users.cpf'), nullable=False)
    root_model = db.Column(db.String(150), default='inception-resnet-v1')
    vetor_entrada = db.Column(db.Text())

    def __init__(self, user_cpf, vetor_entrada, root_model):
        self.user_cpf = user_cpf
        self.vetor_entrada = vetor_entrada
        self.root_model = root_model

# class CaracteristicaSchema(ma.Schema):
#     class Meta:
#         fields = ('id', 'user_cpf', 'vetor_entrada')
#
# feature_schema = CaracteristicaSchema()
# features_schema = CaracteristicaSchema(many=True)