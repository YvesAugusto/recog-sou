import configparser
import os
import random
import string

basedir = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser(interpolation=None)
config.read(f'{basedir}/config.ini')
config.read(f'{basedir}/config.ini')
user = config['BANCO']['user']
passwd = config['BANCO']['passwd']
database = config['BANCO']['db']
host = config['BANCO']['host']
port = int(config['BANCO']['port'])
gen = string.ascii_letters + string.digits + string.ascii_uppercase
key = ''.join(random.choice(gen) for i in range(12))

SQLALCHEMY_DATABASE_URI = f'postgresql://{user}:{passwd}@{host}:{port}/{database}'
# POSTGRES = {
#     'user': 'romero',
#     'pw': '4uBZGCR4mzUKUBbwe7e9',
#     'db': 'face_recognition',
#     'host': '192.169.178.70',
#     'port': '5432',
# }
# SQLALCHEMY_DATABASE_URI = 'postgresql://%(user)s:\%(pw)s@%(host)s:%(port)s/%(db)s' % POSTGRES
SQLALCHEMY_TRACK_MODIFICATIONS = False
SECRET_KEY = key
DEBUG = True