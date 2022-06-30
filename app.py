#import mysql-connector-python
import pandas as pd
import pymysql
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os 
from os.path import join, dirname, realpath
from flask import Flask, jsonify, g,abort, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] =False



@app.route("/")
def upload_file():
    # renderizamos la plantilla "index.html"
    return render_template('index.html')

@app.route("/load_database", methods=['POST', 'GET'])
def loadDB():
    basedir = os.path.abspath(os.path.dirname(__file__))
    engine = create_engine("mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a")
    basedir = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    os.chdir(filepath)

    """
    #procedimiento para crear las tablas
    df1 = pd.read_csv("curv_dec.csv")   
    df2 = pd.read_csv("curv_inc.csv")
    df3 = pd.read_csv("temp_dec.csv")
    df4 = pd.read_csv("temp_inc.csv")
    df1.to_sql('Tx_curv_dec2', engine, index=False)
    df2.to_sql('Tx_curv_inc2', engine, index=False)
    df3.to_sql('Tx_temp_dec2', engine, index=False)
    df4.to_sql('Tx_temp_inc2', engine, index=False)
    """
    table_df1 = pd.read_sql_table(
    'Tx_curv_dec',
    index_col='Wavelength',
    con=engine)
    #Resultado de la tabla
    result = table_df1.to_json(orient="columns")
    #borrar tabalas
    drop_table('Tx_curv_dec2', engine)
    drop_table('Tx_curv_inc2', engine)
    drop_table('Tx_temp_dec2', engine)
    drop_table('Tx_temp_inc2', engine)
    return result


def drop_table(table_name, engine):
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        Base.metadata.drop_all(engine, [table], checkfirst=True)
    return

  
#https://stackoverflow.com/questions/35918605/how-to-delete-a-table-in-sqlalchemy

if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)
