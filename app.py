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

basedir = os.path.abspath(os.path.dirname(__file__))





app = Flask(__name__)
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] =False



@app.route("/")
def upload_file():
    # renderizamos la plantilla "index.html"
    return render_template('index.html')

@app.route("/load_database")
    engine = create_engine("mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a")
    basedir = os.path.abspath(os.path.dirname(__file__))
    filepath = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    os.chdir(filepath)
    df1 = pd.read_csv("curv_dec.csv")   
    df2 = pd.read_csv("curv_inc.csv")
    df3 = pd.read_csv("temp_dec.csv")
    df4 = pd.read_csv("temp_inc.csv")
    df1.to_sql('Tx_curv_dec', engine, index=False)
    df2.to_sql('Tx_curv_inc', engine, index=False)
    df3.to_sql('Tx_temp_dec', engine, index=False)
    df4.to_sql('Tx_temp_inc', engine, index=False)


table_df1 = pd.read_sql_table(
    'Tx_curv_dec',
    index_col='Wavelength',
    con=engine
    return "ya"

if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)

