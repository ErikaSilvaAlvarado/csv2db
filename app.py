#import mysql-connector-python
import pandas as pd
import MyFunctions as fu
import pymysql
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import os 
from os.path import join, dirname, realpath
import json
import plotly
pd.options.plotting.backend = "plotly"

from flask import Flask, jsonify, g,abort, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy import create_engine
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

app = Flask(__name__)
# Carpeta de subida
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a'
#app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+pymysql://esilva:Cr1st0_R3y@localhost/MZI_SCF_fatt'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] =False



@app.route("/")
def upload_file():
    # renderizamos la plantilla "index.html"
    return render_template('index.html')

@app.route("/load_database", methods=['POST', 'GET'])
def loadDB():
    basedir = os.path.abspath(os.path.dirname(__file__))
    engine = create_engine("mysql+pymysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a")
    #engine = create_engine("mysql+pymysql://esilva:Cr1st0_R3y@localhost/MZI_SCF_fatt")
    #metadata = MetaData()
    #metadata.reflect(engine)
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
    engine.execute("DROP table IF EXISTS Tx_curv_inc2")
    engine.execute("DROP table IF EXISTS Tx_curv_dec2")
    engine.execute("DROP table IF EXISTS Tx_temp_inc2")
    engine.execute("DROP table IF EXISTS Tx_temp_dec2")
    inspector = inspect(engine)
    table_names = inspector.get_table_names()
    for table_name in table_names:
        print(f"Table:{table_name}")
    """    
    #para borar en localhost pero en cleardb no creo haya funcionado
    engine=drop_table('Tx_temp_inc2', engine)
    engine=drop_table('Tx_temp_dec2', engine)
    engine=drop_table('Tx_curv_inc2', engine)
    engine=drop_table('Tx_curv_dec2', engine)
    
    #print(table_df1)
    """
    """
    #esto vacía a tabla, pero no la borra
    with engine.connect() as conn, conn.begin():
        df = pd.read_sql('select * from Tx_curv_dec2 limit 1', con=conn)
        print (df.head())
        df.to_sql('Tx_curv_dec2', con=conn, schema='MZI_SCF_fatt', if_exists='replace')
        conn.close()
    """
    table_df1 = pd.read_sql_table('Tx_curv_dec',con=engine)
    fig = fu.PlotParamIntLgd(table_df1,True)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('plotDB.html', graphJSON=graphJSON)

def drop_table(table_name, engine):
    Base = declarative_base()
    metadata = MetaData()
    metadata.reflect(engine)
    metadata.reflect(bind=engine)
    table = metadata.tables[table_name]
    if table is not None:
        Base.metadata.drop_all(engine, [table], checkfirst=True)
    return engine

  
#https://stackoverflow.com/questions/35918605/how-to-delete-a-table-in-sqlalchemy

if __name__ == '__main__':
    # Iniciamos la aplicación
    app.run(debug=True)
