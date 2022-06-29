#import mysql-connector-python
import pandas as pd
import pymysql
import os 
from flask import Flask, jsonify, g,abort, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy import create_engine

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] ='mysql+mysqlconnector://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] =False

#db = SQLAlchemy(app)

#db.create_all()

engine = create_engine("mysql://b07b4484224a54:edf76401@us-cdbr-east-06.cleardb.net/heroku_daac59f6173f49a")

#table_df = pd.read_sql_table(
#    curv_dec,
#    con=engine
#)
#print(table_df)
result = engine.execute('SELECT * FROM '+dbname)
rows = result.fetchall()
print(rows)        
result.close()
