#import mysql-connector-python
import pandas as pd
import pymysql
import os 
from flask import Flask, jsonify, g,abort, render_template, request, redirect, url_for, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from sqlalchemy import create_engine




#db = SQLAlchemy(app)

#db.create_all()

engine = create_engine("mysql+pymysql://esilva:Cr1st0_R3y@localhost/MZI_SCF_fatt")

table_df = pd.read_sql_table("Tx_curv_dec", con=engine
)
print(table_df)
#result = engine.execute('SELECT * FROM '+dbname)
#rows = result.fetchall()
#print(rows)        
#result.close()
