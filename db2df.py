import pandas as pd
from sqlalchemy import create_engine

dbname = pandas_sample

engine = create_engine("mysql+pymysql://esilva:Cr1st0_R3y@localhost/"+dbname)
#table_df = pd.read_sql_table(
#    curv_dec,
#    con=engine
#)
#print(table_df)
result = engine.execute('SELECT * FROM '+dbname)
rows = result.fetchall()
print(rows)        
result.close()
