import pandas as pd
import pymysql
from sqlalchemy import create_engine

df = pd.read_csv("curv_dec.csv")
engine = create_engine("mysql+pymysql://esilva:Cr1st0_R3y@localhost/MZI_SCF_fatt")
df.to_sql('curv_dec', engine, index=False)

table_df = pd.read_sql_table(
    'curv_dec',
    index_col='Wavelength',
    con=engine
)
print(table_df)

