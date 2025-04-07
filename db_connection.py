from langchain_community.utilities import SQLDatabase
import streamlit as st

def connectDB(db_config):
    try:
        if not db_config:
            mysql_uri = st.secrets["default_db_uri"]
        else:
            host = db_config["host"]
            port = db_config["port"]
            user = db_config["user"]
            password = db_config["password"]
            database = db_config["database"]

            mysql_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"

        db = SQLDatabase.from_uri(mysql_uri)
        return db

    except Exception as e:
        return None
    
def getSchema(db):
    if db:
        return db.get_table_info()


