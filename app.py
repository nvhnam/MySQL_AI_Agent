import streamlit as st

from chat_screen import chat_component
from custom_components import notify_message
from db_connection import connectDB

st.set_page_config(
    page_title="MySQL AI Chatter",
    page_icon=":microscope:"
)

with st.sidebar:
    st.header("Database Connection :signal_strength:")
    chosen_db = st.radio(
        "Choose your DB",
        ["NutriGuide :shopping_trolley:", "Custom"],
        index=None,
        horizontal=True,
    )
    if chosen_db == "Custom":
        host = st.text_input("Host", value="localhost")
        port = st.text_input("Port", value="3306")
        username = st.text_input("Username", value="root")
        password = st.text_input("Password", type="password")
        database = st.text_input("Database", value="rag_test")

        if st.button("Connect") and not st.session_state.get("db"):
            st.session_state.db_config = {
                "host": host,
                "port": int(port),
                "user": username,
                "password": password,
                "database": database
            }
            with st.spinner("Connecting..."):
                db = connectDB(db_config=st.session_state.db_config)
                if db:
                    notify_message("✅ Successfully connected to the database!", isValid=True)
                    st.session_state.db = db
                else:
                    notify_message("🚫 Could not connect to the database.", isValid=False)

    elif chosen_db == "NutriGuide :shopping_trolley:":
        if "db" not in st.session_state:
            with st.spinner("Connecting..."):
                db = connectDB(db_config=None)
                if db:
                    notify_message("✅ Successfully connected to the database!", isValid=True)
                    st.session_state.db = db  
                else:
                    notify_message("🚫 Could not connect to the database.", isValid=False)
    st.divider()
    st.markdown('''Made by [@nvhnam](https://github.com/nvhnam)''')

    
with st.container():
    st.title("Welcome to _:blue[MySQL AI Assistant]_ :bar_chart:")
    st.divider()

    st.markdown('''
No more writing SQL manually!  
:blue[**MySQL AI Assistant**] lets you connect to your own MySQL database and query it using natural language.  
Powered by an :blue[**LLM with RAG**], it translates your questions into SQL and returns accurate results.

Built with **Streamlit** and **LangChain** to simplify data access for everyone.
''')
    db = st.session_state.get("db")
    chat_component(db)
    
