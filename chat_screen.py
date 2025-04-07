import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import time

from db_connection import  getSchema

os.environ["GOOGLE_API_KEY"] = st.secrets["gemini_key"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.6,
    max_tokens=200,
    timeout=None,
    max_retries=2,
)

def getQueryFromLLM(question, schema):
    template = """Below is the schema of a MySQL database. Please carefully review the table and column names, paying attention to case sensitivity.

    {schema}

    Please respond **ONLY** with the exact SQL query required to answer the question. Do not include any additional explanations, comments, or text. The query should be valid and complete. Do **not** use markdown or code block formatting like 'sql' or '```'. Just provide the SQL query with no extra characters or formatting.

    for example:
    question: how many albums we have in database
    SQL query: SELECT COUNT(*) FROM album
    question: how many customers are from Brazil in the database ?
    SQL query: SELECT COUNT(*) FROM customer WHERE country=Brazil

    your turn :
    question: {question}
    SQL query :
    please only provide the SQL query and nothing else

    SQL query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm
    response = chain.invoke({"question": question,
                        "schema": schema})
    
    return response.content


def getResponseFromQuery(question, query, result, schema):
    template2 = """
        Below is the schema of a MySQL database. Please carefully read the schema and take note of the table and column names. Also, refer to the conversation if available.

        Please note: **Do not** provide responses that could allow the user to modify or delete any data, especially any sensitive information (e.g., passwords, user personal details, or any type of modification like DELETE, UPDATE, etc.). 
        Only allow queries that fetch data for reading purposes, and always ensure that no actions can lead to unintended changes to the database.

        {schema}

        Here are some example questions and responses for you:
        ---
        **Example 1:**
        question: How many albums do we have in the database?
        SQL query: SELECT COUNT(*) FROM album;
        Result: [(34,)]
        Response: There are 34 albums in the database.

        **Example 2:**
        question: How many users are in the database?
        SQL query: SELECT COUNT(*) FROM customer;
        Result: [(59,)]
        Response: There are 59 users in the database.

        **Example 3:**
        question: How many users from India do we have in the database?
        SQL query: SELECT COUNT(*) FROM customer WHERE country = 'India';
        Result: [(4,)]
        Response: There are 4 users from India in the database.

        ---
        **Instructions for your response:**
        1. Respond only with natural language that describes the result of the SQL query.
        2. Ensure that the SQL query is a read-only operation (e.g., SELECT statements).
        3. Avoid any SQL query or response that suggests modifications to data, such as DELETE, UPDATE, or SELECT queries that include passwords or sensitive details.
        4. If the question could potentially involve sensitive data, suggest a safer way of querying without compromising security or privacy.

        **Your turn:**
        question: {question}
        SQL query: {query}
        Result: {result}
        Response:
        """

    prompt2 = ChatPromptTemplate.from_template(template2)
    parser = StrOutputParser()
    chain2 = prompt2 | llm | parser

    # response = chain2.invoke({
    #     "question": question,
    #     "schema": schema,
    #     "query": query,
    #     "result": result
    # })
    
    # response_content = ""
    return chain2.stream({
        "question": question,
        "schema": schema,
        "query": query,
        "result": result
    })
        # yield chunk
        # response_content += chunk
        # print(response_content)
    # return response_content
    # return response.content

def chat_component(db):
    the_input = st.empty()
    chat_container = st.container()
    
    if "chats" not in st.session_state:
        st.session_state.chats = []

    if question := the_input.chat_input('Chat with your MySQL database'):
        st.session_state.chats.append({
            "role": "user",
            "content": question
        })
        # with st.chat_message("user"):
        #     st.markdown(question)
        if not db:
            st.error('Please connect to the database first.')
        else:
            
            schema = getSchema(db)    
            query = getQueryFromLLM(question, schema)
            print(query)  
            result = db.run(query)
            print(result)

            with st.spinner("Waiting..."):
                response_stream = getResponseFromQuery(question, query, result, schema)
    
            with st.chat_message("assistant"):
                full_response = st.write_stream(response_stream)
            # full_response = ""
            # for word in response_stream:
            #     full_response += word

            st.session_state.chats.append({
                "role": "assistant",
                "content": full_response
            })

    with chat_container:
        for chat in st.session_state.chats:
            # print(chat)
            if chat["role"] == "user":
                st.chat_message(chat["role"]).markdown(chat["content"])
            # else:
            #     st.chat_message(chat["role"]).write_stream(chat["content"])

        
  

    

