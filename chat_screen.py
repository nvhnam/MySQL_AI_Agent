import asyncio
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from transformers import AutoTokenizer

from db_connection import  getSchema
from llm_agent import llm_agentic

os.environ["GOOGLE_API_KEY"] = st.secrets["gemini_key"]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.7,
    max_tokens=1500,
    timeout=None,
    max_retries=2,
)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

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

full_response = ""

async def getResponseFromQuery(question, query, result, schema, chat_history_str):
    template2 = """
    You are an expert MySQL assistant helping users query information from a database. Below is the schema of the MySQL database. Carefully review the table and column names before answering. Additionally, always **refer to the full conversation history** to maintain context and resolve references accordingly (e.g., "it", "they", "that table", etc.).

    **IMPORTANT NOTES:**
    - DO NOT return data about all products unless the user asks for it. If the user says "it", find the most recent item mentioned by them in the conversation history and use it as the context for the current question.
    - NEVER return queries that can alter data (e.g., DELETE, UPDATE, INSERT).
    - Do NOT include or reveal sensitive information such as passwords, emails, or personal details (e.g., user id).
    - Your job is to assist with read-only SQL queries (SELECT) and explain the results in a user-friendly, natural language response.

    ---

    **Database Schema:**
    {schema}

    ---

    **Conversation History:**
    The following is the prior conversation between the user and you (assistant). Use this to understand what "it", "they", or "that" refers to, and ensure continuity in the conversation.

    {chat_history_str}

    ---

    **Examples:**

    **Example 1:**
    User: How many albums do we have in the database?
    SQL query: SELECT COUNT(*) FROM album;
    Result: [(34,)]
    Response: There are 34 albums in the database.

    **Example 2:**
    User: How many users are in the database?
    SQL query: SELECT COUNT(*) FROM customer;
    Result: [(59,)]
    Response: There are 59 users in the database.

    **Example 3:**
    User: How many users from India do we have in the database?
    SQL query: SELECT COUNT(*) FROM customer WHERE country = 'India';
    Result: [(4,)]
    Response: There are 4 users from India in the database.

    ---

    **Guidelines:**
    1. Make sure the SQL query is safe and **read-only** (SELECT only).
    2. Use the chat history to resolve vague or ambiguous references.
    3. If you cannot determine exactly what the user means, ask for clarification rather than guessing.
    4. Always provide a natural language summary of the result, after the query.
    5. When in doubt about privacy, omit sensitive data or suggest a more general/safe query.

    ---

    **Current Input:**
    User’s question: {question}
    Generated SQL query: {query}
    Query result: {result}

    ---

    **Final Response:**
    Please provide a clear, natural language answer based on the result above, using only the result related to what the user is asking **based on context from the chat**. If unsure, politely ask the user to clarify..
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
    global full_response
    # response_content = ""
    for chunk in chain2.stream({
        "question": question,
        "schema": schema,
        "query": query,
        "result": result,
        "chat_history_str": chat_history_str
    }):
        # response = chunk.choices[0].delta.content or ""
        # if response:
        full_response += chunk 
        yield chunk
    #     print(chunk)
    #     yield str(chunk)
        # response_content += chunk
        # print(response_content)
    # return response_content
    # return response.content

def clear_chat_history():
    st.session_state.chats = [{"role": "assistant", "content": "Hi. I'm your MySQL Assistant. Ask me anything about your database!"}]
    # del st.session_state.chats

def chat_component(db):
    # the_input = st.empty()
    # chat_container = st.container()

    if "chats" not in st.session_state.keys():
        st.session_state.chats = [{"role": "assistant", "content": "Hi. I'm your MySQL Assistant. Ask me anything about your database!"}]

    for chat in st.session_state.chats:
        st.chat_message(chat["role"]).write(chat["content"])

    if question := st.chat_input('Chat with your MySQL database'):
        st.session_state.chats.append({
            "role": "user",
            "content": question
        })
        with st.chat_message("user"):
            st.write(question)
        if not db:
            st.error('Please connect to the database first.')
        elif db and question:
            # schema = getSchema(db)    
            # query = getQueryFromLLM(question, schema)
            # print(query)  
            # try:
            #     result = db.run(query)
            # except Exception as e:
            #     result = e
            # print(result)
            # chat_history = []
            # for dict_message in st.session_state.chats:
            #     if dict_message["role"] == "user":
            #         chat_history.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
            #     else:
            #         chat_history.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
            
            # chat_history.append("<|im_start|>assistant")
            # chat_history.append("")
            # chat_history_str = "\n".join(chat_history)
            # # print(chat_history_str)

            # if get_num_tokens(chat_history_str) >= 1000:
            #     st.error("Conversation length too long. Please keep it under 1000 tokens.")
            #     st.button('Clear chat history', on_click=clear_chat_history(), key="clear_chat_history")
            #     st.stop()



            with st.chat_message("assistant"):
                # response_stream = getResponseFromQuery(question, query, result, schema, chat_history_str)
                response_stream = llm_agentic(llm, question)
                full_res = st.write_stream(response_stream)

            # with st.spinner("Waiting..."):
            # if full_response:        
            st.session_state.chats.append({
                "role": "assistant",
                "content": full_res
            })

    # with chat_container:
        
            # else:
            #     st.chat_message(chat["role"]).write_stream(chat["content"])

        
  

    

