import asyncio
import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
import time
from transformers import AutoTokenizer
import unidecode
import types 
import re
import base64
from io import BytesIO
from PIL import Image

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

def parse_plot_output(output: str) -> str:
    match = re.search(r"(iVBORw0KGgo[^\"'\s]+)", output)
    if not match:
        return None
    return match.group(1)


# def clear_chat_history():
#     st.session_state.chats = [{"role": "assistant", "content": "Hi. I'm your MySQL Assistant. Ask me anything about your database!"}]
    # del st.session_state.chats

def chat_component(db):
    # the_input = st.empty()
    # chat_container = st.container()

    if "chats" not in st.session_state:
        st.session_state.chats = [{"role": "assistant", "content": "Hi. I'm your MySQL Assistant. Ask me anything about your database!"}]

    if question := st.chat_input('Chat with your MySQL database'):
        st.session_state.chats.append({
            "role": "user",
            "content": question
        })
        # with st.chat_message("user"):
        #     st.write(question)
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
                
                if isinstance(response_stream, types.GeneratorType):
                    collected_response = ""
                    tool_response = ""
                    for step_text, metadata in response_stream:
                        node = metadata.get("langgraph_node", "")
                        if node == "tools" and step_text:
                            tool_response += step_text
                        if node == "agent" and step_text:
                            collected_response += step_text

                    is_base_64 = parse_plot_output(tool_response)
                    if is_base_64:
                        base64_str = is_base_64
                        img_data = base64.b64decode(base64_str)
                        img = Image.open(BytesIO(img_data))
                        res_img = st.image(img)
                        st.session_state.chats.append({
                            "role": "assistant",
                            "content": f"[IMAGE] IMG"
                        })

                    
                    if collected_response.strip():
                        # st.write(collected_response, unsafe_allow_html=True)
                        st.session_state.chats.append({
                            "role": "assistant",
                            "content": collected_response
                        })
                
                
    for chat in st.session_state.chats:
        if not chat["content"] or chat["content"].strip() == "":
            continue 
        with st.chat_message(chat["role"]):
            if isinstance(chat["content"], str) and chat["content"].startswith("[IMAGE]"):
                st.write(chat['content'][7:])
            else:
                st.write(chat["content"], unsafe_allow_html=True)

    # base64_str = '''
    # iVBORw0KGgoAAAANSUhEUgAAA+gAAAJYCAYAAADxHswlAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjEsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvc2/+5QAAAAlwSFlzAAAPYQAAD2EBqD+naQAAbBpJREFUeJzt3Qm8b1P9P/51zUPm4aJMlQxRhDKlwVgaRLNCCckQipDMIolSIpWh8KXhS0WJKDJkuEiGaCAqQ2Um8/k/Xuv7W+f/Oeeey73ce8++Ps/n4/G5534+n/3Ze+211157vffae+0xAwMDAwUAAAAYVdON7uIBAACAEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKAD8KxOPPHEMmbMmHLVVVeVadlTTz1Vdt9997LooouW6aabrmy88caly9785jfXF1Ne8nn55Zcf7WQAgAAdYDSC3d7XggsuWN7ylreUX/ziF1M9PWeccUZ529veVuaff/4y00wzlUUWWaS8//3vLxdccEEZLd/85jdrPk1uxx9/fPnyl79c3vve95aTTjqp7LLLLqWf/fznPy/77bdfeTG69NJL67rdf//95cVqSu0nAIyuGUZ5+QB96YADDihLLrlkGRgYKHfffXdtaL/97W8vP/vZz8o73vGOKb78LPfjH/94Xe5KK61Udt1117LQQguVO++8swbt66yzTrnkkkvKGmusUUYj8MgJgy233HKyzjcnHV760peWI488crLOd1oO0I8++ugXZZCeAH3//fevZWjuuecuL0ZTaj8BYHQJ0AFGQXqtV1lllcH3W221VRk7dmz5n//5n8kSoD/zzDPliSeeKLPMMsuI33/lK1+pwfnOO+9cjjjiiNqT33z+858v3//+98sMM0zdQ8Sjjz5aZptttik2/3vuuWeqBGuPPPJImX322af4cmBaYH8AmDQucQfogASOs84663hB8eGHH157seebb776/corr1x+9KMfjff7BNg77LBDOeWUU8qrX/3qMvPMM5dzzjlnxGX997//LYccckhZZpll6vx7g/Pmox/9aHn9618/5LPHH3+89rQvsMACtcH9nve8p/zrX/8aMs1PfvKTstFGG9VL5ZOGV7ziFeXAAw8sTz/99Ij3/I4bN66svfbaNTDfa6+9yhJLLFFuuOGGcuGFFw7eAvBc92EnAPjMZz5T7y3PMpdeeum6XrlKIG677bY6n1//+td13m2+v/nNbyY4z6QjJ0rOPffcsuKKK9YTHcstt1z53//93xFvWUh6P/WpT9XbFV72spcN6eVs2yN5sv3224942fVxxx1X8yrbOPn+29/+drxp2rKyPr2yHiOtz+WXX16vyphnnnnq9nrNa15Tvva1r9Xv0uua3vPovd1iUjzb/HuvWnjjG99Yv08Zf/e7311uuummIdMkLcnv4dKzPzxNrZyfeeaZtfwkX5O/vWU9v9ttt93q/3OVSlu34fk2kpTH7G/ZDvntscceO/jdww8/XNfj05/+9Hi/+/vf/16mn376ul8914mz5NEKK6xQy1T2pQ033HDI+A4nnHBCeetb31rLUtYv5e6YY44ZMp/n2k9SxnLyre0Tr3zlK8uXvvSluvxe//nPf+q+Puecc9bts8UWW5Tf//73dX7DL5+fmG3ZttmNN95YPvzhD9eysdZaa9V1yufXXHPNeHnyxS9+sebdP/7xj2fNO4B+oQcdYBQ88MAD5d///ncNItOz+/Wvf70GAB/5yEeGTJfG/Lve9a6y2Wab1R7x0047rbzvfe8rZ511Vg2Ehzegf/CDH9QAJpe+jhT0xMUXX1zuvffe2oBPw3hi7bjjjrXBve+++9Zg56tf/Wpd1umnnz44TRr1L3nJS2ogn79J0z777FMefPDBev/38OAgVxJ88IMfrOudKwgSZGQ5+W168iOfT0jyL/mT4DtXISSY/uUvf1kDtDT4czl7gqBcEXDwwQfXPG5B1LLLLvus6/unP/2pfOADHyif/OQna+CSICN5n2BwvfXWGzJtgvMsJ+uaEwYtWMll1uuuu27Zbrvtys0331wDrSuvvLLePjDjjDPW6b773e+WbbfdtgaG2SZ//etf6zrNO++8NcB6Ps4777x6gmHhhReuAWVuX0gwlXKT91neP//5zzpd8mZyzz9+9atf1e378pe/vOZFTgylnK+55prl6quvnmD5fC4pvzlRkjyfY445ylFHHVU23XTTcvvtt9cTWZtsskm55ZZb6tUo2f7ZFyLb59ncd9999YRDxmD40Ic+VPelbLeMzZDbQVImc1Iq5T1XnfTuO1lWymL202eTMpp9JPnyiU98og5cmJMxv/vd7wavqEkZyUmHlIGcsMttL1nXBNc5wRPZ9ya0n+RKlDe96U21/Gc7L7bYYvWS/z333LPewpLfRub3zne+s1xxxRV1PXPCLifYUtaHm9Rtmf1kqaWWqsF38iXjPiTtOYGYW2p65bPs97n9BID/a9wAMJWccMIJ6dYd7zXzzDMPnHjiieNN/+ijjw55/8QTTwwsv/zyA29961uHfJ55TDfddAM33HDDc6bha1/7Wp3+jDPOmKQ0r7vuugPPPPPM4Oe77LLLwPTTTz9w//33TzC9se222w7MNttsA4899tjgZ29605vqPI899tjxpn/1q19dv58YZ555Zp3PQQcdNOTz9773vQNjxowZ+POf/zxkmZn3xFh88cXrfH/84x8PfvbAAw8MLLzwwgMrrbTSeHmz1lprDTz11FODn99zzz0DM80008D6668/8PTTTw9+/o1vfKNOf/zxxw9uzwUXXHBgxRVXHHj88ccHpzvuuOPqdL350JZ16623Dknrr3/96/p5/kbSseSSS9Z1uO+++4ZM27v9tt9++/q7STWx8886Zd3+85//DH72+9//vpbTzTfffPCzLbbYos5ruH333Xe89OV98rV3u2ae+fzrX//64Gdf/vKXR8yrCWnl8Stf+crgZ9kebR2yneKXv/xlne4Xv/jFkN+/5jWvec4ye8EFF9Tf7rTTTuN915tvI+1DG2ywwcDLX/7yidpPDjzwwIHZZ5994JZbbhny+R577FH319tvv72+T9lOer761a8OTpOymroln6e8Teq2bNvsQx/60HjpymeLLLLIkP3h6quvHm9ZAP3OJe4AoyCXF6cXMq+TTz65juKeHrXhl1DnUtve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+ondlhPbi/7Zz362/t/l7QAjGO1h5AH6/TFrs8wyS32M0THHHDPkcUvx3e9+d2CppZaqj2FbZpll6u8n9PipPDZrUv3oRz+qjwKbd955B2aYYYb6GLEPfOADA7/5zW/GS/OVV175rI/3iksuuWRgtdVWG5h11lnrI5V23333wUdT9U73bI88u+uuuwY22mijgTnmmGO8R42N5KGHHqqPfMvyZpxxxppfeczW8Lyc1MesJQ1Jex6h1fL/hz/84ZDpJpQ3vY9Vy++SrrFjxw5st9124z2aLL75zW/WR5dlOausssrARRddVNM7fN3/8pe/1MfdZbrMb6+99ho477zzxsvfuPjiiwfWW2+9mo957FbWo/dRZHlc2o477jiwwAIL1EfSTWqT4LnmH7/61a8G1lxzzVoe5pxzzoF3vvOdAzfeeON48zr33HPr4wPzCLWll1564OSTT56kcp7tlce1DX/c2Etf+tL6KLDneuRaKxtXXXXVwOqrr173ycwz229C3v72t9f5XnrppQMTK3mespkykXVN3r/tbW8bGDdu3OA0P/3pT2teJg1LLLHEwJe+9KX6WL7h6/Bs+0n2iT333HPgla98ZV3O/PPPP7DGGmsMHH744YOPjIt//etfAx/+8IfrPOaaa66BLbfcsu7Dmd9pp502yduybbPMd0LuvPPO+ri3V73qVROdbwD9ZEz+GSlwB4B+lXuOc69xRtGHkaTHP73Sw+/ZntblCpqs28UXX1zWXHPNyT7/DICXKy722Wef8oUvfGGyzx9gWucedACASZDLzTPCei5Zn5blUvVeGdMgl67nNpuMNTAlZHyFLGdazzuAKcU96ABAlfuPewerGy73qU/o+dr9IOMD5DnlGeQs953nPvFp2Y477liD9NxbntHj8wSGjD3xxS9+cbI/OvCCCy4oN954Yzn44IPr/e/P9SQEgH4lQAcAqk022aRceOGFE/w+j9zKIGP9KnmTQfHyCLSMav5co/l3XQaVy0CRuZUjzybPYHPpQc+AdZPbAQccUIP/XDY/MY+lA+hX7kEHAKpx48aV++67b4Lfp1d1StyXDAD8HwE6AAAAdIBB4gAAAKAD3IPeZ5555pnyz3/+s8wxxxxlzJgxo50cAABglORi6oceeqgsssgiZbrp9N12gQC9zyQ4X3TRRUc7GQAAQEfccccd5WUve9loJwMBev9Jz3nbCfOcUwAAoD89+OCDtfOuxQiMPgF6n2mXtSc4F6ADAABufe0ONxoAAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAH0yueiii8o73/nOssgii5QxY8aUM888c8j3AwMDZZ999ikLL7xwmXXWWcu6665b/vSnPw2Z5t577y2bbbZZmXPOOcvcc89dttpqq/Lwww8Pmea6664rb3zjG8sss8xSFl100XLYYYdNlfUDAABgyhKgTyaPPPJIee1rX1uOPvroEb9PIH3UUUeVY489tlx++eVl9tlnLxtssEF57LHHBqdJcH7DDTeU8847r5x11lk16N9mm20Gv3/wwQfL+uuvXxZffPEybty48uUvf7nst99+5bjjjpsq6wgAAMCUM2YgXbtMVulBP+OMM8rGG29c3yeL07P+mc98pnz2s5+tnz3wwANl7Nix5cQTTywf/OAHy0033VSWW265cuWVV5ZVVlmlTnPOOeeUt7/97eXvf/97/f0xxxxTPv/5z5e77rqrzDTTTHWaPfbYo/bW//GPf5yotCXIn2uuuery01MPAAD0J7FB9+hBnwpuvfXWGlTnsvYmO8Ib3vCGctlll9X3+ZvL2ltwHpl+uummqz3ubZq11157MDiP9MLffPPN5b777htx2Y8//njd8XpfAAAAdI8AfSpIcB7pMe+V9+27/F1wwQWHfD/DDDOUeeedd8g0I82jdxnDHXLIIfVkQHvlvnUAAAC6R4D+IrfnnnvWS1ba64477hjtJAEAADCCGUb6kMlroYUWqn/vvvvuOop7k/crrrji4DT33HPPkN899dRTdWT39vv8zW96tfdtmuFmnnnm+gIApq4l9ji7TEtuO3Sj0U4CQN/Tgz4VLLnkkjWAPv/88wc/y73gubd89dVXr+/z9/7776+jszcXXHBBeeaZZ+q96m2ajOz+5JNPDk6TEd+XXnrpMs8880zVdQIAAGDyEqBPJnle+bXXXltfbWC4/P/222+vo7rvvPPO5aCDDio//elPyx/+8Iey+eab15HZ20jvyy67bNlwww3L1ltvXa644opyySWXlB122KGO8J7p4sMf/nAdIC7PR8/j2E4//fTyta99rey6666juu4AAAC8cC5xn0yuuuqq8pa3vGXwfQuat9hii/ootd13370+Kz3PNU9P+VprrVUfozbLLLMM/uaUU06pQfk666xTR2/fdNNN67PTmwzydu6555btt9++rLzyymX++ecv++yzz5BnpQMAADBt8hz0PuNZhwAwdbgHHeg6sUH3uMQdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6FPJ008/Xb7whS+UJZdcssw666zlFa94RTnwwAPLwMDA4DT5/z777FMWXnjhOs26665b/vSnPw2Zz7333ls222yzMuecc5a55567bLXVVuXhhx8ehTUCAABgchKgTyVf+tKXyjHHHFO+8Y1vlJtuuqm+P+yww8rXv/71wWny/qijjirHHntsufzyy8vss89eNthgg/LYY48NTpPg/IYbbijnnXdeOeuss8pFF11Uttlmm1FaKwAAACaXMQO9XbhMMe94xzvK2LFjy3e/+93BzzbddNPaU37yySfX3vNFFlmkfOYznymf/exn6/cPPPBA/c2JJ55YPvjBD9bAfrnllitXXnllWWWVVeo055xzTnn7299e/v73v9ffP5cHH3ywzDXXXHXe6YUHAKaMJfY4u0xLbjt0o9FOAjCViQ26Rw/6VLLGGmuU888/v9xyyy31/e9///ty8cUXl7e97W31/a233lruuuuuell7k53lDW94Q7nsssvq+/zNZe0tOI9MP91009UedwAAAKZdM4x2AvrFHnvsUc9QLbPMMmX66aev96QffPDB9ZL1SHAe6THvlfftu/xdcMEFh3w/wwwzlHnnnXdwmuEef/zx+mqSBgAAALpHD/pU8oMf/KCccsop5dRTTy1XX311Oemkk8rhhx9e/05JhxxySO2Jb69FF110ii4PAACA50eAPpXstttutRc995KvsMIK5aMf/WjZZZddagAdCy20UP179913D/ld3rfv8veee+4Z8v1TTz1VR3Zv0wy355571ntK2uuOO+6YQmsIAADACyFAn0oeffTReq94r1zq/swzz9T/5/FrCbJzn3rv5ei5t3z11Vev7/P3/vvvL+PGjRuc5oILLqjzyL3qI5l55pnrgA+9LwAAALrHPehTyTvf+c56z/liiy1WXv3qV5drrrmmHHHEEeXjH/94/X7MmDFl5513LgcddFBZaqmlasCe56ZnZPaNN964TrPsssuWDTfcsGy99db1UWxPPvlk2WGHHWqv/MSM4A4AAEB3CdCnkjzvPAH3pz71qXqZegLqbbfdtuyzzz6D0+y+++7lkUceqc81T0/5WmutVR+jNsssswxOk/vYE5Svs846tUc+j2rLs9MBAACYtnkOep/xrEMAmDo8Bx3oOrFB97gHHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdMAMo50AAKB/LbHH2WVactuhG412EgB4EdODDgAAAB2gBx2gT+ipBADoNj3oAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6IC+D9C/973vlccff3y8z5944on6HQAAAEwNfR+gf+xjHysPPPDAeJ8/9NBD9TsAAACYGvo+QB8YGChjxowZ7/O///3vZa655hqVNAEAANB/Zih9aqWVVqqBeV7rrLNOmWGG/z8rnn766XLrrbeWDTfccFTTCAAAQP/o2wB94403rn+vvfbassEGG5SXvOQlg9/NNNNMZYklliibbrrpKKYQAACAftK3Afq+++5b/yYQ/8AHPlBmmWWW0U4SAAAAfaxvA/Rmiy22GBy1/Z577inPPPPMkO8XW2yxUUoZAAAA/aTvA/Q//elP5eMf/3i59NJLRxw8LvejAwAAwJTW96O4b7nllmW66aYrZ511Vhk3bly5+uqr6+uaa66pfyenf/zjH+UjH/lImW+++cqss85aVlhhhXLVVVcNOSmwzz77lIUXXrh+v+6669YTCL3uvffestlmm5U555yzzD333GWrrbYqDz/88GRNJwAAAFNf3/egZ5C4BObLLLPMFF3OfffdV9Zcc83ylre8pfziF78oCyywQA2+55lnnsFpDjvssHLUUUeVk046qSy55JLlC1/4Qh3A7sYbbxy8Rz7B+Z133lnOO++88uSTT9ZntW+zzTbl1FNPnaLpBwAAYMrq+wB9ueWWK//+97+n+HK+9KUvlUUXXbSccMIJg58lCO/tPf/qV79a9t577/Lud7+7fva9732vjB07tpx55pnlgx/8YLnpppvKOeecU6688sqyyiqr1Gm+/vWvl7e//e3l8MMPL4ssssgUXw8AAACmjL6/xD2B8+67715+85vflP/85z/lwQcfHPKaXH7605/WoPp973tfWXDBBetz2L/97W8Pfp/nrt911131svZmrrnmKm94wxvKZZddVt/nby5rb8F5ZPpcon/55ZdPtrQCAAAw9fV9D3oLiNdZZ50pOkjcX//613LMMceUXXfdtey11161F3ynnXaqz1zPSPIJziM95r3yvn2Xvwnue80wwwxl3nnnHZxmuMcff7y+msl50gEAAIDJp+8D9F//+tdTZTl5fFt6vr/4xS/W9+lBv/7668uxxx47+Ki3KeGQQw4p+++//xSbPwAAAJNH3wfob3rTm6bKcjIye+5377XsssuWH//4x/X/Cy20UP17991312mbvF9xxRUHp8mz2ns99dRTdWT39vvh9txzz9pr39uDnnvhAZg2LLHH2WVac9uhG412EgBgmtT3AfpFF130rN+vvfbak2U5GcH95ptvHvLZLbfcUhZffPHBAeMSZJ9//vmDAXmC6dxbvt1229X3q6++ern//vvrqPMrr7xy/eyCCy6ovfO5V30kM888c30BAADQbX0foL/5zW8e77Pce95MrnvQd9lll7LGGmvUS9zf//73lyuuuKIcd9xx9dWWufPOO5eDDjqoLLXUUoOPWcvI7BtvvPFgj/uGG25Ytt5663ppfB6ztsMOO9QR3o3gDgAAMG3r+wA9zyfvlaD3mmuuqcHxwQcfPNmWs+qqq5YzzjijXnJ+wAEH1AA8j1XLc82bjCb/yCOP1Oeap6d8rbXWqo9Va89Aj1NOOaUG5RnULqO3b7rppvXZ6QAAAEzb+j5Az6PMhltvvfXq6Oq5dzuXk08u73jHO+prQtKLnuA9rwnJiO2nnnrqZEsTAAAA3dD3z0GfkDzebPg94wAAADCl9H0P+nXXXTfe88/vvPPOcuihhw4O1gZAt01rI50b5RwAGEnfB+gJwnNpeQLzXquttlo5/vjjRy1dAAAA9Je+D9BvvfXWIe8z8NoCCywwZGA2AAAAmNL6PkBvzyEHAACA0WSQuFLKhRdeWN75zneWV77ylfX1rne9q/z2t78d7WQBAADQR/o+QD/55JPLuuuuW2abbbay00471dess85anzPucWYAAABMLX1/ifvBBx9cDjvssLLLLrsMfpYg/YgjjigHHnhg+fCHPzyq6QMAAKA/9H0P+l//+td6eftwucx9+AByAAAAMKX0fYC+6KKLlvPPP3+8z3/1q1/V7wAAAGBq6PtL3D/zmc/US9qvvfbassYaa9TPLrnkknLiiSeWr33ta6OdPAAAAPpE3wfo2223XVlooYXKV77ylfKDH/ygfrbsssuW008/vbz73e8e7eQBAADQJ/o+QI/3vOc99QUAAACjpe/vQb/yyivL5ZdfPt7n+eyqq64alTQBAADQf/o+QN9+++3LHXfcMd7n//jHP+p3AAAAMDX0fYB+4403lte97nXjfb7SSivV7wAAAGBq6PsAfeaZZy533333eJ/feeedZYYZ3KIPAADA1NH3Afr6669f9txzz/LAAw8Mfnb//feXvfbaq6y33nqjmjYAAAD6R993ER9++OFl7bXXLosvvni9rD3yTPSxY8eW73//+6OdPAAAAPpE3wfoL33pS8t1111XTjnllPL73/++zDrrrOVjH/tY+dCHPlRmnHHG0U4eAAAAfaLvA/SYffbZyzbbbPOs02y00UblO9/5Tll44YWnWroAAADoH31/D/rEuuiii8p///vf0U4GAAAAL1ICdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAO6PsAPaOzP/XUU+N9ns/yXbPXXnuVeeeddyqnDgAAgH7R9wH6W97ylnLvvfeO9/kDDzxQv2v23HPPMvfcc0/l1AEAANAv+j5AHxgYKGPGjBnv8//85z9l9tlnH5U0AQAA0H9mKH1qk002qX8TnG+55ZZl5plnHvzu6aefLtddd11ZY401RjGFAAAA9JO+DdDnmmuuwR70OeaYo8w666yD380000xltdVWK1tvvfUophAAAIB+0rcB+gknnFD/LrHEEmW33XYrs80222gnCQAAgD7W9/egb7755uUf//jHeJ//6U9/KrfddtuopAkAAID+0/cBeu4/v/TSS8f7/PLLL6/fAQAAwNTQ9wH6NddcU9Zcc83xPs896Ndee+2opAkAAID+0/cBekZxf+ihh0Z8DnpGcwcAAICpoe8D9LXXXrsccsghQ4Lx/D+frbXWWqOaNgAAAPpH347i3nzpS1+qQfrSSy9d3vjGN9bPfvvb35YHH3ywXHDBBaOdPAAAAPpE3/egL7fccuW6664r73//+8s999xTL3fPyO5//OMfy/LLLz/ayQMAAKBP9H0PeiyyyCLli1/84mgnAwAAgD7W9wH6RRdd9Kzf5/J3AAAAmNL6PkB/85vfPOLI7o2R3AEAAJga+v4e9Pvuu2/IK/ehn3POOWXVVVct55577mgnDwAAgD7R9z3oc80113ifrbfeemWmmWYqu+66axk3btyopAsAAID+0vc96BMyduzYcvPNN492MgAAAOgTfd+Dnkes9RoYGCh33nlnOfTQQ8uKK644aukCAACgv/R9gJ4gPIPCJTDvtdpqq5Xjjz9+1NIFAABAf+n7AP3WW28d8n666aYrCyywQJlllllGLU0AAAD0n74P0BdffPHRTgIAAAD0Z4B+1FFHTfS0O+200xRNCwAAAPRtgH7kkUcOef+vf/2rPProo2Xuueeu7++///4y22yzlQUXXFCADgAAwFQxXb/ed95eBx98cB0o7qabbir33ntvfeX/r3vd68qBBx442kkFAACgT/RlgN7rC1/4Qvn6179ell566cHP8v/0su+9996jmjYAAAD6R98H6Hnm+VNPPTXe508//XS5++67RyVNAAAA9J++D9DXWWedsu2225arr7568LNx48aV7bbbrqy77rqjmjYAAAD6R98H6Mcff3xZaKGFyiqrrFJmnnnm+nr9619fxo4dW77zne+MdvIAAADoE305inuvBRZYoPz85z8vt9xySx0cbsyYMWWZZZYpr3rVq0Y7aQAAAPSRvg/QmwTkSy21VP1/gnQAAACYmvr+Evf43ve+V1ZYYYUy66yz1tdrXvOa8v3vf3+0kwUAAEAf6fse9COOOKI+am2HHXYoa665Zv3s4osvLp/85CfLv//977LLLruMdhIBAADoA30foOcZ6Mccc0zZfPPNBz9717veVV796leX/fbbT4AOAADAVNH3l7jnOehrrLHGeJ/ns3wHAAAAU0PfB+ivfOUryw9+8IPxPj/99NMHB40DAACAKa3vL3Hff//9ywc+8IFy0UUXDd6Dfskll5Tzzz9/xMAdAAAApoS+70HfdNNNyxVXXFHmn3/+cuaZZ9ZX/p/P3vOe94x28gAAAOgTfd2D/uSTT5Ztt922juJ+8sknj3ZyAAAA6GN93YM+44wzlh//+MejnQwAAADo7wA9Nt5443pZOwAAAIymvr7EPTJS+wEHHFAHhlt55ZXL7LPPPuT7nXbaadTSBgAAQP/o+wD9u9/9bpl77rnLuHHj6qvXmDFjBOgAAABMFX0foN96662D/x8YGBgMzAEAAGBq6vt70Fsv+vLLL19mmWWW+sr/v/Od74x2sgAAAOgjfd+Dvs8++5Qjjjii7LjjjmX11Vevn1122WVll112Kbfffnu9Px0AAACmtL4P0I855pjy7W9/u3zoQx8a/Oxd73pXec1rXlODdgE6AAAAU0PfX+L+5JNPllVWWWW8zzOi+1NPPTUqaQIAAKD/9H2A/tGPfrT2og933HHHlc0222xU0gQAAED/6ftL3Nsgceeee25ZbbXV6vvLL7+83n+++eabl1133XVwutyrDgAAAFNC3/egX3/99eV1r3tdWWCBBcpf/vKX+pp//vnrZ/nummuuqa9rr712si730EMPrY9z23nnnQc/e+yxx8r2229f5ptvvvKSl7ykbLrppuXuu+8e8rucONhoo43KbLPNVhZccMGy2267uRQfAADgRaDve9B//etfT/VlXnnlleVb3/pWHYiuV0aOP/vss8sPf/jDMtdcc5UddtihbLLJJuWSSy6p3z/99NM1OF9ooYXKpZdeWu68887ayz/jjDOWL37xi1N9PQAAAJh8+r4HfWp7+OGH673tGTl+nnnmGfz8gQceqJfa5zL6t771rXWQuhNOOKEG4r/73e/qNLkM/8Ybbywnn3xyWXHFFcvb3va2cuCBB5ajjz66PPHEE6O4VgAAALxQfd+DPrXlEvb0gq+77rrloIMOGvx83LhxdUT5fN4ss8wyZbHFFqvPZc/98fm7wgorlLFjxw5Os8EGG5Ttttuu3HDDDWWllVYab3mPP/54fTUPPvjgFF0//n9L7HF2mZbcduhGo50EAADoawL0qei0004rV199db3Efbi77rqrzDTTTGXuuece8nmC8XzXpukNztv37buRHHLIIWX//fefjGsBAADAlOAS96nkjjvuKJ/+9KfLKaecUmaZZZapttw999yzXj7fXkkHAAAA3SNAn0pyCfs999xTR4efYYYZ6uvCCy8sRx11VP1/esJzH/n9998/5HcZxT2DwkX+Dh/Vvb1v0ww388wzlznnnHPICwAAgO4RoE8l66yzTvnDH/5QH9fWXqusskodMK79P6Oxn3/++YO/ufnmm+tj1VZfffX6Pn8zjwT6zXnnnVeD7uWWW25U1gsAAIDJwz3oU8kcc8xRll9++SGfzT777PWZ5+3zrbbaquy6665l3nnnrUH3jjvuWIPyDBAX66+/fg3EP/rRj5bDDjus3ne+995714Hn0lMOAADAtEuA3iFHHnlkmW666cqmm25aR17PCO3f/OY3B7+ffvrpy1lnnVVHbU/gngB/iy22KAcccMCophsAAIAXToA+in7zm98MeZ/B4/JM87wmZPHFFy8///nPp0LqAAAAmJrcgw4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AEzjHYC6G9L7HF2mZbcduhGo50EAGAK0jYBRpMedAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOhTySGHHFJWXXXVMsccc5QFF1ywbLzxxuXmm28eMs1jjz1Wtt9++zLffPOVl7zkJWXTTTctd99995Bpbr/99rLRRhuV2Wabrc5nt912K0899dRUXhsAAAAmNwH6VHLhhRfW4Pt3v/tdOe+888qTTz5Z1l9//fLII48MTrPLLruUn/3sZ+WHP/xhnf6f//xn2WSTTQa/f/rpp2tw/sQTT5RLL720nHTSSeXEE08s++yzzyitFQAAAJPLDJNtTjyrc845Z8j7BNbpAR83blxZe+21ywMPPFC++93vllNPPbW89a1vrdOccMIJZdlll61B/WqrrVbOPffccuONN5Zf/epXZezYsWXFFVcsBx54YPnc5z5X9ttvvzLTTDON0toBAADwQulBHyUJyGPeeeetfxOop1d93XXXHZxmmWWWKYsttli57LLL6vv8XWGFFWpw3mywwQblwQcfLDfccMNUXwcAAAAmHz3oo+CZZ54pO++8c1lzzTXL8ssvXz+76667ag/43HPPPWTaBOP5rk3TG5y379t3I3n88cfrq0kwDwAAQPfoQR8FuRf9+uuvL6eddtpUGZxurrnmGnwtuuiiU3yZAAAATDoB+lS2ww47lLPOOqv8+te/Li972csGP19ooYXq4G/333//kOkzinu+a9MMH9W9vW/TDLfnnnvWy+nb64477pgCawUAAMALJUCfSgYGBmpwfsYZZ5QLLrigLLnkkkO+X3nllcuMM85Yzj///MHP8hi2PFZt9dVXr+/z9w9/+EO55557BqfJiPBzzjlnWW655UZc7swzz1y/730BAADQPe5Bn4qXtWeE9p/85Cf1WejtnvFcdj7rrLPWv1tttVXZdddd68BxCaR33HHHGpRnBPfIY9kSiH/0ox8thx12WJ3H3nvvXeedQBwAAIBplwB9KjnmmGPq3ze/+c1DPs+j1Lbccsv6/yOPPLJMN910ZdNNN60Du2WE9m9+85uD004//fT18vjtttuuBu6zzz572WKLLcoBBxwwldcGAACAyU2APhUvcX8us8wySzn66KPra0IWX3zx8vOf/3wypw4AAIDR5h50AAAA6AA96MAkW2KPs8u05LZDNxrtJAAAwHPSgw4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHzDDaCQDokiX2OLtMS247dKPRTgIAAJOJHnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNABAACgAwToAAAA0AEzjHYCAACYtiyxx9llWnLboRuNdhIAJooedAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdMMNoJwAAAJjyltjj7DItue3QjUY7CTDV6UEHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAHSBABwAAgA4QoAMAAEAHCNCnQUcffXRZYoklyiyzzFLe8IY3lCuuuGK0kwQAAMALJECfxpx++ull1113Lfvuu2+5+uqry2tf+9qywQYblHvuuWe0kwYAAMALIECfxhxxxBFl6623Lh/72MfKcsstV4499tgy22yzleOPP360kwYAAMALMMML+TFT1xNPPFHGjRtX9txzz8HPpptuurLuuuuWyy67bMTfPP744/XVPPDAA/Xvgw8+WLrgmccfLdOSSck369Yd1u3/WLfueDGv24t9/azb/7Fu3WHdmBx5PDAwMNpJ4f8ZM2BrTDP++c9/lpe+9KXl0ksvLauvvvrg57vvvnu58MILy+WXXz7eb/bbb7+y//77T+WUAgAA04o77rijvOxlLxvtZKAH/cUvve25Z7155plnyr333lvmm2++MmbMmPJiPRO46KKL1opmzjnnLC8m1m3aZN2mTS/mdXuxr591mzZZt2mTdZu2pa/2oYceKosssshoJ4X/R4A+DZl//vnL9NNPX+6+++4hn+f9QgstNOJvZp555vrqNffcc5d+kIr0xVqZWrdpk3WbNr2Y1+3Fvn7Wbdpk3aZN1m3aNddcc412EuhhkLhpyEwzzVRWXnnlcv755w/pEc/73kveAQAAmPboQZ/G5HL1LbbYoqyyyirl9a9/ffnqV79aHnnkkTqqOwAAANMuAfo05gMf+ED517/+VfbZZ59y1113lRVXXLGcc845ZezYsaOdtM7IJf15TvzwS/tfDKzbtMm6TZtezOv2Yl8/6zZtsm7TJusGk5dR3AEAAKAD3IMOAAAAHSBABwAAgA4QoAMAAEAHCNCBiXLiiSeWueeee7zPf/Ob35QxY8aU+++/v0yLnk/63/zmN5edd9558P0SSyxRn6jQZH5nnnlmue222+r/r7322jIanu/yh6/P8y0bk8OkpmVqlIXR3q6Takpuny7ab7/96gCqk2L4Pj0hl1xySZl99tnLdNNNVzbeeONJTluG/dlmm23KvPPO+6xl6LnK4rRW7z6fbTKtrWPSutVWWz3nNDk2TIsmZhtuueWWz2u/mNz5NbnT8XzrC3i+BOhMNqkQU5m213zzzVc23HDDct1115WuyAj42223XVlsscXqiJwLLbRQ2WCDDWqj68VsSqz382lwTY5y1V5//vOfJ2tgMs8885RPfvKT432+/fbb1+UlHc3//u//lgMPPLC8GORpEDvuuGN5+ctfXsvGoosuWu65555yyy23TNLTJSZl+n7yQhrkw/exO+64o3z84x8viyyySJlpppnK4osvXj796U+X//znP0NOaHz4wx9+znI/0nZ/5zvfWc4///wyrfvsZz87ZD1aY/3Z6pKJ3afzqNOXvOQldTukfpmUbZll5RGp+d1ZZ51V7rzzzrL88suXL3/5y/W7NPon1hprrFF/P9dcc02wrpvYOrqlbfhrmWWWmSr19AtZxylhWg6iuxC8Mvrbamq1z5hyPGaNySoB+QknnDDYANx7773LO97xjnL77bePWpqeeOKJ2piNTTfdtL4/6aSTasP07rvvrg253gZuF9I5uXV5vSe1XDULLLDAZF3GS1/60nLaaaeVI488ssw666z1s8cee6yceuqp9cRGr/R+vRikJ3jNNdesjd4ECSussEJ58sknyxvf+Mby4x//uHzzm9+cqPkkv1qejVSWp2TZHqmH8umnny4zzPDiOrz99a9/Lauvvnp51ateVf7nf/6nLLnkkuWGG24ou+22W/nFL35Rfve73010uZzQdv/lL39ZT0j98Y9/nKj55Dczzjhj6ZoE0HlNSl0y/fTTT9S8//KXv5T555+/zn9Sg8WFF164/P73v68nSBN8Nscff/xgHTOx+0qmyXwml1e/+tXlV7/61ZDPJnUfmth6erTWkUmTejQnK3K1yKR45plnpliaoG/kMWswOWyxxRYD7373u4d89tvf/jaP8Ru455576vvbb7994H3ve9/AXHPNNTDPPPMMvOtd7xq49dZb63e//OUvB2aeeeaB++67b8g8dtppp4G3vOUtQ+a51lprDcwyyywDL3vZywZ23HHHgYcffnjw+8UXX3zggAMOGPjoRz86MMccc9R0tfkkLTPNNNPAkksuObD33nsPPPHEE0OWteuuu9ZpZp111oGtttpq4HOf+9zA8ssvXz/79a9/Xaf59re/PfDSl750YMyYMfW11FJLDZx44ol1mrvvvntg++23H1hwwQUHpptuuoHpp59+YMYZZ6zzOPXUU4cs601velOd9tOf/vTAfPPNN/DmN7954GMf+9jARhttNGS6pHGBBRYY+M53vjNivt92220D73jHOwbmnnvugdlmm21gueWWGzj77LPrd0lz0pXXq171qrpeq6+++sAf//jHwd//+c9/rtvhJS95SV2fpDvL22677QYeeuihOs1jjz028La3va1+n/zL922+7XX00UfX9G+44Yb1tcYaa9RtmW2T5c4555x1+y622GIDX/ziFycp/ZnXyiuvPF76v/KVr9T1Sj4n3TPMMMPASiutNHDeeecNWff2SnlI+l/xilcM5mebLtss80kak9ak47jjjhtYYYUVannJNPl+9tlnr/mQ/PrWt741WOaOPPLIgcsvv3xgxRVXrNNmGf/7v/9b/7/NNtsMLLTQQgO///3vx9t+ybf8PunPfLOMVVZZpa7DL37xi4E111yz7i/5fumllx5473vfW5e96KKLDpbXts9k+ZkmnyXdbfknnHBCLW/Jv+Tz+uuvP3DvvffWbbrwwgvXPGzlOfPN9m3bKNNlXvmu5WPmcemllw6uQ/b7pC/7xhJLLFGnee1rXzvwta99raY177O/ZvtcccUVQ8pe8iXfJV/bejepG/Lb7MvZ7q3sZVvl/0n7e97znjrNz3/+84HXve51dX/LNk2Z3Xzzzet0+T7pW3bZZYfMP+UsacqrpSPTZlvGvvvuW9ejbdeUjdQ5rfxfc801dbrvfve7tbxkWZlP9uts02y33vI377zz1umvvfbaur+nvGfZSVvWP3XjXXfdVafJNhteflNnPvroo7Xcp07J75MHmUfqq2zj4b/JK9snafnhD39Yy3b7/I1vfOPALbfcMpgf2X7Z/q0sZNqU/aStbccPfehDg/mU9c0+9ZrXvGawfCSNP/3pTwe+9KUvDYwdO7bm1UEHHTSkzGe6b37zm7WeyLyyjKSt13XXXVfr/XyffNt6660H66O236666qo1D7JuqSNSl/Rut/b/4fnR6vLe41Hy8OUvf/ng8Sj75Stf+cq6zVPektbh88n+k/XNa5FFFhnc/jluPPnkk4NpTVl8/etfP2QfyitlJPVV1rHtNykLb3jDG4bU20lDps37bIusc3ufNLcyPnybD/8s23PTTTcdrw7qza/vfe97ta7N9s72y/bOce2f//znwNvf/va6f2Vee+65Z11u3mff3HjjjQfWWWedWl8lvdnuqSdafZF9c/7556/rmt+1NO+www51uZlHO0a37XPGGWfU92edddaI9XnyPJ/3bquUx9RzeSVN2VeyfZt///vftdy39cjf5EnbXi1f2yvbI/mQdkaOkykL2QbZ/1IvZpmpV84999zB/eODH/xg3UZJ3ze+8Y063xwrsq9mmqxXs95669VtnzRmvkl3KyfJq80226xui5TPTJN6KPVk0pX3qc8ybeqgbJ/8JvtwylDmk/m1dlbyIukfaX94+umnB48dva/zzz+/HpPe+c531nXN/7O/Z/2Shuw/WeeUt+znWV6+y/6csptpevNymWWWqfVUps/27i27SW/q1+OPP36wHObzrGeWcccdd9S6NftY0pL8TJqzzZNHaRfmmNLbHt19991rGy3TZ/lvfetba5uq1cHZ/vk82z3zyz7ygQ98YODBBx8c3EZpw+QYlO2TZR9++OF1HdJ2ezYvtJ57trqrrVfye3h7dqRjRz6Lv/3tb7U8ZF2Sb73HnObMM8+s9VLyLPPeb7/9BvePZ555pqYrZawdg7NvMPkJ0JliAXoqmW233bYeOFP5p/LIgezjH/94rZRuvPHGgQ9/+MO1gn388ccHnnrqqXog7A1Eh3+WyjcVSxrQaVhecskltSLZcsstB3+TijaVbCrRTN8q7FQyqcyy/B//+Md1vmlENieffPLgwSIV1P7771/n0xugZ5o0PHKwTEV61FFH1fSkgZZpcmIgFVfmv9dee9UDTdKa6XJgS0O/SQWfg8puu+1WA868sj6ZLo2hJkFWltHbOO2VgD4H+eTpX/7yl4Gf/exnAxdeeGH9rjVqcgBKYJeAIo2TNGabHKiOPfbYWuHnoJgGUw5aCTATpMf73//+evDIgTfLSGCT5b761a+u65plJM3nnHNO/X/W7ZFHHqm/TZCUA2++SwM6J1h6T1ZMTPqTv7/5zW8GbrjhhiHpT94mqEjDIGUkB5Mc1HOw+9Of/jTw1a9+tW7DNC7SWDrllFPqMn71q18NnHbaaUOWke2cA022SRq9mUcamzmYpdzkfRqaCYqTN0lD8jXbLWXu0EMPrWUjZTrzy/ZvgX2WnfQMd+WVV9btfcghhwwcfPDBAz/4wQ8GPv/5z9eDbZZ3zDHH1PzNb3MgzHbJvG6++eb6m9b4SYCe8pHlJ2jIZ8mPrEdrhGZbZltff/31A1//+tfr/pPtkkZzDsTJy5T77K/5TRpCkb/ZL9IYTV6nvCaPkr7//Oc/dZoWDKQhcvXVVw988pOfrGU2J57WXnvtur2zfVtA1353wQUX1HVI4zLpaeudRkRvgJ4G2o9+9KOaxqxLtkfWJfvTZz7zmTpNGo1pJGd/z/zT+Et+pDxnvRLIZ95t/gnO2smYzD/1V9Lb9pds1zREUi7adk3efeITn6j7fwvQ0wDLPFPWsl1yAiLpTJloQW22VcpotnXSmP0mDe/UhymvKTvf//736/+z70QC8axbpk1dmW2VOqmV++Rd8icN6KQ/65LgIw3clIEsOyeE7rzzzlqOkm/Z1i0Azz6VMpXtnbo5dVsa75lPtkNOriQf8j4n0NrJnrx23nnnuh9/5CMfqdPkdwk6ckIreZNgNPtS8jB1YH7zu9/9brDc533yOuUpeZblJW+ynq1BnLRtsskmA3/4wx/qOmZfaidb01jMOn/2s5+t2zu/y0nSVm56A87sF6m/Ujbzd4MNNqjHm+HHowT7ORbllZNPSU/KbeqslOnkefIy5SX7f8px9tWLLrpo8LiRfS75kHxOGpqUxWyPBGIJopL/SX/WPQ3+5Fk+yzEpJ5qyrJbXqZP/+te/1vKd90lv6rCcQMv71KuZT8pl5pG8TDpzUjxlrAUJKZdZr2zX4XrzK/ti0pBlXHbZZTVwTnlad911a3CYYLj3ZGdORqROzj6SdUxdnuNqOxGeaZMf7eRC1jdlrZ28bycCE8im7PYGIS1gz0nCbLPVVlutbpfkcdKWfSrlum2r1BHJg9RHyZOkNcet3gA99WnKdPalLOdTn/pUXUbyL5JveZ86JuUr5fakk06q8016Uh5TrrO+Kfs5iZrjVdohLajO/p7lt+N+9vmsQysnLUBPYJT5pmy2QDjzTH795Cc/qXVvO7GSNk6WnfxLfie4zO+yr2WabL9sn4svvriWs3bCcY899hhsZ335y18ePHanvKccZ59K3ub434LV1HmtDk+7KenO9Pks+dI6IJI/qT+ynbPtE8TmeJnjZtYl5TnzyjErAXbKftKVdc68c1xO+UqHR5aREw+pl3JsbuUw65b6MdNlHVJ/ZZ/LdkkZTLpyHE1+p65pJ+9be/TAAw+s2yHTZlu3dl/q1dRfyeOkJfmY8pN5J405fjc5dqZjIe2G1BWpDzO/iQnQX0g911t3ZZ/OK9sqsl5pL+YYkBOive3Z3mNH+10+Szs8ZSQnMq666qqah73HnMj6p7ylPk0dkLKb8pRyEO0kb7ZN6tscg1PvM/kJ0JlsUqm0Hsa8Ujml8hk3blz9Pg3QVLA5A9ekskmFn97zSIWXM5zN8F719BKlN7JXDvQ5WPz3v/+t79MwzpnzkaQCzgEgB65M19urmTPOOcD09ozlQNMboOeAloNKPmtSUeYAk2mStqS/dx2bHBRTaTapFHNQHy49cb0nDrK83hMQwyVwapXncC343GeffQbXO43SfNbbkzlcKvY0BHNwyYEl0+dg3NvQaY26toybbrqpNiJz4Eqw2eSAk3weKU8mNv2thzGv1nvXtnevHDyS5qQ/QWjOGic9mb6353SkZaT3J3mfZSVYzNnpNFjScEuDKcvOAb1tuzTW0lBJ8JOylDPRWXbSlfklLWk85P+tfA+XxmIOdr1n63u3QdahacvI/HIwT362E0PZP9JDk+XnoJvPcuBO2lrjbbgcWFsQ0HtSLNsxn7WymRMRyYf0Avbmc/KpXUHQAvR2pUzKRuqC5H3v79LQyUme9rv06OQkQm/Z6F3vFqCnQdSmT69M9re2j7TtlyC8ScMnac4JmSYBWRqKachk/ukBzPZOvrbtmmAj80rjO3mX9UjQ0LZrJHhKo7/VE5lnArXhMt/W69oa5OmZSbCTfMn+lDzKiYImJ6B69822j7Vt2tvzNrzuzffp6cxy0xju3Vd7e1TSWMzfnGBM+U39mxNDqdvS6G/1axpm2catFywN3+G9OElfGtvJyyYBcBp0aQw2qfcTtDSZRxr7vVL/thOCafClbPdeGZXAIulIUJMTMJlHTtqNpDfgbPmTMtp7jMpxJWnPicu2T+dkTvLjC1/4wgT3y+Rr1ieN3EhDPu+zr6f8RALDBCzJg1YW0whPmnLiOOuRMtVO2uT/KVOtodzKfdu3I2U17/O3t9y377ONUzemwd2kcZ9ljbQew/Ort45tr5xgT/DT0pL/t+UmaM3f7A/ZNm2a1rOakw4JJPNZ9ul8nnq0HR/bOrZjbeqO5EVv+WrrnHVMMNCu6ujVerGzjtm/hl+BlpMUvftC6vneK7giQUvWt8kyc5zs1bt9U59n/bJvtLoqgXpvvdm0fb63nmz7ca44y3zSEZF2Ur5LG6G1iVo5zzRpG2UZ+X8CrlaeWw962z6tnZVjWNs+w9tZbX9ox8vkXfIw7Yz8Jic/IwF99uXUh+n5bvV6TvhkupxEaOudabKczLtdyZNjf5bbrrZq6UidmfzOeqd9k3zIsS3lNXVjTtL3bovkVyuHmXfKT+qczH/4VZBNW8ecWMy+2E7IZ52yj2Q5CUKzTimbvftITkKnPmpBck4epI5ssl2yHhMToL+Qeq53PZ5LW68J1YGtTn+uY85I+0fKVNryvcfgCeU7k49B4pis3vKWt9RRafO64oor6kBkb3vb28rf/va3eu9dBoyZY445Bu8RzD2Tuc839/XFZpttVkdu/ec//1nfn3LKKWWjjTYavNcv88hAMe33eWUZuefp1ltvHUxHBuIZ7vTTTy9HHHHE4D2Tf//738u///3v8rrXva7O8+abby6vfe1rh/zm9a9//eD///vf/9Z0nn322fUezbb8gw46qDz88MN1mgzOlHXPfaKrrbZavU8065jpcn/n8HvxV1555fHS+YlPfGLwPr7cK577SzMg0YTstNNONQ25p3TfffcdcVC+HXbYoebpT3/608GBiHI/a9Y7ac+gSrkHMvcc5n6z3Nv6ve99r96jfvnll9f7M59rwKD11luvvPKVr6xp+P73vz+Y/qzzI488UpZeeuma1nPPPXeS07/WWmsNlqvcJx4ZyCz3TGZ9UqaSxve97301zTfeeONgXuc+unz3pje96VnTn3KXQbiy/rkf809/+lPdfhlc74EHHiiPPvpoLWMt/bk3L/dHJh1tXV/zmteUWWaZpb7fZZdd6jgMseCCC04wzzLQV5az3HLL1fvgU9ZTXm666aZa3j/0oQ/VcQOyPm3govw/y8/Adk2mz/Iz4FeTbRxveMMbxlv2/7Uf/v9t0Mpz23fa+ATZJ3KvcdarDfyUfM4+13uvcvKt937TpC3bPYNFtnmnvCUf28jV3/3ud2seZt1713v4fpKBzyLLzX547733lu985zvljDPOqNt3+D6f/TRpziA5KdvLLrtsTVu2UbZX5p/ltHxJvvXmV7ZX266pn3q3a7Q8TjqyX62zzjojbt82v973WW4GGmt1Ueq4Vv+1fMg0I2nbLOU+y0x5SdnPPCJ5OyFJc/bvLKNtnwwemf0y5SxpuOiii8pxxx1Xt/H6669ft3EG/8vf7BttO2cwuY9+9KN1Xx07dmy9j73tv7lvOMvovW8107T8fK68ifxNucto6U3mn3Sknk6dmsGTUvcnLV/72tfqYGKTcoz6yEc+Utcz9XnK3W9/+9ty7LHH1u2dvMl+mf0u65n8HZ63rcwkrUl77i9v65i0pl7NMaaVxQzCF6mLsrys2znnnFPHbsj73nute5f1spe9rH7fBlvL+iefW5lJHuS32267bU17776T9Gd7jLQe+X/bL7/4xS/W9CVfTj755Hrcm3POOev/W72ZdOdY2eS43uqfrHsrYylLWbfMM/tob1qy3bKMHGuWWmqpwe2etGT/fOtb3zo4fera1OPNH/7wh1r+H3zwwSHH/0yTfTPzT92c//fmX+8xPLL8ffbZp5bPVp/ls9RVvb/L9u+V9GVgv9Ql7373u2u9k23S8ruV5+xPEyrXad9EykOcd955dT4ZBC/jfsRXvvKVWsel3ZNtknx/6qmnat2ffTMDRPaWlbZ+bfu0dtZKK61UP//c5z432M5KWU+98YMf/KCuS8pE6vlrrrmmPP7443VsgLbvJ28zhlDqzNRzqesPPvjgug9mcMn4xz/+MVgnZT5ZTsbKyPpkG2cbZl0y3kLmm/KR96kzUxel/ZVl5rjwox/9qPzwhz+s+03ydNy4cXXfjhwDWznMtkoeZ2yYV7ziFYPtuQzw2lsuIm2XjL+TfSiDabZ1yvxTB6etlXZOPu+tn3r35exnSWPvMTTr2budU9Z7l927D76Qeu7ZpD2baVMWetfr2WR5KVOtLhrpmJPyc8ABBwxZn6233rrWr9k/2jE4+1s+zzE45ZPJT4DOZJWKJkFaXquuumo9QOfA9+1vf7tWvAlIW6DVXmkAJrCN/CaVboKwVALZ+dtBLTKPNER6f58KJcFUftebjl6XXXZZnc/b3/72epBKwzKNnEyXhl7+H61R2Ru8tP8nPZEDXw7QbfnXX399baBGKtsclLIe+S4Htxw08/80pFLRD8+v4TbffPN6kEua00BK8NYO3iPJQSbT52CbRkwCla9//etDpslBLA2XBIR5xE/kwJX1TgCTA3aCyRwIU/EnGG+B/MQOFJQGRRr4aTD0pj8VeRqqGSE5efj+97+/vPe9752k9Ocg0cpVazjlYJQBCNN4SGPyqKOOKvvvv3/9LgF2y+s0DCZGmy6DbiWgSMM2+ZDtlwZL3qcxlfTnBEb7zYQGxEleDw9KhktwdfXVV9eTOWm0ZP5Zl+RjBu7KQG1Zv+w/aTSkURFt3UYqr63x91zSQG7rnMZa7z6VNLRR6/P/dpIkr5Sl/I0MKDYhmS5p7p1v9rs0CNLASQMsjeuUjTRwEiBlmqz38P2kScMiDZcERymXn/rUp2rDa0L7UhqUqUPa/BOcJagYPv/hA5217TrS4Ei9+dt7MmRSR9jdc889y2yzzVYuuOCCmidJ54Sk3CdNaURlgLeU++RBykfysQ041ka7Hknbj9t2T3lr5ablR8p99sOcAMi2TeMv6Uv92gYxSz4n37Ntf/3rX9fymQZ+239Tllow3zzbfvJ85SRm6pise+qsnBTNQHkTe4zKNk9dkzKZcpf/5xiR41FOiGa/zGB8Wc8EdKnbex/3NaEyMzFSnyW4SdCT+Q8foK4dayJ1QdKXv7HFFlvUfG71eOqZ5Hka1anje8t2ymcbVHD4erzrXe8a3C8T3GQ/zzSpj/M3eXrVVVdNsFz2rn+rR/I3eZv1SVnqDSRT/nIyIsfH7I85HrbjadISm2yySf2bBn8CsBz7eo/9mX/qzN46JXmYY3DWMWlKeR2+rXrlZGvq8hyzc8K6neRdd911h5yEG37cS12VjoakPSc0c3Ls2eqqkbSAM/mafSR1X/b91o5JunIMzr6Uk5JJT4L4LCeBefaz/C7lfEL1fG87K/VBju/5f/LlW9/6Vq03Ug9mnY8++uj6m1b+kucpM2kjZVmf//zn6zEp88sxMGWpbdO00dJuaycC8jfLSR2d7ZQTOO3Eaj5LWnLiPvmd+isBb8pyAvucoMkxMHVz6qTMO+WjHXuyrVo5TFlPWyLLuPDCCwfbEin/veWinVT5f1cK1xMcbZ2yzVIH5zie+jCBZ28dPKn1Vfaf3mVnW01Jve3Z5Efver1QKT9pR/WuT+qbHANSHtsxOAPIZn/OsWDttdee6HYHE0+AzhTVRgBNgyOBTnby9E61YKu9ehuWqXhSYf/sZz+rv03g12QeOVAO/31ezzYq7KWXXloDu1RiaYilUs7BNlIxtx7e9qio1htz5ZVXDjaWElSk4k2llOl6l9175jIHlYceeqiuRyr8jJaeM5QT+xiq9DjmsRppgObM8sc+9rHn/E0qzRwk8qigz3zmMzWgey7Jg6x3HrXWGvgZ5T0H1qx/vmvB7vDe0mgBa3PooYfWA24C//SE9KY/eZJHcSVdafi1wPOFpD8HjaQrr/w2AUULqNr2aGUi0+Rg/mxyIiGN3iw7vQk54LSTAWms5ZV1SPpz5j89Er3SIE1jv32eRmcLctNTNiGZb8pi1jtn65P3CcTSkL3vvvvqmfH0eqTxObwXrzWs8pv07GT5KbNNC1jSkzBcynM7u59GUyvL2abppWqN1ZaHKRsJcnPyJFfHtLI6IdlPctIn69e7r+RqjvSUpHwkUElA8J73vKc2QtP4SwA6XO9nyYcEl8nbXPXQTpb0ysm6pDtBTabL/HOyJdO2PEx+tfUYnl9NGngpC73bNQ2WfNaCvTRCJ/Rosswv6W37Sd5nuZEeoQQLuYoggUnyI3VbAovWy932seRzArE0ilKXpTynpy2N2uxbabC1+ja/yd/efTOybyTwSR2cxm+uzIk0tnJSLXVbRidPWUx5y0nI9LAPr19Tt6UX5bDDDqtXl6RRmLqt7b+pL1rd+myG53Vv3uRvApZWB0XqqaxDb89VgoKc6Eie5HFlCTBGMryuiuHHo5TX1NPteJRym6At65ntnzKYIGe4pLXlf29aE0im566Vxd6TFikTWZds6yw/+2K72iba9m/1UtKUfTISvCWfWw9+ArAc01KHDl/HLDdlZaT1SPrats3yk57U8QmYUpfnpHD2mXaSMfNOINDkpGrqp17ZR3JSOnmSOqe3Lmrrl/KadUgdEMm71rhP0BvZL7Mtk+7ebZ1AK2V4+LE/+ZxpE3gm8OvdVr1piKx3lpP6NgFzfpOTJe3Z9pH9p/ekZyRfsqzUJdk/sj7Jg+Hlefhxvrdctzo19VJOKKS+S13WymC2Y8phenWzLfL/5Hl+n3U85phj6u/Tk56TEr3r17ZPK9fZT1O/pA7LvJP+Vm9k/TPvdqVi64lOfZwAvbeNlGNM5pdjY45PWe/UeTkWZj/JfhqpO9qJr0hQn/KdMpiAuvVSJ+9Sn2X52U9ysiMnJVIHtasqkmetHEbWvZXDzCfLTr2ectvaEsPblZH1zTE82/VLX/pSTXdv3ZQTDjkGZv1bHTxcy5veY2jKfe92Tjp6l917cueF1nMj1V3P1p5tRvpdlpd6qLcuGn7MSfnJMWGkNnbbP7Ltsu+kUyTH4OzDaY8xeQnQmazScE0FnleCjFwKlTNy2ZkTsKYST8MvDf0EINm5U0HnTHKT6dJ7kd6vnB3t7aXK5VqpnNLIT0M5B46f/OQn9f2zaRVYGo85QGU+uaQqDYM0XJKmpDVBSCrk9CwnXUlHb8MpZxZzQEyllst7chIhjdTW45sz0jnw5gCUBvA3vvGNus577LFHbbhMrAQtSUvyMEHvs9l5553r5fPJz6Q3vVrtANAkWExvdhpoaYxEzqZnvZM3ycvkRdKZz7IdW4WbnrOkIY8CyjRtuyVf8v926WV+c/jhh9ftl4NQgvOkP4225EkaHDmopWGWhkm7bWFi0j+SHKCSnhw0Mv9cJt+2Qzv7nQNyDn4JbrIOCa5b+nPVQK+sTy57TfCSA1YOQpl3LkVO+cw800hI+nPAHt57mgZJGnYpF5GezRa45Wx9yttwOfudg1waF+k1SJnPQTVXnqSBmAZdGmPJ4+Tj8GfzJg1p8Gb+aeAkPxKwRPIx2yMSmOZMd7Z/tkMaejnJkB6NNMDSK5SgK5/n5FDWuzWO2smqNGSSvlxO3A78vQ324dKQym8yv1x6mkZz9t008rIv5kRItnOCjdaDlB6ZkXouLr744nqy65BDDqkNvKQ9J4FSpkfqxU5jO/tlpks5TCMuJ47yvjWeElCm/khjK9s1AcHw51qnlykN3eT9Bz/4wXqyJL1uLVhq2zZ1SrZj5pcy3MphGlvZPll+0p6yk+Wmvko60ghL3Zj1y/6Qq2cSuLfL9VN+U16TP+khyj6WZWU7530alLlapzfP8pucoEq9m7zNOud32abZt1M+c9Ks9bakkZ3GWdKV/TXBQ/aD5EvKZeafBnHrlUog2E4qtcZ5ltH23+xHE3rEWa/kReqU1AmpbxOUtXo8dUjKfvbZBCIpy6mfE9jlRFiWlXKehmHSkfKVvJ9QvZE8SdlPwJL1Tv4NPx5lG2c/T72fcpDtmXXO/BMwJY+HX74c2a/S4G3zz/Eo65OrXZLnrSwmjSlLyZ/kVz5PmpOXCbZT52TfSOM4Zb2dfEtwk3KeICZSNySfM49I2crvclzKeiWAzXc5EZXAJfmbei/7avJ7QuuR32VfSsM++34Ckqx3ezZ8At8Eae1S2Fy5kjqy9wqlXBGSbZZ9PulsvfzZvxJw5YRPOxYkzxI8pi5NHZMTqKnDI+nMcnpvHUkwlV7cnDDPPpf6NWU1y2tX1aUOy7Ek9VPKT45xOQ73pjH7dPJ9q622qvtyyna2e+9tJflt9qFsq3YSImU6ZSHLSbrbpfRZp5SfBEuR+aRNkXxPeyDlvF3l0yTtWcfkTyuDSWuuXEjZy605aWsk33KcSlpyrGhBYspHtm2+T50UuWog88u2TVpTP2Z7Zroc71p7J3VTO6mX9kmkrKWtlToj+ZuTGym/OSZl+ySNmTbT5biUaVJXp+xlf2m9/U2WmXZGlpO8Svlv7cLUo6k3sk5pe2RbZt7ZJjlOZZ9JvZNy2OrR1A2tHGZ7pvzkkX7ZZ7KOOT7m9zme9V4ann0s6Ugas3+mLmu95Klrki+ZR9oI2X9Gqj+SlykrubIu+ZL6KCcHJvbRcy+knuutu7Je2a+zDZNvyb9sn+z7qauGX+nSe+xox4CcxMqJ8NbGTlqGH3NylU32+5S/bI+U5yynlbOUz5xUTnpzgip1U+qB4beEMBlMxvvZ6XNtoKL2aiNsZmC2JoObZFTvNnJ0BojKYEUPPPDAkHm1kUkzUvFwGcwiI8Vm4I8MNJKByXoHJWuPvBpul112qYPoZACODNqTgT/y/4ysmREuI6OwZ6Tu9oi0pDODo/QOXJOBp9rotG2E6Qymk/9ncKAMONMen5R5ZPCfLCPr3TvYx7M9piODoGQ9Murpc8nARhngqT3aJgPdZTTnaAP6ZNTlPOYmaW2DrOV3We8M1pPHfLQRXPM3A5fkUVxtgJ4MkJX37TFrGbk1A33l8TRtpOqse2+asu5ZxwyE0gbhyQA/GYQkoyJPSvp78yGDCrVB0I444ojBEWXzaoN3ZXC/lrcZpKWNzp20tvRnxNfeZWSaDPySUWKT7kyXbZb0t0dTJY1JfwZjyfwzCEsGY2llLiML57NMm8F12gj3GdAs+Z73wwc4zLIy//aYtaQjIxrn84w8nsGKstxslzaqbxtkKMvKADsZKCnzz/v2mKAMsNSWn4HBMthb5pPynYG82uBSyc822Fxe7VF7baDFDILYRmFur/b4pAxW1/uYteED1GTgnYzmnUGBkv7kbQZtyoBZ2VYZeCxlr+V7W++27Xofs5Y6IftUpm2DWGVU5wy61TtYVpMymzqpPU4p2zOD//TOP4MRtf00A1q1EcczCGQbhTuDxfWWsQyWM/wxa3kKQgZmyrLaY2eyHTJSdMpi7yPIMlBSHsOU9c302W75O9IjbzLAXvaxlt8ZCCjr1AY+zHwzunAbWTt5kDKYOrENuNX7mLWMJJ28zP9bvrQRlPM3dUQGxWvpbY9pyvdtP8+8sx5tNOekpXf/TZ3eBsWaUF3X6ovU4/ld9pXTTz99oh8/lDxK+UxeJ23J6wyE2QamGz5AUgYgy7La6PutLu89HrUByLKcjIyeNLf1zDq39LVB4nrXJ4PVtePGSI9ZS1lM+c02aY9ybIMytbzJ6NdZ13yXspgB5No2yKttr/YYzKxPK1PZNhmMLPt423bJg4wK3dYtn6XMDc/nll+9+3dvXZARovP/DDCV7drSkTol2z9lv9XJWX7Ke7ZJltvqzeRN6syUrd5jQfbf5G3yOJ+3R5Xllac/DB8ILwNTZT7t6RVZ7+RVG/0788ky2v9TRpKP2Sa9A3xlRPA2j6Q1dW5vecn+3x69l7IVye/Mp9VVbSDBbM8MmtWeYJJ9O/tx2gCZfvio+W2b5W9GyW5lMHnU6t6W9ykLmbYdH/LK8pOmVl5Th2X6tEuyfdqjIJMvmaY9PjDlOoN/ZZ/JPLLOmTa/zTyy72Rg1N7B/jLgXvKhpbFt+/ZEl/w+gyy2x6xF6qcMBJhjTJbRjmttvinjyZcMsph2QHtkbSvbSVvaSxmVvz16MPtbK4f5fys/GXAtZShpST2UdU6etnQkTanzk472WLGUs6xTq4PbgIZph/QOBNq2e6TeyUCiWU4GGT3ssMMm+jFrL6Se6627Wv3b6q62Xm3fT5p7B0McfuyYlMespSxnX26Px03d1UZqT9sj+Z7P2zE4o9sz+Y3JP5Mj0IcXo/S8pre3DXo2IelZzABDw++9fL5yJjuXoaWHrN2XNy2Z1tPPlJWesVzNkjP/zya97rkPNb1/6VWalqQHI1eH5MVQ6dFMj096Wpk2pdc5V1O0AQtHkitIMl5Fejh7x4iZmnLFSHpc08v9YjShPJ6Y7TOp0pOcKwZy5QHPTT3HCzFxoz9BH8jlagmyc4lzLuXKpXg5uGWQluFyL2gul8ql7DlAZhCT57rMfmLk8sNcjpTLWHPJaxs8Z1oxraefKSuXi+aywryyDwHThlzemxOvuUQ2l6Tvvvvu9SRUBohqEozkkuBcgpuAMZd2Z6TpqRmc57aenFjP5c25HDq3ir2Y6poJ5XEuv88lyc+2fZ6v3LqRW95yG5DgHKYOATr0nO38+c9/XnvDcz9i7uXKPVa9A9U0ud8p98vlXqvco50BZ9q9vy9E7itKj2EGRcm9PhM7gnpXTOvpZ8rKQE8J0jNgz0j3wQLdlHtf99prr3rfaQZay8CiGcy1dzT33GOf8V1yHMh9yzl25mTt1JT7anMPeNKSAfZylU7GdHmxmFAeZ2T459o+z1fukU++ZpyKnPwApjyXuAMAAEAHGMUdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAwDdtvv/3KiiuuONrJAAAmAwE6ADBZn5kNADw/AnQAGGXPPPNMOeyww8orX/nKMvPMM5fFFlusHHzwwfW7z33uc+VVr3pVmW222crLX/7y8oUvfGEwCD7xxBPL/vvvX37/+9+XMWPG1Fc+i/vvv7984hOfKAsssECZc845y1vf+tY6Xa+DDjqoLLjggmWOOeao0+6xxx5DeuOTrgMOOKC87GUvq+nKd+ecc87g97fddltd5umnn17e9KY3lVlmmaUcd9xxdXk/+tGPhizrzDPPLLPPPnt56KGHpmheAsC0bIbRTgAA9Ls999yzfPvb3y5HHnlkWWuttcqdd95Z/vjHP9bvEjwn6F5kkUXKH/7wh7L11lvXz3bffffygQ98oFx//fU1aP7Vr35Vp59rrrnq3/e9731l1llnLb/4xS/qZ9/61rfKOuusU2655ZYy77zzllNOOaWeBPjmN79Z1lxzzXLaaaeVr3zlK2XJJZccTNfXvva1+ll+u9JKK5Xjjz++vOtd7yo33HBDWWqppQanS2Cf6TJNgvScCDjhhBPKe9/73sFp2vukHQAY2ZiBgYGBCXwHAExh6VFOL/c3vvGN2ov9XA4//PAaTF911VWD96Cnd/raa68dnObiiy8uG220Ubnnnntqz3eTHvoE9ttss01ZbbXVyiqrrFKX2+TkwMMPPzw4r5e+9KVl++23L3vttdfgNK9//evLqquuWo4++ujag56A/qtf/Wr59Kc/PTjNFVdcUdZYY41yxx13lIUXXrimI/PKSYT0tAMAI3OJOwCMoptuuqk8/vjjtXd7JLl8PD3cCy20UHnJS15S9t5773L77bc/6zzTg51Ae7755qu/aa9bb721/OUvf6nT3HzzzTXY7tX7/sEHHyz//Oc/67J75X3S3CuB/vD5vPrVry4nnXRSfX/yySeXxRdfvKy99toTlScA0K9c4g4AoyiXoU/IZZddVjbbbLN6n/kGG2xQL1Vvl6I/mwTn6bn+zW9+M953c889d5nccm/5cLkaIL3sufw9l7d/7GMfq/erAwATpgcdAEZR7uVOkH7++eeP992ll15ae54///nP117qTPu3v/1tyDQzzTRTefrpp4d89rrXva7cddddZYYZZqiXtfe+5p9//jrN0ksvXa688sohv+t9n4Hect/7JZdcMmSavF9uueWec70+8pGP1LQeddRR5cYbbyxbbLHFROYIAPQvPegAMIoyqFpGas+94Qm2cwn5v/71r8GB2HI5e3rNc9/32WefXc4444whv19iiSXqpeu5bzyjrWcQtnXXXbesvvrqZeONN66jw2cU+Fyunt+/5z3vqcH+jjvuWAecy/9zv3gupb/uuuvqSPHNbrvtVvbdd9/yile8oo7gnp7wLCcDzD2XeeaZp2yyySZ1Huuvv35NGwDw7PSgA8Aoy6PTPvOZz5R99tmnLLvssnV09gyslhHTd9lll7LDDjvUADk96pm216abblo23HDD8pa3vKUONvc///M/9VLyn//85/We71xangD9gx/8YO3RHjt2bP1dLp3P6PGf/exna497gvwtt9yynjBodtppp7LrrrvWtK2wwgp1tPif/vSnQ0ZwfzZbbbVVeeKJJ8rHP/7xyZxjAPDiZBR3AKBab7316mB03//+9yfL/DKfnGBI732uDgAAnp1L3AGgDz366KPl2GOPrYPPTT/99LXnPY9BO++88ybLvPMs90MPPbRsu+22gnMAmEgucQeAPtR7GfzKK69cfvazn5Uf//jH9f71Fyr3vS+zzDK1Nz6X0QMAE8cl7gAAANABetABAACgAwToAAAA0AECdAAAAOgAAToAAAB0gAAdAAAAOkCADgAAAB0gQAcAAIAOEKADAABABwjQAQAAoAME6AAAANABAnQAAADoAAE6AAAAdIAAHQAAADpAgA4AAAAdIEAHAACADhCgAwAAQAcI0AEAAKADBOgAAADQAQJ0AAAA6AABOgAAAHSAAB0AAAA6QIAOAAAAZfT9fxGzTOaAt15cAAAAAElFTkSuQmCC
    # '''
    # img_data = base64.b64decode(base64_str)
    # img = Image.open(BytesIO(img_data))
    # # print(response_stream)
    # st.image(img)

            # with st.spinner("Waiting..."):
            # print("system response: ", full_res)
            # if full_res:  
            #     st.session_state.chats.append({
            #         "role": "assistant",
            #         "content": full_res
            #     })

    # with chat_container:
        
            # else:
            #     st.chat_message(chat["role"]).write_stream(chat["content"])

        
  

    

