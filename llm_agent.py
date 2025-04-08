import streamlit as st

from db_connection import connectDB
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit


db = connectDB(None)
os.environ["GOOGLE_API_KEY"] = st.secrets["gemini_key"]
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.4,
    max_tokens=1000,
    timeout=None,
    max_retries=2,
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
info = toolkit.get_context()
if isinstance(info, dict):
    formatted_info = "\n".join(f"{k}: {v}" for k, v in info.items())
else:
    formatted_info = str(info)

list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

# from langchain.prompts import PromptTemplate

template = """
You are a helpful and intelligent MySQL assistant agent tasked with answering user questions by interacting with a MySQL database. Carefully review the table and column names before answering and you can use the following tools to retrieve information when needed:

**Available Tools:**
- `list_all_tables`: Use this to list all available tables in the database.
- `get_table_schema`: Use this to get the schema (column names and types) of one or more tables.
- `run_sql_query`: Use this to run safe, read-only SQL SELECT queries and get results from the database.
- `save_memory`: Use this to save important facts, conclusions, or context from the current query or response, so that they can be recalled in future queries.

---

**Database Schema:**
    {info}

---

**Workflow:**

1. When the user asks a question, **autonomously determine what information is needed** to answer it.
2. If you're unsure about available tables or their structure, first check the given database schema {info} or use `list_all_tables`, `get_table_schema` tools to gather the required context without asking.
3. If relevant context or conversation history would help answer the user's query, use the `save_memory` tool to store important details, and rely on saved memories to guide follow-up answers.
4. Automatically retrieve necessary table rows (e.g., product IDs or nutrient values) via SQL queries if it helps answer the question.
5. Construct a safe and read-only SQL query using the appropriate table(s) and columns.
   - NEVER include any query that alters the data (e.g., INSERT, UPDATE, DELETE).
   - NEVER expose or request sensitive information (e.g., passwords, emails, user IDs).
6. Run the SQL query using `run_sql_query` and wait for the result.
7. Translate the result into a friendly, human-readable natural language response.
8. NEVER ask the user for permission to proceed when a tool can help.
9. In future questions, if a reference is made to previous topics (e.g., "that product", "those values", "it", "its"), use the `save_memory` tool to get the previously stored memories for context.
10. If the question cannot be answered with available data or is too vague, ask the user for clarification.

---

**Guidelines:**

- Always prioritize data safety, privacy, and user clarity.
- When a vague reference is made (e.g., "that product"), refer to previous conversation turns for context.
- Never return all data from a table unless the user explicitly asks for it.
- Avoid guessing: If you're uncertain about the table or column referenced, use the appropriate schema tools or ask the user.
- Always summarize the result clearly and concisely for the user without saying your steps but only your final results.

---

Your role is to bridge the gap between natural language and SQL through thoughtful reasoning, safe tool usage, and conversational clarity. Proceed step by step and only use tools when needed to fulfill the user’s request.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("placeholder", "{messages}"),
    # ("user", "Remember, always be professional!"),
])

def list_all_tables():
    """
    List all tables available in the current MySQL database.
    """
    result = list_tables_tool.run("")  
    return result

def get_table_schema(table_names: str):
    """
    table_names: A string with one or more table names, e.g., "users" or "users, orders"
    """
    result = get_schema_tool.run(table_names)
    return result

def run_sql_query(query: str):
    """
    query: A safe SQL SELECT statement as a string.
    """
    result = query_tool.run(query)
    return result

from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated
from langgraph.store.memory import InMemoryStore
from langchain_core.messages.base import BaseMessage
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langgraph.managed import IsLastStep

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: int
    is_last_step: IsLastStep

def save_memory(memory: str, *, config: RunnableConfig, store: Annotated[BaseStore, InjectedStore()]) -> str:
    """
    Use this tool to save any useful or important information shared by the user during the conversation.
    
    You should call this tool whenever the user mentions something that could be helpful in the future,
    such as:
      - Preferences (e.g., dietary restrictions like "I prefer low-sugar foods")
      - Previous queries that may need follow-up (e.g., "show me last week's orders")
      - Clarifications that disambiguate their requests (e.g., "by 'items', I meant 'fruits'")
    
    The memory is stored and will be used in future system prompts to help maintain context.
    
    Only save meaningful phrases or short sentences — do not repeat the entire conversation.

    Example usage:
        save_memory("User prefers vegetarian meals")
        save_memory("The product ID for 'organic oats' is 1234")
    """
    print("----save_memory START------")
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    key = f"memory_{len(store.search(namespace))}"
    print(f"Saving memory for {user_id} → {key}: {memory}")
    store.put(namespace, f"memory_{len(store.search(namespace))}", {"data": memory})
    return f"Saved memory: {memory}"

def prepare_model_inputs(state: AgentState, config: RunnableConfig, store: BaseStore):
    user_id = config.get("configurable", {}).get("user_id")
    namespace = ("memories", user_id)
    memories = [m.value["data"] for m in store.search(namespace)]
    system_prompt = template.format(
        dialect="MySQL",
        top_k=5,
        info=formatted_info
    )

    system_msg = f"{system_prompt}\n\nUser memories: {', '.join(memories)}"
    
    return [{"role": "system", "content": system_msg}] + state["messages"]

def debug_print_memories(user_id: str, store: BaseStore):
    namespace = ("memories", user_id)
    results = store.search(namespace)
    print(f"\n--- Saved Memories for user {user_id} ---")
    for r in results:
        print(f"Key: {r.key}, Value: {r.value}")
    print(f"Total: {len(results)} memories.\n")

memory = MemorySaver()

def llm_agentic(llm, question):
    store = InMemoryStore()
    tools = [list_all_tables, get_table_schema, run_sql_query, save_memory]
    agent_executor = create_react_agent(model=llm, tools=tools, store=store, prompt=prepare_model_inputs, checkpointer=memory, state_schema=AgentState)
    config = {"configurable": {"thread_id": "thread-1", "user_id": "1"}}

    for step, metadata in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        if metadata["langgraph_node"] == "agent" and (text := step.text()):
            # print(text, end="")
            yield text
