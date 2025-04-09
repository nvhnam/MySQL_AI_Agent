import streamlit as st

from db_connection import connectDB
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
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

template = """
You are a helpful and intelligent MySQL assistant agent tasked with answering user questions by interacting with a MySQL database. You can write SQL queries, generate Python visualizations, and explain findings clearly — unless returning a chart, in which case you must return only the result without any explanation. Carefully review the table and column names before answering and you can use the following tools to retrieve or visualize information when needed:

**Available Tools:**
- `list_all_tables`: Use this to list all available tables in the database.
- `get_table_schema`: Use this to get the schema (column names and types) of one or more tables.
- `run_sql_query`: Use this to run safe, read-only SQL SELECT queries and get results from the database.
- `save_memory`: Use this to save important facts, conclusions, or context from the current query or response, so that they can be recalled in future queries.
- `plot_chart`: Use this to visualize SQL results using Python code. You can load results (a dictionary from your database query) into it to generate plots (bar charts, pie charts, etc.). Always return the result received from the `plot_chart` tool directly to the user.

---

**`plot_chart` Tool Usage Guide:**
Use plot_chart when the user asks for data visualization, summary insights, or trends.

Arguments:

- data: A Python dictionary from a SQL query result. Must be passed as a dictionary, not a string.
- chart_type: (Optional) "bar", "line", or "pie". Choose automatically based on context:

Use "bar" for category comparisons.

Use "line" for time-series or trends.

Use "pie" for composition or percentage shares.

- x: Column name to use as the X-axis or labels.

- y: Column name to use as the Y-axis or values.

Automatically infer chart_type, x, and y from the SQL result if the user doesn’t specify.

The plot_chart tool returns a base64 image string.

**Important:**

When returning the output from `plot_chart`, return only the raw result from the tool — no explanations, no descriptions, no surrounding text. The image will be rendered separately.


---

**Workflow:**

1. When the user asks a question, **autonomously determine what information is needed** to answer it.
2. If you're unsure about available tables or their structure use `list_all_tables`, `get_table_schema` tools to gather the required context without asking.
3. Use `save_memory` to store important details that might be useful later.
4. Automatically retrieve necessary data rows (e.g., product nutrients, purchase history) via SQL queries using `run_sql_query`.
5. If visualization is requested or beneficial:
    - Format SQL result obtained after using the `run_sql_query` tool as a Python dictionary.
    - Choose appropriate chart_type, x, and y if the user does not specify.
    - Call `plot_chart`, and return the tool’s output directly and silently..
6. Construct **safe** and **read-only** SQL queries.
   - NEVER include any modifying queries like INSERT, UPDATE, DELETE.
   - NEVER access or reveal sensitive info like emails, passwords, or user IDs.
7. Return a friendly and clear summary of your findings — statistical, or textual but suppress text when returning images.
8. Do not ask the user for permission to use tools. Just use them when appropriate.
9. In follow-ups, refer to past queries using `save_memory` content as context.
10. Ask the user for clarification only if absolutely necessary.

---

**Guidelines:**

- Always prioritize data safety, privacy, and user clarity.
- When a vague reference is made (e.g., "that product", "it"), refer to previous conversation turns for context.
- Use `plot_chart` when the user asks to visualize data (e.g., bar chart, pie chart, line chart), explore trends, calculate averages, or summarize values.
- Transform SQL output into a dictionary from a database query then load it into a DataFrame to draw, plot and then return the figure to user.
- Never return full tables unless requested. Always summarize clearly.
- Be safe, precise, and avoid assumptions.
- Never expose sensitive data.
- Avoid guessing: If you're uncertain about the table or column referenced, use the appropriate schema tools or ask the user.
- Always summarize the result clearly and concisely for the user without saying your steps but only your final results but suppress text when returning images when using the `plot_chart` tool.
- When returning charts using the `plot_chart` tool, return only the tool result, silently.
- When you are done, summarize your findings or results to the user. Avoid not returning anything.

---

Your role is to bridge natural language with SQL and Python for data retrieval, visualization, and interpretation — either in text or in charts. Be thoughtful, clear, and helpful in every response. Use tools when necessary to deliver insightful answers.

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
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL

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
    if state["messages"]:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, HumanMessage):
            key = f"memory_{len(store.search(('memories', user_id)))}"
            store.put(("memories", user_id), key, {"data": last_msg.content})
            # print(f"Saved to memory (auto): {last_msg.content}")
    namespace = ("memories", user_id)
    memories = [m.value["data"] for m in store.search(namespace)]
    system_prompt = template.format(
        dialect="MySQL",
        top_k=5,
        # info=formatted_info
    )
    system_msg = f"{system_prompt}\n\nUser memories: {', '.join(memories)}"
    
    messages = [msg for msg in state["messages"] if  not isinstance(msg, SystemMessage)]
    return [{"role": "system", "content": system_msg}] + messages

# def debug_print_memories(user_id: str, store: BaseStore):
#     namespace = ("memories", user_id)
#     results = store.search(namespace)
#     print(f"\n--- Saved Memories for user {user_id} ---")
#     for r in results:
#         print(f"Key: {r.key}, Value: {r.value}")
#     print(f"Total: {len(results)} memories.\n")

# python_repl = PythonREPL()
# repl_tool = Tool(
#     name="python_repl",
#     description="""
#     A Python code execution tool. 
#     Use this to perform any data processing, analysis, or visualization tasks using Python.

#     Typical usage includes:
#     - Loading SQL query results into a pandas DataFrame
#     - Performing calculations, aggregations, filtering, or formatting
#     - Creating plots using matplotlib or seaborn (e.g., bar charts, pie charts, line graphs)
#     - Returning summarized results or printed output from the computation

#     **Input should be a valid Python statement or code block.**
#     If you want to return a value or see output, make sure to use `print(...)`.

#     Examples:
#     - To inspect the first rows of a DataFrame: `print(df.head())`
#     - To create a bar chart: 
#     ```python
#     import matplotlib.pyplot as plt
#     df.plot(kind="bar", x="category", y="total_orders")
#     plt.title("Total Orders by Category")
#     plt.show()
    
#     If SQL results were returned as a CSV or markdown table, load them into a DataFrame first using pandas:
#     import pandas as pd
#     from io import StringIO

#     data = \"\"\"category,total_orders
#     Fruits,25
#     Vegetables,40
#     Snacks,15
#     \"\"\"

#     df = pd.read_csv(StringIO(data))
#     Make sure your output includes print() or displays a chart with plt.show() if you want the result to appear. 
#     """,
#         func=python_repl.run,
# )

from langchain.tools import tool
import matplotlib.pyplot as plt
import pandas as pd
import io
from typing import Union, Dict
import base64
import json
import textwrap

@tool
def plot_chart(data: Union[str, Dict], chart_type: str = "line", x: str = None, y: str = None) -> str:
    """
    Plot a chart using the given data.

    Args:
        data (dict): A **Python dictionary** where:
            - Keys are column names (e.g., "category", "values")
            - Values are lists (e.g., {"category": ["A", "B"], "values": [1, 2]})
            - ⚠️ Do NOT pass this dictionary as a string.
            - ✅ Correct: {"category": ["A", "B"], "values": [1, 2]}
            - ❌ Incorrect: "{\"category\": [\"A\", \"B\"], \"values\": [1, 2]}"
        
        chart_type (str): Type of chart to generate. Supported types: "line", "bar", "pie".
        x (str): Column name to use for the X-axis.
        y (str): Column name to use for the Y-axis.

    Returns:
        str: Base64-encoded image string (PNG format) to be displayed in Streamlit.

    Note:
        If `data` is mistakenly passed as a string, the function will attempt to parse it.
        If parsing fails, a helpful error message will be returned.
    """
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            return "Invalid stringified dictionary format."
    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    df[x] = df[x].apply(lambda s: "\n".join(textwrap.wrap(s, width=15)))


    if chart_type == "line":
        plt.plot(df[x], df[y])
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
    elif chart_type == "bar":
        plt.bar(df[x], df[y])
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
    elif chart_type == "pie":
        plt.pie(df[y], labels=df[x], autopct='%1.1f%%')
    else:
        return f"Chart type '{chart_type}' not supported."

    plt.title(f"{chart_type.title()} Chart of {y} by {x}")
    plt.xlabel(x)
    plt.ylabel(y)

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    res = base64.b64encode(buf.read()).decode("utf-8")
    if res:
        print("Tool plot_chart finished")
    return res


memory = MemorySaver()

def llm_agentic(llm, question):
    store = InMemoryStore()
    tools = [list_all_tables, get_table_schema, run_sql_query, save_memory, plot_chart]
    agent_executor = create_react_agent(model=llm, tools=tools, store=store, prompt=prepare_model_inputs, checkpointer=memory, state_schema=AgentState)
    config = {"configurable": {"thread_id": "thread-1", "user_id": "1"}}

    for step, metadata in agent_executor.stream(
        {"messages": [HumanMessage(content=question)]},
        config=config,
        stream_mode="messages",
    ):
        # if metadata["langgraph_node"] == "agent" and (text := step.text()):
        #     # print(text, end="")
        #     yield text
        if text := step.text():
            # print("text: ", text)
            # print("metadata: ", metadata)
            yield text, metadata

# print(f"state[`messages`]: ", {AgentState["messages"]})