# 🧠 MySQL AI Agent

A lightweight AI-powered agent that understands and interacts with MySQL databases through natural language. Built to support developers, analysts, and students in exploring and querying databases using plain English — no manual SQL writing required.

---

## 🚀 Features

- 🔍 Natural Language Interface: Ask questions in plain English and get meaningful answers back.
- 📊 Database Insights: Retrieve data summaries, visualize results (line, bar, pie charts), and explore tables effortlessly.
- 🔄 Multi-query Support: Handles follow-up questions and context-aware dialogue.
- 🧠 LLM-Powered Responses: Uses Google Gemini + LangChain to return human-friendly explanations instead of raw query outputs.
- ⚙️ Customizable Schema: Connect to your own MySQL database with ease.
- 💬 Conversational UI: Clean chat interface for seamless interaction.

---

## 🛠️ Tech Stack
- 🧪 Programming Language: Python
- 🧠 AI Framework: LangChain + Gemini
- 📊 Database: MySQL
- 🌐 Frontend/UI: Streamlit

---

## 📦 Setup Instructions
1. Clone the repository
- git clone https://github.com/nvhnam/mysql-ai-agent.git
- cd mysql-ai-agent

2. Install dependencies
- pip install -r requirements.txt

3. Configure Environment
- Create a folder named .streamlit in the project root (if it doesn't already exist):
<pre> mkdir .streamlit </pre>

- Inside the .streamlit folder, create a file named secrets.toml and add your credentials:
<pre> gemini_key = "your_gemini_api_key" </pre> 
<pre> default_db_uri = "mysql+pymysql://username:password@host:port/database_name" </pre>
  > 💡 Make sure to replace the values with your actual Gemini API key and MySQL connection URI.

4. Run the app
- streamlit run app.py

---

## 🤝 Contributing
I'm happy to hear your thoughts or ideas! Feel free to contact me with your ideas or collaboration.

---

## ⭐ Support the Project
If you find this useful, feel free to star the repo and share it with others who might benefit!
