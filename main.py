import os
import json
import base64
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sales Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- CONFIG ----------
DB_PATH = "db-sample.sqlite3"
TABLE_NAME = "sales_salesrecord"

load_dotenv()

# ---------- LLM CONFIGURATIONS ----------
# GPT-4o (Default)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set")

azure_endpoint = "https://ai-hassaan-9463.cognitiveservices.azure.com/"
api_version = "2024-12-01-preview"
deployment_name = "gpt-4o"

client = AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# GPT-3.5
ENDPOINT_GPT35 = os.getenv("ENDPOINT_GPT35")
DEPLOYMENT_GPT35 = os.getenv("DEPLOYMENT_GPT35")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Llama
AZURE_LLAMA_ENDPOINT = os.getenv("AZURE_LLAMA_ENDPOINT")
AZURE_LLAMA_API_KEY = os.getenv("AZURE_LLAMA_API_KEY")

# Ministral
MINISTRAL_API_URL = os.getenv("MINISTRAL_API_URL")
MINISTRAL_API_KEY = os.getenv("MINISTRAL_API_KEY")

# Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ---------- SESSION MEMORY ----------
SESSION_MEMORY = {}
SESSION_CONTEXT = {}  # Stores active context/filters per session

# ---------- TRANSCRIPT ----------
TRANSCRIPT_DIR = "transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)


def save_transcript(session_id: str, user_msg: str, bot_msg: str, image=None):
    file_path = os.path.join(TRANSCRIPT_DIR, f"{session_id}.json")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "user": user_msg,
        "assistant": bot_msg,
        "image_generated": bool(image)
    }

    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = []

    data.append(entry)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


# ---------- REQUEST / RESPONSE MODELS ----------
class ChatRequest(BaseModel):
    session_id: str
    question: str
    model: str = "gpt-4o"  # Default model


class ChatResponse(BaseModel):
    answer: str
    image: str | None = None
    chart_type: str | None = None  # Return chart type for UI selection
    chart_data: dict | None = None  # Return data for dynamic charts


# ---------- SCHEMA (cached) ----------
_schema_cache = None

def get_schema():
    global _schema_cache
    if _schema_cache:
        return _schema_cache
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({TABLE_NAME});")
    cols = cursor.fetchall()
    conn.close()
    _schema_cache = ", ".join([f"{c[1]} ({c[2]})" for c in cols])
    return _schema_cache


# ---------- LLM CLIENT SELECTOR ----------
def get_llm_client(model_name: str):
    """Get the appropriate LLM client based on model selection"""
    
    if model_name == "gpt-4o":
        return ("azure_openai", client, "gpt-4o")
    
    elif model_name == "gpt-3.5":
        gpt35_client = AzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            api_version="2024-02-15-preview",
            azure_endpoint="https://machineagentopenai.openai.azure.com/"
        )
        return ("azure_openai", gpt35_client, DEPLOYMENT_GPT35)
    
    elif model_name == "llama":
        return ("llama", AZURE_LLAMA_ENDPOINT, AZURE_LLAMA_API_KEY)
    
    elif model_name == "ministral":
        return ("ministral", MINISTRAL_API_URL, MINISTRAL_API_KEY)
    
    elif model_name == "gemini":
        return ("gemini", GOOGLE_API_KEY, None)
    
    else:
        return ("azure_openai", client, "gpt-4o")  # Default


# ---------- CALL LLM ----------
def call_llm(model_info: tuple, prompt: str, temperature: float = 0):
    """Universal LLM caller that handles different model types"""
    import requests
    
    model_type, client_or_endpoint, model_or_key = model_info
    
    if model_type == "azure_openai":
        # Azure OpenAI (GPT-4o, GPT-3.5)
        response = client_or_endpoint.chat.completions.create(
            model=model_or_key,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    
    elif model_type == "llama":
        # Azure Llama
        headers = {
            "Authorization": f"Bearer {model_or_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2000
        }
        response = requests.post(client_or_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    elif model_type == "ministral":
        # Azure Ministral
        headers = {
            "Authorization": f"Bearer {model_or_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 2000
        }
        response = requests.post(client_or_endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    elif model_type == "gemini":
        # Google Gemini
        import google.generativeai as genai
        genai.configure(api_key=client_or_endpoint)
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------- SQL CLEAN ----------
def clean_sql(raw_sql: str) -> str:
    sql = raw_sql.strip()
    if sql.startswith("```"):
        parts = sql.split("```")
        if len(parts) >= 2:
            sql = parts[1]
        sql = sql.replace("sql", "", 1).strip()
    return sql.replace("```", "").strip()


# ---------- SQL VALIDATE ----------
def validate_sql(sql: str) -> str:
    lowered = sql.lower()
    if not lowered.strip().startswith("select"):
        raise ValueError("Only SELECT queries allowed")
    blocked = ["drop", "delete", "update", "insert", "alter", "truncate"]
    if any(word in lowered for word in blocked):
        raise ValueError("Unsafe SQL detected")
    return sql


# ---------- EXECUTE SQL ----------
def run_sql(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df


# ---------- CHART → BASE64 ----------
def generate_chart_base64(df: pd.DataFrame, chart_type: str = "vertical_bar") -> str | None:
    if df.shape[1] < 2:
        return None

    labels = df.iloc[:, 0].astype(str)
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0)

    plt.figure(figsize=(12, 6))

    if chart_type == "vertical_bar":
        # Short labels, less than 8 bars - Region, Category, Brand comparisons
        bars = plt.bar(labels, values, color="steelblue", edgecolor="white", linewidth=1.5)
        plt.xticks(rotation=30, ha="right", fontsize=10)
        plt.ylabel(df.columns[1], fontsize=11, fontweight="bold")
        plt.title(f"{df.columns[1]} by {df.columns[0]}", fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height,
                    f"€{height:,.0f}" if height >= 1000 else f"€{height:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    elif chart_type == "horizontal_bar":
        # Long labels, more than 8 bars - Customer, Country rankings
        colors = plt.cm.Blues(values / values.max())
        bars = plt.barh(labels, values, color=colors, edgecolor="white", linewidth=1.5)
        plt.xlabel(df.columns[1], fontsize=11, fontweight="bold")
        plt.title(f"{df.columns[1]} by {df.columns[0]}", fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis="x", linestyle="--", alpha=0.4)
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2,
                    f"  €{width:,.0f}" if width >= 1000 else f"  €{width:.2f}",
                    ha="left", va="center", fontsize=9, fontweight="bold")

    elif chart_type == "area_line":
        # Time-based trends - Monthly, Quarterly, Yearly
        plt.plot(range(len(labels)), values, color="steelblue", linewidth=2.5, marker="o", markersize=6)
        plt.fill_between(range(len(labels)), values, alpha=0.3, color="steelblue")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=10)
        plt.ylabel(df.columns[1], fontsize=11, fontweight="bold")
        plt.title(f"{df.columns[1]} over {df.columns[0]}", fontsize=13, fontweight="bold", pad=15)
        plt.grid(axis="y", linestyle="--", alpha=0.4)
        # Add value labels on points
        for i, (label, value) in enumerate(zip(labels, values)):
            plt.text(i, value, f"€{value:,.0f}" if value >= 1000 else f"€{value:.2f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    elif chart_type == "donut":
        # Share & distribution - % of total
        colors = plt.cm.Set3(range(len(labels)))
        wedges, texts, autotexts = plt.pie(values, labels=labels, autopct="%1.1f%%",
                                            startangle=140, pctdistance=0.85, colors=colors,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
        # Create donut effect
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        plt.gca().add_artist(centre_circle)
        # Add total in center
        total = values.sum()
        plt.text(0, 0, f"Total\n€{total:,.0f}", ha="center", va="center",
                fontsize=14, fontweight="bold", color="#333")
        plt.title(f"{df.columns[1]} Distribution", fontsize=13, fontweight="bold", pad=15)
        plt.axis("equal")

    elif chart_type == "pie":
        # Simple pie chart (fallback)
        plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140,
                textprops={'fontsize': 10, 'fontweight': 'bold'})
        plt.axis("equal")
        plt.title(f"{df.columns[1]} Distribution", fontsize=13, fontweight="bold", pad=15)

    else:  # Default to vertical bar
        bars = plt.bar(labels, values, color="steelblue")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(df.columns[1])
        plt.title(f"{df.columns[1]} by {df.columns[0]}")
        plt.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


# ---------- PREPARE CHART DATA FOR PLOTLY ----------
def prepare_chart_data(df: pd.DataFrame, chart_type: str = "vertical_bar") -> dict:
    """Prepare data for Plotly.js dynamic charts"""
    if df.shape[1] < 2:
        return None

    labels = df.iloc[:, 0].astype(str).tolist()
    values = pd.to_numeric(df.iloc[:, 1], errors="coerce").fillna(0).tolist()
    
    x_label = df.columns[0]
    y_label = df.columns[1]
    
    return {
        "labels": labels,
        "values": values,
        "x_label": x_label,
        "y_label": y_label,
        "chart_type": chart_type
    }


# ---------- NATURAL LANGUAGE ANSWER ----------
def generate_natural_answer(question: str, df: pd.DataFrame, model_info: tuple) -> str:
    """Convert raw dataframe into natural language answer using selected LLM"""
    data_str = df.to_string(index=False)
    
    summary_prompt = f"""You are a Sales Analytics Assistant providing clear, structured insights.

The user asked: "{question}"

The data result is:
{data_str}

Provide a well-structured, friendly answer following these guidelines:

1. **Start with a clear summary** - State the key finding in 1 sentence
2. **Provide specific details** - Include actual numbers and comparisons
3. **Add context** - Explain what this means for the business
4. **Use formatting** - Break into short paragraphs for readability
5. **Be conversational** - Write like a helpful analyst, not a robot

Format your response like this example:

"Based on the data, [KEY FINDING].

Here are the highlights:
• [Top performer with specific number]
• [Second item with comparison]
• [Third item if relevant]

This shows that [BUSINESS INSIGHT]. [Optional recommendation or observation]."

Do NOT show raw tables or column names. Use natural language with proper formatting.
If the data shows "no results" or is empty, say: "I don't have information about that in the current database. Try asking about revenue, customers, regions, or product categories."
"""
    
    try:
        return call_llm(model_info, summary_prompt, temperature=0.3)
    except Exception as e:
        print(f"[ERROR] Natural answer generation failed: {e}")
        return data_str  # Fallback to raw data


# ---------- EXTRACT CONTEXT FROM SQL ----------
def extract_context_from_sql(sql: str) -> dict:
    """Extract filters/context from SQL query to maintain session state"""
    context = {}
    sql_lower = sql.lower()
    
    # Extract country filter
    if "country_name" in sql_lower:
        import re
        match = re.search(r"country_name\s*=\s*'([^']+)'", sql_lower)
        if match:
            context["country"] = match.group(1)
    
    # Extract customer/company filter
    if "customer_name" in sql_lower:
        import re
        if " in (" in sql_lower:
            match = re.search(r"customer_name\s+in\s*\(([^)]+)\)", sql_lower)
            if match:
                customers = [c.strip().strip("'\"") for c in match.group(1).split(",")]
                context["customers"] = customers
        else:
            match = re.search(r"customer_name\s*=\s*'([^']+)'", sql_lower)
            if match:
                context["customers"] = [match.group(1)]
    
    # Extract region filter
    if "region" in sql_lower and "where" in sql_lower:
        import re
        match = re.search(r"region\s*=\s*'([^']+)'", sql_lower)
        if match:
            context["region"] = match.group(1)
    
    # Extract brand filter
    if "brand" in sql_lower and "where" in sql_lower:
        import re
        match = re.search(r"brand\s*=\s*'([^']+)'", sql_lower)
        if match:
            context["brand"] = match.group(1)
    
    return context


# ---------- PROMPT ----------
def build_prompt(question: str, conversation_history: list = None, active_context: dict = None) -> str:
    schema = get_schema()
    
    # Build context from recent conversation (last 2-3 exchanges)
    context_section = ""
    if conversation_history and len(conversation_history) > 1:
        recent_history = conversation_history[-6:]  # Last 3 Q&A pairs (6 messages)
        context_section = "\n\nRECENT CONVERSATION CONTEXT:\n"
        for msg in recent_history[:-1]:  # Exclude current question
            role = "User" if msg["role"] == "user" else "Assistant"
            context_section += f"{role}: {msg['content']}\n"
        context_section += "\nUse this context to understand references like 'these companies', 'compare them', 'those', etc.\n"
    
    # Add active persistent context
    if active_context:
        context_section += "\n\nACTIVE CONTEXT (maintain these filters unless user explicitly changes topic):\n"
        if "country" in active_context:
            context_section += f"- Country filter: {active_context['country']}\n"
        if "customers" in active_context:
            context_section += f"- Specific customers: {', '.join(active_context['customers'])}\n"
        if "region" in active_context:
            context_section += f"- Region filter: {active_context['region']}\n"
        if "brand" in active_context:
            context_section += f"- Brand filter: {active_context['brand']}\n"
        context_section += "IMPORTANT: Keep applying these filters to ALL follow-up questions unless the user explicitly asks about a different country/region/brand/customer.\n"

    return f"""
You are a Sales Analytics Assistant. Return ONLY valid JSON. No markdown. No extra text.

RESPONSE TYPES:

1) Normal conversation (greetings, help, who-are-you):
{{
  "type": "chat",
  "message": "your response"
}}

2) Data analytics (sales, revenue, profit, customers, regions, brands, trends):
{{
  "type": "sql",
  "query": "SELECT query",
  "chart": "bar|line|pie"
}}

3) Out of scope (unrelated to sales):
{{
  "type": "chat",
  "message": "I can only answer questions about sales data. Try asking about revenue, customers, or regions."
}}

CHART RULES (choose the best visualization for the data):
- "vertical_bar"    → Region, Category, Brand, Quarter comparisons (short labels, less than 8 items, MINIMUM 2 items)
- "horizontal_bar"  → Customer, Country, Product rankings (long labels or more than 8 items, MINIMUM 2 items)
- "area_line"       → Time-based trends (monthly, quarterly, yearly revenue/profit trends, MINIMUM 3 data points)
- "donut"           → Share/distribution/percentage breakdown (MINIMUM 2 categories)
- Omit "chart" for single-value results, single items, or simple counts

IMPORTANT: Only include "chart" key when there are MULTIPLE items to compare. 
- If result has only 1 row → NO CHART (just text answer)
- If result has 2+ rows → Include appropriate chart type

CHART SELECTION GUIDE:
• Use "vertical_bar" for: revenue by region (multiple regions), sales by category (multiple categories)
• Use "horizontal_bar" for: top 10 customers, companies in a country (multiple companies)
• Use "area_line" for: monthly revenue trend (multiple months), quarterly performance
• Use "donut" for: revenue share by category (multiple categories), sales distribution
• NO CHART for: single customer, single category, single value, counts

TABLE: {TABLE_NAME}
COLUMNS: {schema}

KEY COLUMNS:
- total_eur       → Revenue in EUR (always use for revenue/sales)
- profit          → Profit in EUR
- profit_percent  → Profit margin % (pre-calculated, never recompute)
- discount_amount → Discount given
- quantity        → Units sold
- unit_price      → Price per unit
- invoice_date    → Date (YYYY-MM-DD)
- customer_name   → Customer/company name
- customer_code   → Customer ID
- country_name    → Country of sale
- country_code    → Country code
- region          → Sales region (APAC, Central EU, Eastern EU, LATAM, Southern EU, Western EU, International)
- brand           → Product brand
- category        → Product category (DCB, DES, PTCA, STX, Stents - Flex)
- description     → Product description
- item_number     → Product SKU
- is_open_order   → 1 = open/pending, 0 = completed
- fiscal_year     → Fiscal year
- fiscal_quarter  → Fiscal quarter
- fiscal_month    → Fiscal month
- sales_director  → Assigned sales director

RULES:
- Revenue/sales = SUM(total_eur)
- Total sales = SUM(total_eur)
- Top N = ORDER BY metric DESC LIMIT N
- Always GROUP BY for comparisons
- Always ROUND(SUM(total_eur), 2) for money values
- Always alias aggregates: SUM(total_eur) AS total_revenue
- Always filter blanks when grouping: WHERE column != '' AND column IS NOT NULL
- ALWAYS use LOWER() for string filters: WHERE LOWER(country_name) = 'france'
- Time grouping: strftime('%Y-%m', invoice_date) for month, strftime('%Y', invoice_date) for year
- Open orders: is_open_order = 1 | Completed orders: is_open_order = 0
- Default LIMIT 100 unless user specifies N
- SELECT only — no INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE

EXAMPLES:

User: hi
{{"type": "chat", "message": "Hi! I'm your Sales Analytics Assistant. Ask me about revenue, customers, regions, or product performance."}}

User: who are you
{{"type": "chat", "message": "I'm your Sales Analytics Assistant. I turn your sales data into insights."}}

User: total sales in eur
{{"type": "sql", "query": "SELECT ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME};"}}

User: companies in france
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(country_name) = 'france' AND customer_name != '' GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 100;", "chart": "horizontal_bar"}}

User: compare sales of companies in france
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(country_name) = 'france' AND customer_name != '' GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 100;", "chart": "horizontal_bar"}}

User: compare sales of CLINIQUE SAINT HILAIRE (BLOC) and Clinique Pasteur
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(customer_name) IN ('clinique saint hilaire (bloc)', 'clinique pasteur') GROUP BY customer_name ORDER BY total_revenue DESC;", "chart": "horizontal_bar"}}

User: revenue by region
{{"type": "sql", "query": "SELECT region, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE region != '' AND region IS NOT NULL GROUP BY region ORDER BY total_revenue DESC;", "chart": "vertical_bar"}}

User: top 5 customers by revenue
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE customer_name != '' GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 5;", "chart": "horizontal_bar"}}

User: monthly revenue trend
{{"type": "sql", "query": "SELECT strftime('%Y-%m', invoice_date) AS month, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE invoice_date != '' GROUP BY month ORDER BY month ASC;", "chart": "area_line"}}

User: revenue share by category
{{"type": "sql", "query": "SELECT category, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE category != '' GROUP BY category ORDER BY total_revenue DESC;", "chart": "donut"}}

User: sales distribution by region
{{"type": "sql", "query": "SELECT region, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE region != '' GROUP BY region ORDER BY total_revenue DESC;", "chart": "donut"}}

User: how many open orders
{{"type": "sql", "query": "SELECT COUNT(*) AS open_orders FROM {TABLE_NAME} WHERE is_open_order = 1;"}}

User: which product category has the highest profit margin
{{"type": "sql", "query": "SELECT category, ROUND(AVG(profit_percent), 2) AS avg_profit_margin FROM {TABLE_NAME} WHERE category != '' GROUP BY category ORDER BY avg_profit_margin DESC LIMIT 1;"}}

User: what is the total revenue
{{"type": "sql", "query": "SELECT ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME};"}}

User: what is the weather
{{"type": "chat", "message": "I can only answer questions about sales data. Try asking about revenue, customers, or regions."}}

User: tell me about employees
{{"type": "chat", "message": "I don't have information about that in the current database. I can help you with sales data, revenue, customers, regions, products, and financial metrics."}}

User: show me marketing data
{{"type": "chat", "message": "I don't have marketing data available. I specialize in sales analytics - ask me about revenue, profit, customers, regions, or product performance."}}

CONTEXT-AWARE EXAMPLES (using conversation history):

Previous: "what are the companies in france?" (Active Context: country='france')
Current: "can you compare sales of these companies?"
→ MAINTAIN France filter - interpret "these companies" as companies in France
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(country_name) = 'france' AND customer_name != '' GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 100;", "chart": "horizontal_bar"}}

Previous: "companies in france" (Active Context: country='france')
Current: "show me their profit"
→ MAINTAIN France filter - show profit for companies in France
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(profit), 2) AS total_profit FROM {TABLE_NAME} WHERE LOWER(country_name) = 'france' AND customer_name != '' GROUP BY customer_name ORDER BY total_profit DESC LIMIT 100;", "chart": "horizontal_bar"}}

Previous: "companies in france" (Active Context: country='france')
Current: "CLINIQUE SAINT HILAIRE (BLOC) and Clinique Pasteur compare these two"
→ MAINTAIN France filter + add specific customers
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(country_name) = 'france' AND LOWER(customer_name) IN ('clinique saint hilaire (bloc)', 'clinique pasteur') GROUP BY customer_name ORDER BY total_revenue DESC;", "chart": "horizontal_bar"}}

Previous: "top 5 brands by revenue" (Active Context: top 5 brands)
Current: "show me their profit margins"
→ MAINTAIN top 5 brands context
{{"type": "sql", "query": "SELECT brand, ROUND(AVG(profit_percent), 2) AS avg_profit_margin FROM {TABLE_NAME} WHERE brand != '' GROUP BY brand ORDER BY SUM(total_eur) DESC LIMIT 5;", "chart": "vertical_bar"}}

Previous: "companies in france" (Active Context: country='france')
Current: "what about germany?"
→ User explicitly changed context - switch to Germany
{{"type": "sql", "query": "SELECT customer_name, ROUND(SUM(total_eur), 2) AS total_revenue FROM {TABLE_NAME} WHERE LOWER(country_name) = 'germany' AND customer_name != '' GROUP BY customer_name ORDER BY total_revenue DESC LIMIT 100;", "chart": "horizontal_bar"}}
{context_section}
User message:
{question}
"""


# ---------- MAIN API ENDPOINT ----------
@app.post("/v2/chat-finace", response_model=ChatResponse, status_code=200)
def chat(request: ChatRequest):
    session_id = request.session_id
    question = request.question
    model_name = request.model

    try:
        # Get or initialize session history and context
        history = SESSION_MEMORY.get(session_id, [])
        active_context = SESSION_CONTEXT.get(session_id, {})
        history.append({"role": "user", "content": question})

        # Get the appropriate LLM client info
        model_info = get_llm_client(model_name)

        # Build prompt with conversation context and persistent filters
        prompt = build_prompt(question, history, active_context)

        # Call the LLM
        content = call_llm(model_info, prompt, temperature=0)

        # Robust JSON parsing (handles accidental markdown wrapping)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            clean = content.strip("```json").strip("```").strip()
            data = json.loads(clean)

        # ---------- CHAT RESPONSE ----------
        if data["type"] == "chat":
            answer = data["message"]
            history.append({"role": "assistant", "content": answer})
            SESSION_MEMORY[session_id] = history
            save_transcript(session_id, question, answer)
            return ChatResponse(answer=answer, image=None, chart_type=None, chart_data=None)

        # ---------- SQL RESPONSE ----------
        if data["type"] == "sql":
            sql = validate_sql(clean_sql(data["query"]))
            print(f"[SQL] Session: {session_id} | Model: {model_name} | Query: {sql}")

            # Extract and update persistent context from SQL
            new_context = extract_context_from_sql(sql)
            if new_context:
                # Merge with existing context (new filters override old ones)
                active_context.update(new_context)
                SESSION_CONTEXT[session_id] = active_context
                print(f"[CONTEXT] Session: {session_id} | Active Context: {active_context}")

            try:
                df = run_sql(sql)

            except Exception as db_error:
                print(f"[SQL ERROR] {db_error}")
                answer = "I couldn't run that query. Please try rephrasing your question."
                history.append({"role": "assistant", "content": answer})
                save_transcript(session_id, question, answer)
                SESSION_MEMORY[session_id] = history
                return ChatResponse(answer=answer, image=None, chart_type=None, chart_data=None)

            # No results
            if df.empty:
                answer = "I don't have information about that in the current database. Try asking about revenue, customers, regions, or product categories."
                image = None
                chart_type_used = None
                chart_data = None

            # Single value result or single row (no comparison needed)
            elif len(df) == 1 and df.shape[1] == 1:
                # Single value — just show the number
                answer = generate_natural_answer(question, df, model_info)
                image = None
                chart_type_used = None
                chart_data = None
            
            # Single row with multiple columns (still just one item, no comparison)
            elif len(df) == 1:
                # Only one item - no need for chart
                answer = generate_natural_answer(question, df, model_info)
                image = None
                chart_type_used = None
                chart_data = None

            # Table result with multiple rows (comparison makes sense)
            else:
                # Multi-row — natural language summary + chart
                chart_type_used = data.get("chart", "vertical_bar")
                image = generate_chart_base64(df, chart_type_used) if chart_type_used else None
                chart_data = prepare_chart_data(df, chart_type_used) if chart_type_used else None
                answer = generate_natural_answer(question, df, model_info)

            # Store assistant response in history with SQL context
            history.append({"role": "assistant", "content": f"SQL: {sql}\nResult: {answer}"})
            save_transcript(session_id, question, answer, image)
            SESSION_MEMORY[session_id] = history
            return ChatResponse(answer=answer, image=image, chart_type=chart_type_used, chart_data=chart_data)

        raise HTTPException(status_code=400, detail="Invalid model response type")

    except json.JSONDecodeError as je:
        print(f"[JSON ERROR] {je}")
        raise HTTPException(status_code=500, detail="Failed to parse model response as JSON")

    except ValueError as ve:
        print(f"[VALIDATION ERROR] {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# ---------- SESSION LISTING ENDPOINTS ----------
@app.get("/v2/sessions")
def list_sessions():
    """Get all available chat sessions"""
    try:
        sessions = []
        if os.path.exists(TRANSCRIPT_DIR):
            for filename in os.listdir(TRANSCRIPT_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(TRANSCRIPT_DIR, filename)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    if data:
                        sessions.append({
                            "session_id": filename.replace('.json', ''),
                            "message_count": len(data),
                            "last_updated": data[-1]["timestamp"] if data else None
                        })
        
        # Sort by last updated (most recent first)
        sessions.sort(key=lambda x: x["last_updated"] or "", reverse=True)
        
        return {"sessions": sessions}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v2/session/{session_id}")
def get_session(session_id: str):
    """Get chat history for a specific session"""
    try:
        file_path = os.path.join(TRANSCRIPT_DIR, f"{session_id}.json")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Session not found")
        
        with open(file_path, 'r') as f:
            messages = json.load(f)
        
        return {"session_id": session_id, "messages": messages}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- CHART REGENERATION ENDPOINT ----------
@app.post("/v2/regenerate-chart")
def regenerate_chart(request: dict):
    """Regenerate chart with different visualization type"""
    try:
        session_id = request.get("session_id")
        chart_type = request.get("chart_type", "vertical_bar")
        
        # Get last SQL query from session
        history = SESSION_MEMORY.get(session_id, [])
        if not history:
            raise HTTPException(status_code=404, detail="No session history found")
        
        # Find last SQL query
        last_sql = None
        for msg in reversed(history):
            if msg["role"] == "assistant" and "SQL:" in msg["content"]:
                last_sql = msg["content"].split("SQL:")[1].split("\n")[0].strip()
                break
        
        if not last_sql:
            raise HTTPException(status_code=404, detail="No SQL query found in history")
        
        # Re-run query and generate new chart
        df = run_sql(last_sql)
        if df.empty or len(df) == 1:
            return {"image": None, "chart_data": None, "message": "Cannot generate chart for this data"}
        
        # Generate both image and chart_data
        image = generate_chart_base64(df, chart_type)
        chart_data = prepare_chart_data(df, chart_type)
        
        print(f"[REGENERATE] Chart type: {chart_type}, Has image: {image is not None}, Has chart_data: {chart_data is not None}")
        
        return {"image": image, "chart_data": chart_data, "chart_type": chart_type}
    
    except Exception as e:
        print(f"[REGENERATE ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- GET AVAILABLE MODELS ----------
@app.get("/v2/models")
def get_models():
    """Get list of available LLM models"""
    return {
        "models": [
            {"id": "gpt-4o", "name": "GPT-4o (Default)", "provider": "Azure OpenAI"},
            {"id": "gpt-3.5", "name": "GPT-3.5 Turbo", "provider": "Azure OpenAI"},
            {"id": "llama", "name": "Llama 3.3 70B", "provider": "Azure"},
            {"id": "ministral", "name": "Ministral 3B", "provider": "Azure"},
            {"id": "gemini", "name": "Gemini Pro", "provider": "Google"}
        ]
    }
