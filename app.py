
import re
import math
import streamlit as st
import pandas as pd
from groq import Groq

# =====================================================
# STOCKSENSE AI - INVENTORY INTELLIGENCE AGENT
# =====================================================

st.set_page_config(
    page_title="StockSense AI",
    page_icon="📦",
    layout="wide"
)

# -------------------- DESIGN -------------------------

st.markdown("""
<style>
.stApp {
    background: #f4f7f5;
}
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #12372a, #247653);
    color: white;
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    font-size: 38px;
}
.hero p {
    color: #e0eee5;
    font-size: 16px;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e0e9e3;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -------------------- SAMPLE INVENTORY ---------------

INVENTORY = [
    {"Product": "Wireless Mouse", "Category": "Electronics",
     "Stock": 20, "Daily Sales": 8, "Lead Time": 5,
     "Unit Cost": 650, "Supplier": "TechSource"},

    {"Product": "USB-C Cable", "Category": "Accessories",
     "Stock": 85, "Daily Sales": 12, "Lead Time": 4,
     "Unit Cost": 180, "Supplier": "ConnectPlus"},

    {"Product": "Laptop Stand", "Category": "Accessories",
     "Stock": 12, "Daily Sales": 3, "Lead Time": 7,
     "Unit Cost": 950, "Supplier": "OfficePro"},

    {"Product": "Bluetooth Headphones", "Category": "Electronics",
     "Stock": 15, "Daily Sales": 6, "Lead Time": 6,
     "Unit Cost": 1250, "Supplier": "TechSource"},

    {"Product": "Notebook Pack", "Category": "Stationery",
     "Stock": 100, "Daily Sales": 10, "Lead Time": 3,
     "Unit Cost": 120, "Supplier": "PaperWorld"},

    {"Product": "Wireless Keyboard", "Category": "Electronics",
     "Stock": 10, "Daily Sales": 4, "Lead Time": 8,
     "Unit Cost": 1100, "Supplier": "TechSource"},

    {"Product": "Desk Lamp", "Category": "Office",
     "Stock": 25, "Daily Sales": 3, "Lead Time": 5,
     "Unit Cost": 780, "Supplier": "BrightWorks"},

    {"Product": "Water Bottle", "Category": "Lifestyle",
     "Stock": 18, "Daily Sales": 5, "Lead Time": 4,
     "Unit Cost": 350, "Supplier": "DailyNeeds"},

    {"Product": "Phone Charger", "Category": "Electronics",
     "Stock": 14, "Daily Sales": 7, "Lead Time": 5,
     "Unit Cost": 420, "Supplier": "ConnectPlus"},

    {"Product": "File Organizer", "Category": "Office",
     "Stock": 40, "Daily Sales": 2, "Lead Time": 6,
     "Unit Cost": 260, "Supplier": "OfficePro"},
]

# -------------------- BUILT-IN KNOWLEDGE BASE --------
# These are illustrative policies created for this demo,
# not official rules from a real company.

POLICIES = [
    {
        "id": "INV-01",
        "title": "Reorder Point Policy",
        "text": (
            "Reorder point is calculated as average daily sales "
            "multiplied by supplier lead time, plus safety stock. "
            "A product should be reviewed for replenishment when "
            "current stock is at or below its reorder point."
        )
    },
    {
        "id": "INV-02",
        "title": "Safety Stock Policy",
        "text": (
            "For this demonstration, safety stock is set to three "
            "days of average sales. Safety stock is a buffer against "
            "unexpected demand or supplier delays."
        )
    },
    {
        "id": "INV-03",
        "title": "Target Stock Policy",
        "text": (
            "For this demonstration, target stock covers supplier "
            "lead time plus seven additional days of average sales. "
            "Suggested order quantity is target stock minus current "
            "stock, but never less than zero."
        )
    },
    {
        "id": "INV-04",
        "title": "Urgency Policy",
        "text": (
            "Classify stock as Critical when current stock is below "
            "or equal to demand during supplier lead time. Classify "
            "as Reorder when stock is at or below reorder point. "
            "Otherwise classify as Monitor."
        )
    },
    {
        "id": "INV-05",
        "title": "Purchase Budget Policy",
        "text": (
            "Estimated purchase cost equals suggested order quantity "
            "multiplied by unit cost. This is an estimate and excludes "
            "taxes, shipping, discounts and other supplier charges."
        )
    },
    {
        "id": "INV-06",
        "title": "Data Quality Policy",
        "text": (
            "Recommendations depend on the accuracy of stock levels, "
            "average sales and supplier lead times. The application "
            "must disclose when it uses sample data and should not "
            "place an order automatically."
        )
    },
]

# -------------------- SIMPLE RAG ---------------------

def tokenize(text):
    """Convert text to searchable lowercase words."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve_policies(question, top_k=3):
    """
    Retrieve policy passages using lexical similarity.
    This is a lightweight, transparent RAG retriever.
    """
    query_terms = tokenize(question)

    scored = []

    for policy in POLICIES:
        passage = policy["title"] + " " + policy["text"]
        passage_terms = tokenize(passage)

        overlap = query_terms.intersection(passage_terms)

        # Overlap score normalized by query size.
        score = len(overlap) / max(len(query_terms), 1)

        # Give a small preference to meaningful matches.
        scored.append((score, policy))

    scored.sort(key=lambda item: item[0], reverse=True)

    # If there is no keyword match, return no evidence.
    matches = [
        {"score": score, **policy}
        for score, policy in scored[:top_k]
        if score > 0
    ]

    return matches


# -------------------- INVENTORY CALCULATIONS ----------

def analyse_inventory():
    results = []

    for item in INVENTORY:
        daily_sales = item["Daily Sales"]
        lead_time = item["Lead Time"]
        stock = item["Stock"]

        safety_stock = daily_sales * 3
        lead_time_demand = daily_sales * lead_time
        reorder_point = lead_time_demand + safety_stock
        target_stock = daily_sales * (lead_time + 7)

        order_qty = max(0, math.ceil(target_stock - stock))
        estimated_cost = order_qty * item["Unit Cost"]

        if stock <= lead_time_demand:
            urgency = "Critical"
        elif stock <= reorder_point:
            urgency = "Reorder"
        else:
            urgency = "Monitor"

        results.append({
            **item,
            "Lead-Time Demand": lead_time_demand,
            "Safety Stock": safety_stock,
            "Reorder Point": reorder_point,
            "Target Stock": target_stock,
            "Suggested Order": order_qty,
            "Estimated Cost (₹)": estimated_cost,
            "Urgency": urgency,
            "Days of Stock": round(stock / daily_sales, 1)
                if daily_sales > 0 else None
        })

    return pd.DataFrame(results)


# -------------------- AGENT WORKFLOW -----------------

def classify_request(question):
    q = question.lower()

    if any(word in q for word in [
        "budget", "cost", "spend", "price", "₹", "rupee"
    ]):
        return "Budget Analysis"

    if any(word in q for word in [
        "all", "every", "priority", "urgent", "critical",
        "reorder", "restock", "inventory", "stock"
    ]):
        return "Inventory Analysis"

    return "Policy and Inventory Question"


def build_evidence(question, inventory_df):
    """
    Agent workflow:
    1. Classify request.
    2. Retrieve policy evidence.
    3. Calculate inventory metrics.
    4. Prepare evidence for the response.
    """
    task = classify_request(question)
    retrieved = retrieve_policies(question)

    question_terms = tokenize(question)
    matched_products = inventory_df[
        inventory_df["Product"].apply(
            lambda name: bool(tokenize(name) & question_terms)
        )
    ]

    return task, retrieved, matched_products


# -------------------- HEADER --------------------------

st.markdown("""
<div class="hero">
    <h1>📦 StockSense AI</h1>
    <p>Inventory Intelligence & Replenishment Agent</p>
    <p>
        Discover stock risks, retrieve inventory policies,
        calculate reorder quantities and understand the evidence.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Academic prototype • Uses illustrative sample inventory "
    "and policy rules, not live business data."
)

# -------------------- DATA AND METRICS ----------------

df = analyse_inventory()

critical_count = int((df["Urgency"] == "Critical").sum())
reorder_count = int((df["Urgency"] == "Reorder").sum())
monitor_count = int((df["Urgency"] == "Monitor").sum())

m1, m2, m3, m4 = st.columns(4)

m1.metric("Products", len(df))
m2.metric("Critical", critical_count)
m3.metric("Reorder", reorder_count)
m4.metric("Monitor", monitor_count)

# -------------------- TABS ----------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Inventory Dashboard",
    "🤖 Ask StockSense",
    "📚 RAG Knowledge Base"
])

with tab1:
    st.subheader("Inventory Overview")

    show = df[[
        "Product", "Category", "Stock", "Daily Sales",
        "Lead Time", "Reorder Point", "Suggested Order",
        "Estimated Cost (₹)", "Urgency"
    ]].copy()

    st.dataframe(show, use_container_width=True, hide_index=True)

    st.subheader("Products Requiring Attention")

    urgent = df[df["Urgency"] != "Monitor"].sort_values(
        by=["Urgency", "Days of Stock"]
    )

    if urgent.empty:
        st.success("No products currently require a reorder review.")
    else:
        st.dataframe(
            urgent[[
                "Product", "Stock", "Days of Stock",
                "Reorder Point", "Suggested Order",
                "Urgency"
            ]],
            use_container_width=True,
            hide_index=True
        )

    total_cost = int(df["Estimated Cost (₹)"].sum())
    st.metric(
        "Estimated cost to bring all products to target stock",
        f"₹{total_cost:,.0f}"
    )

with tab2:
    st.subheader("Ask the Inventory Agent")

    st.write(
        "Ask a question about products, replenishment, urgency, "
        "inventory rules or estimated purchase cost."
    )

    example = st.selectbox(
        "Try an example",
        [
            "Which products should I reorder first?",
            "Why is Wireless Mouse marked for reorder?",
            "How is safety stock calculated?",
            "What is the estimated budget to replenish all products?",
            "Explain the reorder point policy."
        ]
    )

    question = st.text_input(
        "Your question",
        value=example,
        placeholder="e.g. Why should I reorder Phone Charger?"
    )

    run_agent = st.button(
        "Analyse with StockSense AI",
        type="primary",
        use_container_width=True
    )

    if run_agent and question.strip():
        task, evidence, product_matches = build_evidence(
            question, df
        )

        st.markdown("### Agent Workflow")
        st.write(f"**Detected task:** {task}")
        st.write("1. Request classified")
        st.write("2. Inventory data checked")
        st.write("3. Relevant policy passages retrieved")
        st.write("4. Evidence prepared for the response")

        st.markdown("### Retrieved Policy Evidence")

        if evidence:
            for source in evidence:
                with st.expander(
                    f"{source['id']} — {source['title']} "
                    f"(match score: {source['score']:.2f})",
                    expanded=True
                ):
                    st.write(source["text"])
        else:
            st.warning(
                "No relevant policy passage was retrieved. "
                "The agent should not invent a policy-based answer."
            )

        st.markdown("### Relevant Inventory Data")

        if not product_matches.empty:
            st.dataframe(
                product_matches[[
                    "Product", "Stock", "Daily Sales", "Lead Time",
                    "Reorder Point", "Suggested Order",
                    "Estimated Cost (₹)", "Urgency"
                ]],
                use_container_width=True,
                hide_index=True
            )
        elif any(word in question.lower() for word in [
            "all", "every", "priority", "urgent", "critical",
            "reorder", "restock", "inventory", "stock", "budget",
            "cost", "spend", "price"
        ]):
            st.dataframe(
                df[[
                    "Product", "Stock", "Daily Sales", "Lead Time",
                    "Reorder Point", "Suggested Order",
                    "Estimated Cost (₹)", "Urgency"
                ]],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(
                "No specific product was identified. Try using "
                "a product name or ask about the overall inventory."
            )

        # Prepare exact evidence for the language model.
        policy_context = "\n\n".join(
            f"{p['id']} - {p['title']}: {p['text']}"
            for p in evidence
        )

        if product_matches.empty:
            data_context = df[[
                "Product", "Stock", "Daily Sales", "Lead Time",
                "Reorder Point", "Suggested Order",
                "Estimated Cost (₹)", "Urgency"
            ]].to_string(index=False)
        else:
            data_context = product_matches[[
                "Product", "Stock", "Daily Sales", "Lead Time",
                "Reorder Point", "Suggested Order",
                "Estimated Cost (₹)", "Urgency"
            ]].to_string(index=False)

        st.markdown("### Agent Recommendation")

        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")

            if not api_key:
                st.warning(
                    "Groq API key is not configured. The retrieval "
                    "and calculations above work, but AI-generated "
                    "explanations require GROQ_API_KEY in Streamlit Secrets."
                )
            elif not evidence:
                st.warning(
                    "The agent abstains because it could not retrieve "
                    "supporting policy evidence. Try a question about "
                    "inventory, safety stock, reorder points or budget."
                )
            else:
                client = Groq(api_key=api_key)

                prompt = f"""
You are StockSense AI, an inventory-management decision-support agent.

Answer the manager's question using only the policy passages and
inventory data supplied below.

Rules:
- Treat the policy passages as the only authority for policy claims.
- Use the supplied calculated figures. Do not invent new figures.
- Explain the reason behind each recommendation.
- Clearly distinguish policy rules from calculated results.
- If evidence is incomplete, say what is missing.
- Do not claim an order has been placed.
- State that the data is illustrative sample data.
- Cite policy IDs such as [INV-01] next to policy-based claims.
- Do not claim a policy supports something it does not say.

Manager's question:
{question}

Retrieved policy evidence:
{policy_context}

Inventory data and calculated results:
{data_context}

Detected task:
{task}
"""

                with st.spinner("StockSense is preparing its recommendation..."):
                    response = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a careful inventory analyst. "
                                    "Be transparent, evidence-based and concise."
                                )
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.2,
                        max_tokens=900
                    )

                st.markdown(response.choices[0].message.content)

        except Exception as exc:
            st.error(
                "The AI API request failed. Check the API key, "
                "model availability and free-tier limits."
            )
            st.caption(f"Technical detail: {exc}")

        st.info(
            "Human review required: verify stock figures, supplier "
            "lead times and purchase costs before acting."
        )

with tab3:
    st.subheader("Built-in Inventory Knowledge Base")

    st.write(
        "These illustrative policy passages form the knowledge base "
        "used by the retrieval step. Retrieved passages are supplied "
        "to the language model as context."
    )

    for policy in POLICIES:
        with st.expander(
            f"{policy['id']} — {policy['title']}"
        ):
            st.write(policy["text"])

    st.caption(
        "RAG method: keyword-overlap retrieval over policy passages. "
        "This is a lightweight baseline; it does not use embeddings "
        "or a vector database."
    )

# -------------------- FOOTER --------------------------

st.divider()
st.caption(
    "StockSense AI | Agentic AI Assignment | "
    "Recommendations are advisory and use sample data."
)
