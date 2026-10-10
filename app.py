
import re
import pandas as pd
import streamlit as st
from groq import Groq

st.set_page_config(
    page_title="SpendWise AI",
    page_icon="💸",
    layout="wide"
)

st.title("💸 SpendWise AI")
st.subheader("RAG-Powered Expense Auditor & Budget Advisor")
st.caption(
    "Educational prototype using synthetic transactions "
    "and illustrative policies. Human review is required."
)

# --------------------------------------------------
# 1. SAMPLE TRANSACTION DATA
# --------------------------------------------------

rows = [
    ["2026-09-02", "Office supplies", "Office", 1850, "Stationery Hub"],
    ["2026-09-03", "Team lunch", "Meals", 4200, "Urban Cafe"],
    ["2026-09-04", "Cloud hosting", "Software", 6200, "CloudWorks"],
    ["2026-09-05", "Taxi to client", "Travel", 1450, "RideNow"],
    ["2026-09-06", "Premium design tool", "Software", 3100, "DesignPro"],
    ["2026-09-08", "Client dinner", "Meals", 6800, "The Terrace"],
    ["2026-09-10", "Printer cartridges", "Office", 2750, "PrintPoint"],
    ["2026-09-12", "Rail ticket", "Travel", 2300, "RailConnect"],
    ["2026-09-14", "Project management subscription", "Software", 5400, "TaskPilot"],
    ["2026-09-16", "Team snacks", "Meals", 980, "QuickMart"],
    ["2026-09-18", "Hotel stay", "Travel", 7900, "CityStay"],
    ["2026-09-20", "Printer paper", "Office", 1250, "Stationery Hub"],
    ["2026-09-22", "AI writing subscription", "Software", 2200, "WriteSmart"],
    ["2026-09-24", "Business cab", "Travel", 1850, "RideNow"],
    ["2026-09-26", "Department dinner", "Meals", 7200, "Garden Table"],
]

df = pd.DataFrame(
    rows,
    columns=[
        "Date", "Description", "Category",
        "Amount (₹)", "Merchant"
    ]
)
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------------------------
# 2. POLICY KNOWLEDGE BASE FOR RAG
# --------------------------------------------------

POLICIES = [
    {
        "id": "POL-01",
        "title": "Monthly Budget Limits",
        "text": (
            "Illustrative monthly total budget is ₹40,000. "
            "Category guide limits: Office ₹8,000, Meals ₹10,000, "
            "Software ₹15,000, Travel ₹12,000. The overall total "
            "budget remains the cap; category guides are examples."
        )
    },
    {
        "id": "POL-02",
        "title": "Receipt and Documentation",
        "text": (
            "Illustrative policy: every expense above ₹2,000 "
            "should have a receipt or invoice attached. Flag it "
            "for review; do not automatically reject expenses."
        )
    },
    {
        "id": "POL-03",
        "title": "Meals and Entertainment",
        "text": (
            "Illustrative policy: meal expenses above ₹5,000 "
            "per transaction require manager approval and a "
            "stated business purpose. Approval cannot be verified "
            "from this sample dataset."
        )
    },
    {
        "id": "POL-04",
        "title": "Software Subscriptions",
        "text": (
            "Illustrative policy: recurring software subscriptions "
            "should be reviewed monthly for duplicate tools, "
            "active users and continued business need. Recurring "
            "does not automatically mean wasteful."
        )
    },
    {
        "id": "POL-05",
        "title": "Travel Expenses",
        "text": (
            "Illustrative policy: travel claims should include a "
            "business purpose and supporting receipt. Check hotel "
            "and transport costs against the actual travel policy "
            "before reimbursement."
        )
    },
    {
        "id": "POL-06",
        "title": "Responsible Review",
        "text": (
            "The dataset is synthetic and has no receipt images, "
            "approval records, employee details or verified "
            "business purpose. Flag potential checks only; never "
            "claim fraud or make final reimbursement decisions."
        )
    },
]

# --------------------------------------------------
# 3. RAG RETRIEVAL FUNCTION
# --------------------------------------------------

def retrieve_policies(question, top_k=3):
    """Retrieve policy passages using keyword overlap."""

    stop_words = {
        "the", "a", "an", "is", "are", "of", "for",
        "to", "and", "in", "on", "my", "me", "what",
        "why", "how", "does", "this", "that", "with",
        "show", "tell", "about"
    }

    query_tokens = set(
        re.findall(r"[a-zA-Z0-9]+", question.lower())
    ) - stop_words

    ranked = []

    for policy in POLICIES:
        policy_text = policy["title"] + " " + policy["text"]

        policy_tokens = set(
            re.findall(r"[a-zA-Z0-9]+", policy_text.lower())
        )

        overlap = query_tokens.intersection(policy_tokens)
        score = len(overlap) / max(1, len(query_tokens))

        if score > 0:
            ranked.append({
                **policy,
                "score": round(score, 3)
            })

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return ranked[:top_k]


# --------------------------------------------------
# 4. EXPENSE AUDIT RULES
# --------------------------------------------------

def add_flags(data):
    result = data.copy()

    def flag_transaction(row):
        flags = []

        if row["Amount (₹)"] > 2000:
            flags.append("Receipt check (> ₹2,000)")

        if (
            row["Category"] == "Meals"
            and row["Amount (₹)"] > 5000
        ):
            flags.append("Manager approval check (> ₹5,000)")

        if row["Category"] == "Software":
            flags.append("Subscription review")

        if row["Category"] == "Travel":
            flags.append("Travel purpose/receipt check")

        if not flags:
            return "No rule-based flag"

        return "; ".join(flags)

    result["Potential Flag"] = result.apply(
        flag_transaction,
        axis=1
    )

    return result


# --------------------------------------------------
# 5. APPLICATION TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    "📊 Expense Dashboard",
    "🤖 Ask SpendWise",
    "📚 RAG Policy Library"
])


# --------------------------------------------------
# TAB 1: EXPENSE DASHBOARD
# --------------------------------------------------

with tab1:
    st.subheader("Expense Overview")

    budget = st.number_input(
        "Monthly total budget (₹)",
        min_value=1000,
        value=40000,
        step=1000
    )

    categories = sorted(df["Category"].unique())

    selected_categories = st.multiselect(
        "Filter categories",
        categories,
        default=categories
    )

    data = df[
        df["Category"].isin(selected_categories)
    ].copy()

    total_spend = data["Amount (₹)"].sum()
    remaining = budget - total_spend

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Recorded Spend",
        f"₹{total_spend:,.0f}"
    )

    col2.metric(
        "Budget Remaining",
        f"₹{remaining:,.0f}"
    )

    col3.metric(
        "Transactions",
        len(data)
    )

    col4.metric(
        "Budget Used",
        f"{total_spend / budget * 100:.1f}%"
    )

    if remaining < 0:
        st.warning(
            "Recorded spending exceeds the selected budget."
        )
    else:
        st.success(
            "Recorded spending is within the selected total budget."
        )

    st.markdown("### Spending by Category")

    category_summary = (
        data.groupby("Category")["Amount (₹)"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_summary)

    st.markdown("### Transaction Audit")

    audited_data = add_flags(data)

    st.dataframe(
        audited_data,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Audit CSV",
        data=audited_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="spendwise_audit.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# TAB 2: ASK SPENDWISE
# --------------------------------------------------

with tab2:
    st.subheader("Ask the Expense Advisor")

    sample_questions = [
        "Which expenses need receipt checks?",
        "Why should meals above ₹5,000 be reviewed?",
        "How much did we spend on software subscriptions?",
        "Which categories are over their guide limits?",
        "What should I check before approving travel expenses?",
        "How does the monthly budget policy work?"
    ]

    selected_question = st.selectbox(
        "Choose a sample question",
        sample_questions
    )

    typed_question = st.text_input(
        "Or type your own question",
        placeholder=(
            "Ask about receipts, meals, software, "
            "travel or budgets"
        )
    )

    if st.button("Analyse with SpendWise", type="primary"):

        question = typed_question.strip() or selected_question

        # Step 1: Retrieve relevant policies
        retrieved = retrieve_policies(question, top_k=3)

        # Step 2: Identify relevant transaction records
        q = question.lower()

        if "software" in q or "subscription" in q:
            relevant_data = df[
                df["Category"] == "Software"
            ]

        elif (
            "meal" in q or "dinner" in q or "food" in q
        ):
            relevant_data = df[
                df["Category"] == "Meals"
            ]

        elif (
            "travel" in q or "hotel" in q
            or "taxi" in q or "cab" in q
        ):
            relevant_data = df[
                df["Category"] == "Travel"
            ]

        elif (
            "office" in q or "stationery" in q
            or "printer" in q
        ):
            relevant_data = df[
                df["Category"] == "Office"
            ]

        else:
            relevant_data = df

        st.markdown("### Agent Workflow")

        st.write(
            "1. Interpret question → "
            "2. Retrieve policy evidence → "
            "3. Analyse transactions → "
            "4. Generate a grounded explanation"
        )

        st.markdown("### Retrieved Policy Evidence")

        if retrieved:
            for policy in retrieved:
                with st.expander(
                    f'{policy["id"]} — {policy["title"]} '
                    f'| Score: {policy["score"]}',
                    expanded=True
                ):
                    st.write(policy["text"])
        else:
            st.warning(
                "No policy matched this wording. Try asking "
                "about budgets, receipts, meals, software or travel."
            )

        st.markdown("### Relevant Transaction Data")

        st.dataframe(
            relevant_data,
            use_container_width=True,
            hide_index=True
        )

        # Step 3: Optional LLM generation
        api_key = st.secrets.get("GROQ_API_KEY", "")

        if api_key and retrieved:
            try:
                client = Groq(api_key=api_key)

                policy_context = "\n\n".join(
                    f'{p["id"]} — {p["title"]}: {p["text"]}'
                    for p in retrieved
                )

                prompt = f"""
Question:
{question}

Retrieved policy evidence:
{policy_context}

Synthetic transaction data:
{relevant_data.to_string(index=False)}

Answer concisely using only the supplied evidence.
Cite relevant policy IDs.
Do not claim fraud or claim receipts or approvals are missing,
because the dataset does not contain those records.
Call findings potential checks.
State that policies are illustrative.
"""

                response = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are SpendWise AI, a cautious "
                                "expense-audit assistant. Use only "
                                "retrieved policies and supplied data. "
                                "Show calculations clearly. Do not "
                                "make final reimbursement decisions."
                            )
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.2,
                    max_tokens=800
                )

                st.markdown("### SpendWise Recommendation")

                st.write(
                    response.choices[0].message.content
                )

            except Exception as error:
                st.error(
                    "The Groq request failed. Check the API key, "
                    "model availability or free-tier rate limit."
                )

                st.caption(str(error)[:250])

        elif retrieved:
            st.info(
                "RAG retrieval and transaction analysis work "
                "without an API key. Add GROQ_API_KEY in "
                "Streamlit Secrets to enable AI-generated explanations."
            )

        else:
            st.info(
                "No AI response was generated because no "
                "matching policy evidence was retrieved."
            )


# --------------------------------------------------
# TAB 3: RAG POLICY LIBRARY
# --------------------------------------------------

with tab3:
    st.subheader("RAG Policy Library")

    st.write(
        "These illustrative policy passages form the knowledge "
        "base used by Ask SpendWise."
    )

    for policy in POLICIES:
        with st.expander(
            f'{policy["id"]} — {policy["title"]}'
        ):
            st.write(policy["text"])

    st.caption(
        "Retrieval uses keyword-overlap ranking. This is a "
        "lightweight RAG implementation, not embedding-based "
        "semantic search."
    )

st.divider()

st.caption(
    "SpendWise AI | Synthetic sample data | "
    "Educational prototype | Human review required"
)
