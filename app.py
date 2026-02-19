"""
app.py — Streamlit Chat UI for E-COM AI ANALYST (Phase 5)

Features:
  • Chat-like interface with message history
  • Conversational memory — follow-up questions inherit context
  • Guardrail — off-topic questions are politely declined
  • Explainability — every answer displays Data, Explanation
    (with Confidence), and a toggleable SQL block
  • Self-healing status shown when retries occur
  • Streamlit Cloud compatible — reads GEMINI_API_KEY from st.secrets
"""

import os
import streamlit as st

# ── Streamlit Cloud Secrets Injection ────────────────────────────────────
# On Streamlit Community Cloud, secrets are stored in the dashboard (not
# .env). We inject them into os.environ BEFORE importing agent so that
# agent.init_gemini() picks up the key via os.environ.get("GEMINI_API_KEY").
# Locally, agent still reads from .env via python-dotenv — no change needed.
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

import pandas as pd
import sqlite3
import tempfile
import agent  # the core engine

# ── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Com AI Analyst",
    page_icon="📊",
    layout="centered",
)

# ── Custom CSS — hide Streamlit defaults for a cleaner look ──────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 2rem;}
</style>
""", unsafe_allow_html=True)

# ── Session State Initialisation ─────────────────────────────────────────
if "messages" not in st.session_state:
    # UI message list: [{role, content, ...}, ...]
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    # LLM conversational memory (Gemini format)
    st.session_state.chat_history = None

if "client" not in st.session_state:
    st.session_state.client = agent.init_gemini()

# Dynamic-upload state
if "dynamic_schema" not in st.session_state:
    st.session_state.dynamic_schema = None

if "business_context" not in st.session_state:
    st.session_state.business_context = None

if "db_path" not in st.session_state:
    # Default: legacy ecom.db
    st.session_state.db_path = agent.DB_PATH

if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "default_profiled" not in st.session_state:
    st.session_state.default_profiled = False

# ── Auto-profile the default database on first load ──────────────────────
# This ensures ecom.db (or any default DB) gets the same dynamic profiler
# treatment as uploaded CSVs — zero hardcoding, fully SaaS-ready.
if (not st.session_state.default_profiled
        and st.session_state.uploaded_file_name is None
        and os.path.exists(st.session_state.db_path)):
    try:
        schema_str = agent.get_schema(st.session_state.db_path)
        st.session_state.dynamic_schema = schema_str

        # Grab sample rows for the profiler
        conn = sqlite3.connect(st.session_state.db_path)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        if tables:
            first_table = tables[0][0]
            sample_df = pd.read_sql(
                f"SELECT * FROM [{first_table}] LIMIT 5", conn
            )
            sample_json = sample_df.to_json(orient="records",
                                             date_format="iso")
            biz_ctx = agent.profile_dataset(
                st.session_state.client, schema_str, sample_json
            )
            st.session_state.business_context = biz_ctx
        conn.close()
    except Exception:
        # Non-fatal: the agent still works, just without profiler context
        pass
    st.session_state.default_profiled = True


# ── Helper: load DataFrame into a temp-file SQLite DB ────────────────────
def _load_df_to_sqlite(df: pd.DataFrame,
                       table_name: str = "uploaded_data") -> str:
    """
    Write *df* into a temporary physical SQLite file and return its path.

    Why a temp file instead of :memory:?
      • Thread-safe — every agent.execute_sql() call opens/closes its own
        short-lived connection to the file, exactly like the legacy ecom.db.
      • No pickling issues — only a plain string path is stored in
        st.session_state.
      • Auto-cleanup — OS clears the temp dir on restart; we also delete
        the previous temp file when a new CSV is uploaded.
    """
    # Clean up any previous temp DB
    old_path = st.session_state.get("_tmp_db_path")
    if old_path and os.path.exists(old_path):
        try:
            os.remove(old_path)
        except OSError:
            pass

    # Create a fresh temp file  (delete=False so it persists across reruns)
    tmp = tempfile.NamedTemporaryFile(
        suffix=".db", prefix="ecom_upload_", delete=False
    )
    tmp_path = tmp.name
    tmp.close()  # close the handle; sqlite3 will reopen it

    conn = sqlite3.connect(tmp_path)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()

    # Remember the path so we can clean it up next time
    st.session_state["_tmp_db_path"] = tmp_path
    return tmp_path


# ── Smart Rendering Helper ───────────────────────────────────────────────
def _render_smart(columns: list, rows: list, formatted_fallback: str,
                  widget_key: str = ""):
    """
    Smart result display:
      • 1×1 scalar          → st.metric  (big bold number)
      • N×2 text + number   → bar chart  + table
      • N×2 date + number   → line chart + table
      • Everything else      → interactive table
    Always appends a CSV download button.
    """
    if not rows or not columns:
        st.code(formatted_fallback, language=None)
        return

    df = pd.DataFrame(rows, columns=columns)

    # ── Scalar: 1 row × 1 col ──
    if len(rows) == 1 and len(columns) == 1:
        value = rows[0][0]
        label = columns[0].replace("_", " ").title()
        if isinstance(value, float):
            st.metric(label=label, value=f"{value:,.2f}")
        elif isinstance(value, int):
            st.metric(label=label, value=f"{value:,}")
        else:
            st.metric(label=label, value=str(value))

    # ── Two columns + multiple rows: auto-chart ──
    elif len(columns) == 2 and len(rows) > 1:
        DATE_HINTS = {"month", "date", "year", "week", "day", "day_of_week",
                      "quarter", "period", "year_month"}

        col_a, col_b = columns
        numeric_col, category_col = None, None

        # Detect which column is numeric
        try:
            df[col_b] = pd.to_numeric(df[col_b])
            numeric_col, category_col = col_b, col_a
        except (ValueError, TypeError):
            pass

        if numeric_col is None:
            try:
                df[col_a] = pd.to_numeric(df[col_a])
                numeric_col, category_col = col_a, col_b
            except (ValueError, TypeError):
                pass

        if numeric_col and category_col:
            chart_df = df.set_index(category_col)[[numeric_col]]
            if category_col.lower() in DATE_HINTS:
                st.line_chart(chart_df)
            else:
                st.bar_chart(chart_df)

        # Always show the data table below any chart
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── 3+ columns or single-row multi-col: interactive table ──
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Download button ──
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv_data,
        file_name="query_result.csv",
        mime="text/csv",
        key=f"dl_{widget_key}",
    )


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 E-Com AI Analyst")
    st.caption("Text-to-SQL for any business dataset")
    st.divider()

    # ── CSV Upload ───────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📂 Upload a CSV file", type=["csv"],
        help="Upload any CSV. The AI will learn its structure automatically."
    )

    if uploaded_file is not None:
        # Only re-process when a *new* file is uploaded
        if uploaded_file.name != st.session_state.uploaded_file_name:
            with st.spinner("Analysing your dataset…"):
                df = pd.read_csv(uploaded_file)
                st.session_state.uploaded_df = df
                st.session_state.uploaded_file_name = uploaded_file.name

                # 1. Load into in-memory SQLite
                db_uri = _load_df_to_sqlite(df)
                st.session_state.db_path = db_uri

                # 2. Generate dynamic schema
                schema_str = agent.generate_dynamic_schema(df)
                st.session_state.dynamic_schema = schema_str

                # 3. Profile the dataset (single LLM call — fault-tolerant)
                sample_json = df.head(3).to_json(orient="records",
                                                  date_format="iso")
                try:
                    biz_ctx = agent.profile_dataset(
                        st.session_state.client, schema_str, sample_json
                    )
                    st.session_state.business_context = biz_ctx
                except Exception as prof_err:
                    st.session_state.business_context = None
                    st.warning(
                        "⚠️ Data Profiler skipped — API rate-limited or "
                        "unavailable. Your data is still loaded and fully "
                        "queryable.\n\n"
                        f"Details: `{prof_err}`"
                    )

                # Reset conversation for the new dataset
                st.session_state.messages = []
                st.session_state.chat_history = None

            st.success(f"✅ Loaded **{uploaded_file.name}** "
                       f"({len(df):,} rows × {len(df.columns)} cols)")

        # Show detected context
        if st.session_state.business_context:
            with st.expander("🧠 Detected Business Context"):
                st.markdown(st.session_state.business_context)

    st.divider()
    st.markdown(
        "**Ask any business question** about your data:\n"
        "- Revenue, trends, top products\n"
        "- Customer analysis\n"
        "- Date-range comparisons\n"
        "- …or anything your CSV contains"
    )
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = None
        st.rerun()
    st.divider()
    st.caption("Built with Gemini 2.0 Flash · Zero Hallucination Pipeline")


# ── Render Chat History ──────────────────────────────────────────────────
for _msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # If the assistant message has structured data, show it
        if msg["role"] == "assistant" and "result_data" in msg:
            data = msg["result_data"]

            # Smart data display (metric / chart / table + download)
            _render_smart(
                columns=data.get("columns", []),
                rows=data.get("rows", []),
                formatted_fallback=data.get("formatted", ""),
                widget_key=f"hist_{_msg_idx}",
            )

            # Confidence badge
            conf = data.get("confidence", "")
            conf_color = {
                "High": "🟢", "Medium": "🟡", "Low": "🔴"
            }.get(conf, "⚪")
            st.caption(f"{conf_color} Confidence: **{conf}**")

            # Self-healing notice
            if data.get("self_healed"):
                attempts = data.get("attempts", [])
                with st.expander(
                    f"🔧 Self-healed after {len(attempts)} attempts"
                ):
                    for a in attempts:
                        status = "✅ OK" if a["error"] is None else f"❌ {a['error']}"
                        st.text(f"Attempt {a['attempt']}: {status}")

            # Toggleable SQL
            with st.expander("🔍 Show SQL"):
                st.code(data["sql"], language="sql")


# ── Chat Input ───────────────────────────────────────────────────────────
# Guard: if no data is loaded (Cloud deployment with no ecom.db and no
# uploaded CSV), show a prompt to upload data instead of a broken chat.
_data_is_ready = (
    st.session_state.uploaded_file_name is not None
    or os.path.exists(st.session_state.db_path)
)

if not _data_is_ready:
    st.info(
        "📂 **Upload a CSV file** in the sidebar to get started. "
        "Once uploaded, you can ask any business question about your data."
    )

if _data_is_ready and (prompt := st.chat_input("Ask a business question…")):
    # ── Show user message ──
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── Generate response ──
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = agent.ask(
                    question=prompt,
                    client=st.session_state.client,
                    chat_history=st.session_state.chat_history,
                    db_path=st.session_state.db_path,
                    dynamic_schema=st.session_state.dynamic_schema,
                    business_context=st.session_state.business_context,
                )
            except RuntimeError as e:
                # Self-healing exhausted
                error_msg = (
                    "⚠️ I tried to fix the query automatically but couldn't "
                    f"resolve it after {agent.MAX_RETRIES} attempts.\n\n"
                    f"```\n{e}\n```\n\n"
                    "Please try rephrasing your question."
                )
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.stop()
            except Exception as e:
                error_msg = f"⚠️ Something went wrong: {e}"
                st.markdown(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
                st.stop()

        # ── Blocked by guardrail or input validation ──
        if result.get("blocked"):
            block_msg = f"⚠️ {result['block_reason']}"
            st.markdown(block_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": block_msg}
            )
            st.stop()

        # ── Clarification needed ──
        if result.get("response_type") == "clarify":
            clarify_msg = f"🤔 {result['clarify_question']}"
            st.markdown(clarify_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": clarify_msg}
            )
            if result.get("history"):
                st.session_state.chat_history = result["history"]
            st.stop()

        # ── Schema check: UNABLE_TO_ANSWER ──
        if result.get("response_type") == "unable":
            unable_msg = f"⚠️ {result['unable_reason']}"
            st.markdown(unable_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": unable_msg}
            )
            # Preserve conversational memory
            if result.get("history"):
                st.session_state.chat_history = result["history"]
            st.stop()

        # ── Success: display structured answer ──
        # Update conversational memory
        st.session_state.chat_history = result["history"]

        # Explanation text (main assistant message)
        explanation = result["explanation"]
        st.markdown(f"💡 {explanation}")

        # Empty result notice
        if result.get("empty_result"):
            st.info("📭 The query returned no data. Try adjusting your "
                    "filters (dates, countries, products).")

        # Smart data display (metric / chart / table + download)
        _render_smart(
            columns=result.get("columns", []),
            rows=result.get("rows", []),
            formatted_fallback=result.get("formatted", ""),
            widget_key="live",
        )

        # Confidence badge
        conf = result.get("confidence", "")
        conf_color = {
            "High": "🟢", "Medium": "🟡", "Low": "🔴"
        }.get(conf, "⚪")
        st.caption(f"{conf_color} Confidence: **{conf}**")

        # Self-healing notice
        if result.get("self_healed"):
            attempts = result.get("attempts", [])
            with st.expander(
                f"🔧 Self-healed after {len(attempts)} attempts"
            ):
                for a in attempts:
                    status = "✅ OK" if a["error"] is None else f"❌ {a['error']}"
                    st.text(f"Attempt {a['attempt']}: {status}")

        # Toggleable SQL
        with st.expander("🔍 Show SQL"):
            st.code(result["sql"], language="sql")

        # Store the message with attached structured data for re-render
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"💡 {explanation}",
            "result_data": {
                "formatted": result["formatted"],
                "columns": result.get("columns", []),
                "rows": [list(r) for r in result.get("rows", [])],
                "sql": result["sql"],
                "confidence": result.get("confidence", ""),
                "self_healed": result.get("self_healed", False),
                "attempts": result.get("attempts", []),
            },
        })
