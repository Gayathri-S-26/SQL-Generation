def analytics_dashboard_page():
    import sqlite3
    import pandas as pd
    import streamlit as st
    import plotly.express as px
    from pathlib import Path

    # ----------------------------
    # DB Connection
    # ----------------------------
    BASE_DIR = Path(__file__).resolve().parent
    DB_PATH = BASE_DIR / "app_data.db"

    @st.cache_data
    def load_data():
        conn = sqlite3.connect(DB_PATH)
        queries = pd.read_sql("SELECT * FROM queries", conn)
        feedback = pd.read_sql("SELECT * FROM feedback", conn)
        projects = pd.read_sql("SELECT * FROM projects", conn)
        documents = pd.read_sql("SELECT * FROM documents", conn)
        users = pd.read_sql("SELECT * FROM users", conn)   # 👈 load users
        conn.close()
        return queries, feedback, projects, documents, users

    queries, feedback, projects, documents, users = load_data()

    # Merge usernames into queries
    if "user_id" in queries.columns:
        queries = queries.merge(
    users[["id", "username"]],
    left_on="user_id",
    right_on="id",
    how="left"
)  # now queries has "username"

    # ----------------------------
    # Page Layout
    # ----------------------------
    st.set_page_config(page_title="📊 Analytics Dashboard", layout="wide")
    st.title("📊 SQL Assistant Analytics Dashboard")

    # ----------------------------
    # Overview KPIs
    # ----------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Projects", len(projects))
    col2.metric("Total Documents", len(documents))
    col3.metric("Total Queries", len(queries))
    col4.metric("Unique Conversations", queries["conversation_id"].nunique())
    col5.metric("Feedback Collected", len(feedback))

    st.markdown("---")

    # ----------------------------
    # Collapse/Expand Toggle
    # ----------------------------
    expand_all = st.checkbox("🔽 Expand All Sections", value=True)

    # =====================================================
    # 🧑‍🤝‍🧑 USER ANALYTICS
    # =====================================================
    with st.expander("🧑‍🤝‍🧑 User Analytics", expanded=expand_all):

        st.subheader("Queries per User")
        if "username" in queries.columns:
            queries_per_user = queries.groupby("username")["query"].count().reset_index(name="query_count")
            fig_user_queries = px.bar(
                queries_per_user,
                x="username", y="query_count",
                title="Number of Queries per User",
                labels={"username": "User", "query_count": "Queries"},
                text_auto=True
            )
            st.plotly_chart(fig_user_queries, use_container_width=True)

        st.subheader("Sessions per User")
        sessions_per_user = queries.groupby("username")["conversation_id"].nunique().reset_index(name="session_count")
        fig_sessions = px.bar(
            sessions_per_user,
            x="username", y="session_count",
            title="Number of Sessions per User",
            labels={"username": "User", "session_count": "Sessions"},
            text_auto=True
        )
        st.plotly_chart(fig_sessions, use_container_width=True)

        st.subheader("Average Queries per Session")
        queries_per_session = queries.groupby("conversation_id")["query"].count()
        avg_queries_per_session = queries_per_session.mean()
        st.write(f"On average, users ask **{avg_queries_per_session:.2f} queries** per session.")

    # =====================================================
    # 📂 QUERY ANALYTICS
    # =====================================================
    with st.expander("📂 Query Analytics", expanded=expand_all):

        st.subheader("📂 Queries per Project")
        queries_per_project = (
            queries.groupby("project_id")["query"].count().reset_index(name="query_count")
        )
        queries_per_project = queries_per_project.merge(
            projects[["id", "name"]],
            left_on="project_id",
            right_on="id",
            how="left"
        ).rename(columns={"name": "project_name"})

        fig1 = px.bar(
            queries_per_project,
            x="project_name",
            y="query_count",
            title="Number of Queries per Project",
            labels={"project_name": "Project", "query_count": "Queries"},
            color="query_count",
            text_auto=True
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("💬 Conversation Depth")
        conv_depth = queries.groupby("conversation_id")["query"].count()
        avg_depth = conv_depth.mean()
        st.write(f"On average, users ask **{avg_depth:.2f} queries** per conversation.")

        fig2 = px.histogram(conv_depth, nbins=10, title="Distribution of Queries per Conversation")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("⏳ Queries Over Time")
        queries["created_at"] = pd.to_datetime(queries["created_at"])
        trend = queries.groupby(queries["created_at"].dt.date)["query"].count().reset_index()
        trend.columns = ["date", "total_queries"]

        fig4 = px.line(trend, x="date", y="total_queries", markers=True, title="Query Trends Over Time")
        st.plotly_chart(fig4, use_container_width=True)

    # =====================================================
    # ⚡ PERFORMANCE & QUALITY METRICS
    # =====================================================
    with st.expander("⚡ Performance & Quality Metrics", expanded=expand_all):

        if "response_time" in queries.columns:
            avg_response_time = queries["response_time"].dropna().mean()
            st.metric("Avg Response Time (sec)", f"{avg_response_time:.2f}")

            st.subheader("Latency Trend")
            latency_trend = queries.dropna(subset=["response_time"]).copy()
            latency_trend["created_at"] = pd.to_datetime(latency_trend["created_at"])
            latency_trend = latency_trend.groupby(latency_trend["created_at"].dt.date)["response_time"].mean().reset_index()

            fig_latency = px.line(
                latency_trend,
                x="created_at", y="response_time",
                markers=True,
                title="Average Response Time Over Time"
            )
            st.plotly_chart(fig_latency, use_container_width=True)

        if "attempts" in queries.columns:
            retry_rate = (queries["attempts"] > 1).mean() * 100
            st.metric("Retry Rate", f"{retry_rate:.2f}%")

        if "is_fallback" in queries.columns:
            fallback_rate = queries["is_fallback"].mean() * 100
            st.metric("Fallback Rate", f"{fallback_rate:.2f}%")

    # =====================================================
    # 👍👎 FEEDBACK ANALYTICS
    # =====================================================
    with st.expander("👍👎 Feedback Analytics", expanded=expand_all):

        if not feedback.empty:
            fb_counts = feedback["feedback"].value_counts().reset_index()
            fb_counts.columns = ["feedback_type", "count"]

            fig_fb = px.pie(
                fb_counts,
                names="feedback_type", values="count",
                title="Positive vs Negative Feedback"
            )
            st.plotly_chart(fig_fb, use_container_width=True)

    # =====================================================
    # 🔍 CONVERSATION EXPLORER
    # =====================================================
    with st.expander("🔍 Conversation Explorer", expanded=expand_all):

        if "username" in queries.columns:
            selected_user = st.selectbox("Select a User", queries["username"].unique())
            user_convs = queries[queries["username"] == selected_user]["conversation_id"].unique()

            if len(user_convs) > 0:
                selected_conv = st.selectbox("Select a Conversation", user_convs)

                conv_history = queries[
                    (queries["username"] == selected_user) &
                    (queries["conversation_id"] == selected_conv)
                ].sort_values("created_at")

                st.subheader(f"Conversation {selected_conv} (User: {selected_user})")

                for _, row in conv_history.iterrows():
                    with st.chat_message("user"):
                        st.markdown(row["query"])
                    with st.chat_message("assistant"):
                        st.markdown(row["answer"])
            else:
                st.info("No conversations found for this user.")
        else:
            st.warning("No username column in queries table.")
