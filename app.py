import streamlit as st
from recommend import get_top_k, generate_response

st.set_page_config(page_title="SHL Assessment Recommender", layout="centered")
st.title("🧠 SHL Assessment Recommendation Engine")

query = st.text_area("Enter job description or requirement:")

if st.button("Get Recommendations"):
    with st.spinner("Thinking..."):
        top_df = get_top_k(query)
        response = generate_response(query, top_df)

        st.subheader("🔎 Top Matches")
        for i, row in top_df.iterrows():
            st.markdown(f"""
            **[{row['Assessment Name']}]({row['URL']})**
            - Type: {row['Test Type']}
            - Duration: {row['Duration']}
            - Remote: {row['Remote Support']}
            - IRT: {row['IRT Support']}
            """)

        st.subheader("💡 AI Recommendation")
        st.write(response)
