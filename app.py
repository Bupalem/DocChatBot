import streamlit as st

st.title("📄 Document Chatbot")
st.write("Loading RAG pipeline...")

# Import after setting up Streamlit UI
from rag_pipeline import ask_question

st.success("Pipeline loaded successfully!")

user_input = st.text_input("Ask something about your documents:")

if user_input:
    with st.spinner("Thinking..."):
        try:
            response = ask_question(user_input)

            st.write("### Answer:")
            st.write(response["result"])

            with st.expander("Sources"):
                for doc in response["source_documents"]:
                    st.write(f"📄 {doc.metadata.get('source')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
else:
    st.info("Enter a question to get started!")