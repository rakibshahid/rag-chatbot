import streamlit as st
import pandas as pd
import faiss
import numpy as np
from openai import OpenAI
import os


# Load preprocessed FAISS index and DataFrame
faiss_index = faiss.read_index("document_index.faiss")
df = pd.read_csv("document_with_embeddings.csv")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to query the most relevant sections using FAISS
def query_document(user_query, k=100):
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=[user_query]
        )
        query_embedding = np.array(response.data[0].embedding).astype('float32').reshape(1, -1)
        distances, indices = faiss_index.search(query_embedding, k)
        relevant_rows = df.iloc[indices[0]]
        return relevant_rows[['Speaker', 'Date', 'Brief Summary']]
    except Exception as e:
        st.error(f"An error occurred while querying the document: {e}")
        return pd.DataFrame()

# Main Chatbot App
def app():
    st.set_page_config(page_title="ChatGPT RAG Chatbot", layout="wide")
    st.title("💬 ChatGPT RAG Chatbot")

    st.markdown(
        """
        <style>
        body {
            background-color: #f4f4f4;
        }
        .chat-container {
            max-width: 800px;
            margin: 20px auto;
        }
        .message-card {
            border-radius: 8px;
            padding: 10px 15px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        .user-message {
            background-color: #d1ffd6;
            text-align: right;
            border-left: 4px solid #4caf50;
        }
        .assistant-message {
            background-color: #f9f9f9;
            text-align: left;
            border-left: 4px solid #ccc;
        }
        .divider {
            height: 1px;
            background-color: #ddd;
            margin: 20px 0;
        }
        .input-container {
            max-width: 800px;
            margin: 20px auto;
        }
        .form-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I am a RAG-based chatbot. I have access to CIMM's (Committee on Citizenship and Immigration) commitee discussions for the year 2023 and 2024. Please feel free to talk to me about it. Since the data provided to me is stored in a vector embedded file, I will pull information based on the questions you ask and as I pull more information, I will have more context so initially my answers might seem a bit off but talk to me more and you will get insights."}]
        st.session_state.trigger_rerun = False  # Initialize rerun trigger

    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] != "system":  # Exclude system messages from default display
            role_class = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(
                f'<div class="message-card {role_class}"><strong>{message["role"].capitalize()}:</strong><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

    # Collapsible container for system messages
    with st.expander("Show System Messages (Hidden by Default)"):
        for message in st.session_state.messages:
            if message["role"] == "system":
                st.markdown(
                    f'<div class="message-card assistant-message"><strong>System:</strong><br>{message["content"]}</div>',
                    unsafe_allow_html=True,
                )

    # Input Form
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    with st.form("user_input_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here:", placeholder="Ask a question...")
        submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        # Add user's message to the conversation
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Query the document for relevant sections
        with st.spinner("Fetching relevant context..."):
            relevant_data = query_document(user_input)
            if not relevant_data.empty:
                context = "\n".join(
                    [f"Date: {row['Date']}, Summary: {row['Brief Summary']}" for _, row in relevant_data.iterrows()]
                )
                structured_context = f"The following information is related to your query:\n{context}"
            else:
                structured_context = "No relevant data was found for your query."
            st.session_state.messages.append({"role": "system", "content": structured_context})

        # OpenAI API call
        with st.spinner("Generating response..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages[-10:]  # Limit history sent to API
                )
                assistant_response = response.choices[0].message.content
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                # Trigger the hidden rerun button
                st.session_state.trigger_rerun = True
                st.rerun()
            
            except Exception as e:
                st.error(f"An error occurred: {e}")


        

# Run the app
if __name__ == "__main__":
    app()
