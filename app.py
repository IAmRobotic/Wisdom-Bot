import streamlit as st
import os
from dotenv import load_dotenv
from guru_chatbot import GuruChatbot
import toml

load_dotenv()

if os.path.exists(".streamlit/config.toml"):
    config = toml.load(".streamlit/config.toml")
    st.set_page_config(
        page_title="Wisdom Bot: AI RAG proof of concept",
        page_icon="💡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# st.title("Wisdom Bot: Timeless insights powered by AI")
st.markdown(
    '<h1 style="color:#06BFAD;">Wisdom Bot: Timeless insights powered by AI</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "<h5>All responses derived via AI RAG techniques from text of Meditations by Marcus Aurelius and the Tao Te Ching.</h5>",
    unsafe_allow_html=True,
)

st.markdown(
    '<h6 style="color:#9FD4D3;"> Instructions: try a question like "My son is struggling with grades at school. How can I help him?</h6>',
    unsafe_allow_html=True,
)


# Initialize the chatbot in the session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = GuruChatbot(
        openai_api_key=st.secrets.get("OPENAI_API_KEY"),
        google_api_key=st.secrets.get("GOOGLE_API_KEY"),
        qdrant_url=st.secrets.get("QDRANT_URL"),
        qdrant_api_key=st.secrets.get("QDRANT_API_KEY"),
    )

# Initialize messages in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Handle user input
if prompt := st.chat_input("Unburden your mind. What's troubling you?"):
    # Clear the example question after it's used
    st.session_state.example_question = ""
    # Add user message to the state
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot's response in form of a tuple with summary response, tao quote and meditations quote
    with st.spinner("Contemplating..."):
        responses = st.session_state.chatbot.get_advice(prompt)
        response = responses[0] if responses else None
        tao_quote = responses[1] if responses else "NONE"
        meditations_quote = responses[2] if responses else "NONE"

    # Display response and source quotes
    with st.chat_message("assistant"):
        if response:
            st.markdown(response)
            if tao_quote != "NONE" and tao_quote != None:
                st.markdown("#### Quote from Tao Te Ching:")
                st.markdown(tao_quote.strip())
            if meditations_quote != "NONE" and meditations_quote != None:
                st.markdown("#### Quote from Meditations:")
                st.markdown(meditations_quote.strip())

            # updates history of responses
            if (tao_quote != "NONE" and tao_quote != None) or (
                meditations_quote != "NONE" and meditations_quote != None
            ):
                session_content = f"{response}\n\n#### Quote from Tao Te Ching:\n{tao_quote}\n\n#### Quote from Meditations:\n{meditations_quote}"
            else:
                session_content = response

            st.session_state.messages.append(
                {"role": "assistant", "content": session_content}
            )
        else:
            st.markdown("I'm sorry, I couldn't find relevant advice.")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "I'm sorry, I couldn't find relevant advice.",
                }
            )
