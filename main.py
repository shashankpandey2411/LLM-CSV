import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
import replicate
import os

if 'REPLICATE_API_TOKEN' in st.secrets:
    replicate_api = st.secrets['REPLICATE_API_TOKEN']
else:
    replicate_api = st.text_input('Enter Replicate API token:', type='password')
    if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
        st.warning('Please enter your credentials!')
    else:
        st.success('Proceed to entering your prompt message!')

os.environ['REPLICATE_API_TOKEN'] = replicate_api

# Set up Streamlit page configuration
st.set_page_config(
    page_title="App for statistical analysis of CSV",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar options
st.sidebar.title('Options for analysis')
option = st.sidebar.radio("Select an option:", ["Statistical Analysis and Visualization", "General Query about Data"])

# Common CSS styles
styles = """
<style>
img {
    max-width: 50%;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
</style>
"""
st.markdown(styles, unsafe_allow_html=True)

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You should respond as 'Data analyst'."
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = replicate.run(
        'a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5',
        input={
            "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 512,
            "repetition_penalty": 1
        }
    )
    return output

# Function for generating LLaMA prompt
def generate_llama_prompt(df, user_prompt):
    data_sample = df.head().to_string()
    data_summary = df.describe().to_string()
    return f"""
    CSV Data Sample:
    {data_sample}

    Data Summary:
    {data_summary}

    User Request:
    {user_prompt}

    Please understand the data first then answer the query according to the data given.
    """

# Main content
st.title("App for statistical analysis of CSV")
st.markdown(
    "An application that can perform statistical analysis of CSV files using the Prompt and LLM model, and generate plots based on the results. "
    "The app will answer your questions and provide you with insights about your data."
)

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Here are the first 5 rows of the dataset:")
    st.write(df.head())
    st.write("Data Description:")
    st.write(df.describe())

    if option == "Statistical Analysis and Visualization":
        # Statistical Analysis and Visualization code
        llm = GooglePalm(api_key="AIzaSyCMX48owzYyq0oCrilvA08sP-Lren0b3NI")
        sdf = SmartDataframe(df, config={"llm": llm})
        query = st.text_input(label='Enter your query')
        Analyze = st.button(label='Analyze')
        if Analyze:
            result = sdf.chat(query)
            st.write(result)

    elif option == "General Query about Data":
        # Display or clear chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Enter your query about the data:", disabled=not replicate_api)
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # Generate a new response if the last message is not from assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        prompt = generate_llama_prompt(df, question)
                        response = generate_llama2_response(prompt)
                        placeholder = st.empty()
                        full_response = ''
                        for item in response:
                            full_response += item
                            placeholder.markdown(full_response)
                        placeholder.markdown(full_response)
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)

else:
    st.warning("Please select a CSV file to continue.")
    st.stop()
