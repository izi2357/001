import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load the GPT-Neo model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit app layout
st.title("Text Generation with GPT-Neo")
st.write("Enter a prompt for text generation:")

prompt = st.text_input("Prompt")

if st.button("Generate Text"):
    if prompt:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write(generated_text)
    else:
        st.write("Please enter a prompt.")

# Add a chat feature
st.title("Chat with GPT-Neo")
st.write("Enter your message to chat with the model:")

chat_history = st.text_area("Chat History", height=300)
user_input = st.text_input("Your message")

if st.button("Send"):
    if user_input:
        chat_history += f"\nUser: {user_input}"
        inputs = tokenizer(chat_history, return_tensors="pt")
        outputs = model.generate(inputs.input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
        bot_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        chat_history += f"\nGPT-Neo: {bot_response}"
        st.write(chat_history)
    else:
        st.write("Please enter a message.")
