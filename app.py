import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Function to generate a response
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit app
st.title("Symptom Checker Chatbot")
st.write("Please describe your symptoms below. Remember, this is not a substitute for professional medical advice.")

# User input
user_input = st.text_area("Enter your symptoms:")

if st.button("Submit"):
    if user_input.strip() == "":
        st.warning("Please enter your symptoms.")
    else:
        with st.spinner("Generating response..."):
            response = generate_response(user_input)
            st.success("Chatbot Response:")
            st.write(response)
            st.write("If symptoms persist or worsen, please consult a doctor.")
