import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer

# Load the classifier and language model using the pipeline for zero-shot classification
@st.cache_resource
def load_models():
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    gpt2_model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
    model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
    return classifier, model, tokenizer

classifier, gpt2_model, gpt2_tokenizer = load_models()

# Initialize session state for storing the conversation history
if 'conversation' not in st.session_state:
    st.session_state.conversation = []

st.title("Career Transition Advisor Chat")
st.write("Chat with the advisor to get guidance on career transitions.")

# Define possible labels for classification
labels = ["Yes", "No"]

# Use a form for user input to handle the submission more smoothly
with st.form(key='user_input_form'):
    user_input = st.text_input("You:", "")
    submit_button = st.form_submit_button(label='Send')

# Process input when the form is submitted
if submit_button and user_input:
    # Add user's input to the conversation history
    st.session_state.conversation.append({"role": "user", "text": user_input})
    
    # Perform zero-shot classification
    result = classifier(user_input, labels)
    
    # Extract the answer with the highest confidence
    answer = result['labels'][0]
    confidence = result['scores'][0] * 100  # Convert to percentage

    # Generate an explanation using GPT-2 with a better prompt
    explanation_prompt = f"{user_input} Answer: {answer}. Please briefly explain why this is the case."
    inputs = gpt2_tokenizer.encode(explanation_prompt, return_tensors='pt')

    # Generate a response for the explanation
    explanation_outputs = gpt2_model.generate(
        inputs,
        max_length=50,  # You can adjust this as needed
        num_return_sequences=1,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7,
        eos_token_id=gpt2_tokenizer.eos_token_id,  # Stop at the EOS token
        pad_token_id=gpt2_tokenizer.eos_token_id  # Set padding to EOS
    )

    # Decode the output into text and take the first explanation
    explanation = gpt2_tokenizer.decode(explanation_outputs[0], skip_special_tokens=True)

    # Ensure the explanation ends with a full stop
    if not explanation.endswith('.'):
        # If it doesn't, you could add a full stop or truncate the output
        explanation = explanation.rsplit('.', 1)[0] + '.' if '.' in explanation else explanation

    # Formulate the assistant's response
    assistant_response = f"*Answer:* {answer} (Confidence: {confidence:.2f}%)\n\n*Explanation:* {explanation}"

    # Add the assistant's response to the conversation history
    st.session_state.conversation.append({"role": "assistant", "text": assistant_response})

# Display the conversation history
for message in st.session_state.conversation:
    if message["role"] == "user":
        st.write(f"You: {message['text']}")
    else:
        st.write(f"Advisor: {message['text']}")