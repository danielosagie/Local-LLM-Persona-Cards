import streamlit as st
import json
from PersonaCard import LLMProcessor
import time

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'persona_data' not in st.session_state:
    st.session_state.persona_data = {}
if 'system_message' not in st.session_state:
    st.session_state.system_message = '''
    You are an AI assistant helping to create a persona card. Respond to the user's input by providing relevant information in JSON format. Include categories such as Education, Experience, Skills, Strengths, Goals, and Values. Keep your responses concise and relevant.

    Example response format:
    {
        "Education": ["High School"],
        "Experience": ["1 year retail"],
        "Skills": ["Adaptable", "Quick learner", "Tech-savvy"],
        "Strengths": ["Flexible schedule", "Eager to work"],
        "Goals": ["Gain experience", "Learn new skills"],
        "Values": ["Community involvement"]
    }
    '''

# Streamlit UI
st.title("LLM Persona Cards (Local Version)")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    if st.button("Initialize Local Model"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)

        st.session_state.processor = LLMProcessor(progress_callback)
        st.success("Local model initialized successfully!")

    # System Message Editor
    with st.expander("Edit System Message", expanded=False):
        edited_system_message = st.text_area("System Message:", 
                                             value=st.session_state.system_message,
                                             height=300)
        if st.button("Update System Message"):
            st.session_state.system_message = edited_system_message
            if st.session_state.processor:
                st.session_state.processor.update_system_message(edited_system_message)
            st.success("System message updated!")

# Chat interface
st.header("Chat Interface")

def stream_ai_response(user_input):
    if st.session_state.processor:
        yield from st.session_state.processor.process_with_llm("User", user_input)
    else:
        yield "Please initialize the local model first."

def get_text():
    input_text = st.text_input("You: ", key="input")
    return input_text

user_input = get_text()

if user_input:
    st.session_state.chat_history.append(("human", user_input))
    ai_response = st.write_stream(stream_ai_response(user_input))
    st.session_state.chat_history.append(("ai", ai_response))
    
    # Try to parse the response as JSON and update persona data
    try:
        parsed_response = json.loads(ai_response)
        st.session_state.persona_data.update(parsed_response)
    except json.JSONDecodeError:
        st.warning("Failed to parse AI response as JSON. Persona data not updated.")

# Display chat history
for i in range(len(st.session_state.chat_history)-1, -1, -1):
    if st.session_state.chat_history[i][0] == "ai":
        st.markdown(f"**AI**: {st.session_state.chat_history[i][1]}")
    else:
        st.markdown(f"**You**: {st.session_state.chat_history[i][1]}")

# Persona output
st.subheader("Current Persona Data")
if st.session_state.persona_data:
    st.json(st.session_state.persona_data)
else:
    st.info("No persona data available yet. Start a conversation to generate data.")

# Reset and Save buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Reset Chat and Persona"):
        st.session_state.chat_history = []
        st.session_state.persona_data = {}
        st.success("Chat and persona reset successfully!")
        st.experimental_rerun()

with col2:
    if st.button("Save Persona"):
        if st.session_state.persona_data:
            with open('persona.json', 'w') as f:
                json.dump(st.session_state.persona_data, f, indent=4)
            st.success("Persona saved to persona.json")
        else:
            st.warning("No persona data to save.")