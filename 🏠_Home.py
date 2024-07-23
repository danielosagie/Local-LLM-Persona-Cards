import streamlit as st

st.set_page_config(page_title="LLM Persona Cards", page_icon="🃏", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown('<h1 class="dashboard_title">🃏 LLM Persona Cards</h1>', unsafe_allow_html=True)

st.write("""
Welcome to LLM Persona Cards! This application helps you create and manage personas based on survey data and AI-generated insights.

Navigate through the pages using the sidebar:
- 📊 Survey: Collect and manage survey responses
- 🎭 Persona Generator: Create and modify personas
- 👤 Persona Viewer: View and analyze generated personas

Get started by selecting a page from the sidebar.
""")

# You can add any global settings or state initialization here
if 'processor' not in st.session_state:
    from persona_processor import PersonaProcessor
    st.session_state.processor = PersonaProcessor()

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="persona_detail">© 2023 LLM Persona Cards</p>', unsafe_allow_html=True)