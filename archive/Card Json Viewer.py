import streamlit as st
import json
import re

def load_data(file_content):
    data = json.loads(file_content)
    return data

def highlight_text(text, items):
    highlighted_text = text
    for item in items:
        pattern = re.compile(re.escape(item), re.IGNORECASE)
        highlighted_text = pattern.sub(f'<span style="background-color: #FFEDB7">{item}</span>', highlighted_text)
    return highlighted_text

def display_chat_history(data, highlight_enabled):
    st.subheader("Chat History")
    cumulative_summary = {
        "Education": [],
        "Experience": [],
        "Skills": [],
        "Strengths": [],
        "Goals": [],
        "Values": []
    }
    
    for item in data[1:-1]:  # Exclude the first (persona) and last (summary) items
        st.write(f"**Q: {item['question']}**")
        
        processed = json.loads(item['processed_response'])
        new_additions = {}
        
        for category, items in processed.items():
            new_items = [item for item in items if item not in cumulative_summary[category]]
            if new_items:
                new_additions[category] = new_items
                cumulative_summary[category].extend(new_items)
        
        # Highlight new additions in the response if enabled
        if highlight_enabled:
            highlighted_response = item['response']
            for category, items in new_additions.items():
                highlighted_response = highlight_text(highlighted_response, items)
            st.write("A:", unsafe_allow_html=True)
            st.markdown(highlighted_response, unsafe_allow_html=True)
        else:
            st.write("A:", item['response'])
        
        if new_additions:
            st.write("**New Additions:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Text Format")
                for category, items in new_additions.items():
                    st.write(f"**{category}:**")
                    for item in items:
                        st.markdown(f"- **{item}**")
            
            with col2:
                st.write("JSON Format")
                st.json(new_additions)
        
        st.write("---")

def display_persona_summary(data):
    st.subheader("Persona Summary")
    
    summary = data[-1]  # The last item is the summary
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Text Format**")
        for category, items in summary.items():
            st.write(f"**{category}:**")
            for item in items:
                st.write(f"- {item}")
            st.write()
    
    with col2:
        st.write("**JSON Format**")
        st.json(summary)

def main():
    st.title("Persona Chat History and Summary")

    uploaded_file = st.file_uploader("Choose a JSON file", type="json")
    if uploaded_file is not None:
        file_content = uploaded_file.getvalue().decode("utf-8")
        data = load_data(file_content)

        st.write(f"**Persona:** {data[0]['persona']}")
        st.write("---")

        # Add a toggle for highlighting
        highlight_enabled = st.checkbox("Enable highlighting", value=True)

        display_chat_history(data, highlight_enabled)
        display_persona_summary(data)

if __name__ == "__main__":
    main()