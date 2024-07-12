import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sentence_transformers import SentenceTransformer, util
import json
import time
import base64

class LLMProcessor:
    def __init__(self, progress_callback=None):
        self.huggingface_api_token = "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"  # Replace with your HuggingFace API token
        self.endpoint_url = "http://ice183:8900"  # Replace with your endpoint URL
        
        if progress_callback:
            progress_callback(0, "Initializing LLM...")
        
        start_time = time.time()
        self.llm = HuggingFaceEndpoint(
            endpoint_url=self.endpoint_url,
            max_new_tokens=2096,
            top_k=10,
            top_p=0.95,
            typical_p=0.95,
            temperature=0.01,
            repetition_penalty=1.1,
            huggingfacehub_api_token=self.huggingface_api_token
        )
        
        end_time = time.time()
        if progress_callback:
            progress_callback(1, f"LLM initialized in {end_time - start_time:.2f} seconds")

        self.system_message = '''
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

    def update_system_message(self, new_message):
        self.system_message = new_message

    def process_with_llm(self, prompt):
        full_prompt = f"{self.system_message}\n\nUser: {prompt}\n\nAI:"
        return self.llm.invoke(full_prompt)

    def merge_json(self, existing_data, new_data):
        for label, new_content in new_data.items():
            if label in existing_data:
                if isinstance(existing_data[label], list):
                    if isinstance(new_content, list):
                        existing_data[label].extend(item for item in new_content if item not in existing_data[label])
                    else:
                        if new_content not in existing_data[label]:
                            existing_data[label].append(new_content)
                else:
                    if existing_data[label] != new_content:
                        existing_data[label] = [existing_data[label], new_content] if isinstance(new_content, str) else [existing_data[label]] + new_content
            else:
                existing_data[label] = new_content

    def parse_json_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error: The response is not valid JSON.")
            return None

    def rerank(self, data):
        st.write("Enter a goal or career choice that will rerank the list of labels:")
        desired_job = st.text_input("Career/Goal:")
        if desired_job:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            job_embedding = model.encode(desired_job, convert_to_tensor=True)

            def rank_items_by_relevance(data, job_embedding):
                ranked_data = {}
                for category, items in data.items():
                    item_embeddings = model.encode(items, convert_to_tensor=True)
                    scores = util.pytorch_cos_sim(job_embedding, item_embeddings)[0]
                    ranked_items = [item for _, item in sorted(zip(scores, items), reverse=True)]
                    ranked_data[category] = ranked_items
                return ranked_data

            ranked_data = rank_items_by_relevance(data, job_embedding)
            st.json(ranked_data)

def get_table_download_link(df):
    json_string = df.to_json(orient='records')
    b64 = base64.b64encode(json_string.encode()).decode()
    return f'<a href="data:application/json;base64,{b64}" download="persona_data.json">Download Persona Data</a>'

st.set_page_config(page_title="LLM Persona Cards", page_icon="🦜")
st.title("🦜 LLM Persona Cards")

if "processor" not in st.session_state:
    st.session_state.processor = LLMProcessor(lambda p, m: st.sidebar.progress(p, m))

if "persona_data" not in st.session_state:
    st.session_state.persona_data = {}

# Initialize StreamlitChatMessageHistory
msgs = StreamlitChatMessageHistory(key="chat_messages")
memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you create a persona?")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # System Message Editor
    with st.expander("Edit System Message", expanded=False):
        edited_system_message = st.text_area("System Message:", 
                                             value=st.session_state.processor.system_message,
                                             height=300)
        if st.button("Update System Message"):
            st.session_state.processor.update_system_message(edited_system_message)
            st.success("System message updated!")

    # Reranking
    if st.button("Rerank Persona Data"):
        st.session_state.processor.rerank(st.session_state.persona_data)
        
    
    # Persona viewer
    st.subheader("Current Persona Data")
    if st.session_state.persona_data:
        st.json(st.session_state.persona_data)
        st.markdown(get_table_download_link(st.session_state.persona_data), unsafe_allow_html=True)
    else:
        st.info("No persona data available yet. Start a conversation to generate data.")

# Display chat messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Chat input
if prompt := st.chat_input("Ask about the persona or for more information"):
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        response = st.session_state.processor.process_with_llm(prompt)
        st.write(response)

    msgs.add_user_message(prompt)
    msgs.add_ai_message(response)

    # Try to parse the response as JSON and update persona data
    try:
        parsed_response = st.session_state.processor.parse_json_response(response)
        if parsed_response:
            st.session_state.processor.merge_json(st.session_state.persona_data, parsed_response)
    except json.JSONDecodeError:
        pass
