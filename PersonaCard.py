import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint, Ollama
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer, util
import json
import time

class LLMProcessor:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.llm = None
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
        self.stats = {
            "total_tokens": 0,
            "total_responses": 0,
            "average_response_time": 0,
            "total_response_time": 0
        }

    def initialize_local_llm(self, model_name):
        if self.progress_callback:
            self.progress_callback(0, f"Initializing local LLM: {model_name}")
        
        start_time = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
            end_time = time.time()
            if self.progress_callback:
                self.progress_callback(1, f"Local LLM initialized in {end_time - start_time:.2f} seconds")
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(1, f"Failed to initialize local LLM: {str(e)}")
            raise

    def initialize_hpc_llm(self, endpoint_url, api_token):
        if self.progress_callback:
            self.progress_callback(0, "Initializing HPC LLM")
        
        start_time = time.time()
        try:
            self.llm = HuggingFaceEndpoint(
                endpoint_url=endpoint_url,
                huggingfacehub_api_token=api_token,
                max_new_tokens=2096,
                top_k=10,
                top_p=0.95,
                typical_p=0.95,
                temperature=0.01,
                repetition_penalty=1.1,
                streaming=True
            )
            
            end_time = time.time()
            if self.progress_callback:
                self.progress_callback(1, f"HPC LLM initialized in {end_time - start_time:.2f} seconds")
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(1, f"Failed to initialize HPC LLM: {str(e)}")
            raise

    def initialize_ollama(self, model_name):
        if self.progress_callback:
            self.progress_callback(0, f"Initializing Ollama LLM: {model_name}")
        
        start_time = time.time()
        try:
            self.llm = Ollama(
                model=model_name,
                callback_manager=None,
                temperature=0.01
            )
            
            end_time = time.time()
            if self.progress_callback:
                self.progress_callback(1, f"Ollama LLM initialized in {end_time - start_time:.2f} seconds")
        except Exception as e:
            if self.progress_callback:
                self.progress_callback(1, f"Failed to initialize Ollama LLM: {str(e)}")
            raise

    def update_system_message(self, new_message):
        self.system_message = new_message

    def process_with_llm(self, prompt):
        if not self.llm:
            return "LLM not initialized. Please initialize an LLM first."
        
        full_prompt = f"{self.system_message}\n\nUser: {prompt}\n\nAI:"
        try:
            start_time = time.time()
            response = ""
            for chunk in self.llm.stream(full_prompt):
                response += chunk
                yield chunk
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update stats
            self.stats["total_responses"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = self.stats["total_response_time"] / self.stats["total_responses"]
            self.stats["total_tokens"] += len(response.split())
            
            yield "\n\nStats:\n"
            yield f"Response time: {response_time:.2f} seconds\n"
            yield f"Total tokens: {self.stats['total_tokens']}\n"
            yield f"Average response time: {self.stats['average_response_time']:.2f} seconds"
        except Exception as e:
            yield f"Error processing prompt: {str(e)}"

    def test_connection(self):
        test_prompt = "Hello, can you hear me?"
        response = "".join(self.process_with_llm(test_prompt))
        return f"Test connection successful. Response: {response}"

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

# Streamlit app
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
    st.header("LLM Configuration")
    
    llm_option = st.radio("Select LLM", ["Local HuggingFace", "HPC Server", "Ollama"])
    
    if llm_option == "Local HuggingFace":
        local_model = st.text_input("Enter local model name", "gpt2")
        if st.button("Initialize Local LLM"):
            st.session_state.processor.initialize_local_llm(local_model)
    elif llm_option == "HPC Server":
        hpc_endpoint = st.text_input("Enter HPC endpoint URL", "http://ice183:8900")
        hpc_token = st.text_input("Enter HuggingFace API token", type="password")
        if st.button("Initialize HPC LLM"):
            st.session_state.processor.initialize_hpc_llm(hpc_endpoint, hpc_token)
    else:  # Ollama
        ollama_model = st.text_input("Enter Ollama model name", "llama2")
        if st.button("Initialize Ollama LLM"):
            st.session_state.processor.initialize_ollama(ollama_model)
    
    if st.button("Test Connection"):
        st.write(st.session_state.processor.test_connection())
    
    st.header("System Message")
    edited_system_message = st.text_area("Edit System Message:", 
                                         value=st.session_state.processor.system_message,
                                         height=300)
    if st.button("Update System Message"):
        st.session_state.processor.update_system_message(edited_system_message)
        st.success("System message updated!")

    # Reranking
    if st.button("Rerank Persona Data"):
        st.session_state.processor.rerank(st.session_state.persona_data)

    # Persona viewer
    st.header("Current Persona Data")
    persona_data_placeholder = st.empty()

# Main area for chat interface
st.header("Chat Interface")

# Display chat messages
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Chat input
if prompt := st.chat_input("Ask about the persona or for more information"):
    st.chat_message("human").write(prompt)

    with st.chat_message("ai"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in st.session_state.processor.process_with_llm(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    msgs.add_user_message(prompt)
    msgs.add_ai_message(full_response)

    # Try to parse the response as JSON and update persona data
    try:
        parsed_response = st.session_state.processor.parse_json_response(full_response)
        if parsed_response:
            st.session_state.processor.merge_json(st.session_state.persona_data, parsed_response)
            persona_data_placeholder.json(st.session_state.persona_data)
    except json.JSONDecodeError:
        pass

# Display stats
st.sidebar.header("Stats")
st.sidebar.write(f"Total responses: {st.session_state.processor.stats['total_responses']}")
st.sidebar.write(f"Total tokens: {st.session_state.processor.stats['total_tokens']}")
st.sidebar.write(f"Average response time: {st.session_state.processor.stats['average_response_time']:.2f} seconds")

# Download button for persona data
if st.session_state.persona_data:
    st.sidebar.download_button(
        label="Download Persona Data",
        data=json.dumps(st.session_state.persona_data, indent=2),
        file_name="persona_data.json",
        mime="application/json"
    )