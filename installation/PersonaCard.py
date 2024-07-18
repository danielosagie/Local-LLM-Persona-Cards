import sys

if sys.version_info < (3, 9):
    print("This application requires Python 3.9 or higher.")
    sys.exit(1)


import streamlit as st
import streamlit_survey as ss
import json
import pandas as pd
import os
from datetime import datetime
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint, Ollama
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from sentence_transformers import SentenceTransformer, util
import time
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import HumanMessage
import queue

# Set page config
st.set_page_config(page_title="LLM Persona Cards", page_icon="🃏", layout="wide")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

class StreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.queue.put(token)

class LLMProcessor:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.llm = None
        self.ongoing_persona = {}
        self.system_message = '''
        You are an AI assistant helping to create a persona card. Respond to the user's survey input and chat input by providing and/or matching relevant information in JSON format. Understand the military spouse experience and be creative to help the spouse understand their history/potential. Include categories such as Education, Experience, Skills, Strengths, Goals, and Values. Keep your responses concise and relevant.

        Example sample response format:
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
        self.stream_handler = None
        self.response_queue = queue.Queue()

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

    
    def load_persona(self, filename):
        try:
            full_path = os.path.join("generated_personas", filename)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File not found: {full_path}")
            with open(full_path, 'r') as f:
                loaded_persona = json.load(f)
            self.ongoing_persona = loaded_persona
            return loaded_persona
        except json.JSONDecodeError as e:
            st.error(f"Error parsing JSON in {filename}: {str(e)}")
        except Exception as e:
            st.error(f"Error loading persona {filename}: {str(e)}")
        return None
        
    def update_ongoing_persona(self, new_data):
        if isinstance(new_data, dict):
            for category, items in new_data.items():
                if category not in self.ongoing_persona:
                    self.ongoing_persona[category] = []
                if isinstance(items, list):
                    self.ongoing_persona[category] = list(set(self.ongoing_persona[category] + items))
                elif isinstance(items, str):
                    if items not in self.ongoing_persona[category]:
                        self.ongoing_persona[category].append(items)
        else:
            st.warning(f"Received non-dict data to update persona: {type(new_data)}")

    def process_with_llm(self, prompt):
        if not self.llm:
            yield "LLM not initialized. Please initialize an LLM first."
            return

        full_prompt = f"{self.system_message}\n\nUser: {prompt}\n\nAI:"
        try:
            start_time = time.time()
        
            full_response = ""
            # Check if the LLM supports streaming
            if hasattr(self.llm, 'stream') and callable(self.llm.stream):
                for chunk in self.llm.stream(full_prompt):
                    full_response += chunk
                    yield chunk
                    self.stats["total_tokens"] += len(chunk.split())
            else:
                # Fallback for non-streaming LLMs
                full_response = self.llm.invoke(full_prompt)
                yield full_response
                self.stats["total_tokens"] += len(full_response.split())

            end_time = time.time()
            response_time = end_time - start_time
        
            # Update stats
            self.stats["total_responses"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = self.stats["total_response_time"] / self.stats["total_responses"]
        
            # Try to parse the response as JSON and update the ongoing persona
            try:
                parsed_response = json.loads(full_response)
                self.update_ongoing_persona(parsed_response)
            except json.JSONDecodeError:
                # If it's not valid JSON, try to extract information anyway
                self.extract_info_from_text(full_response)
        
        except Exception as e:
            yield f"Error processing prompt: {str(e)}"

    def extract_info_from_text(self, text):
        categories = ["Name", "Education", "Experience", "Skills", "Strengths", "Goals", "Values"]
        extracted_data = {}
        for category in categories:
            if category in text:
                start = text.index(category) + len(category)
                end = text.find("\n", start)
                if end == -1:
                    end = len(text)
                items = text[start:end].strip(": ").split(", ")
                extracted_data[category] = items
        self.update_ongoing_persona(extracted_data)
    
    def compare_with_job_description(self, persona_data, job_description):
        prompt = f"""
        Given the following persona data:
        {json.dumps(persona_data, indent=2)}

        And the following job description:
        {job_description}

        Provide a detailed comparison analysis in JSON format with the following structure:
        {{
            "match_percentage": float,
            "strengths": [list of strengths that align with the job],
            "gaps": [list of areas where the persona lacks required skills or experience],
            "recommendations": [list of suggestions for improving the match]
        }}
        """
        response = self.process_with_llm(prompt)
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse comparison result"}

    def export_chat_history(self, msgs):
        chat_history = []
        for msg in msgs.messages:
            chat_history.append({
                "type": msg.type,
                "content": msg.content
            })
        return json.dumps(chat_history, indent=2)

    def import_chat_history(self, chat_history_json, msgs):
        chat_history = json.loads(chat_history_json)
        msgs.clear()
        for msg in chat_history:
            if msg["type"] == "human":
                msgs.add_user_message(msg["content"])
            elif msg["type"] == "ai":
                msgs.add_ai_message(msg["content"])
    
    def rerank(self, data):
        st.write("Enter a goal or career choice that will rerank the list of labels:")
        desired_job = st.text_input("Career/Goal:")
        if desired_job:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            job_embedding = model.encode(desired_job, convert_to_tensor=True)

            def rank_items_by_relevance(data, job_embedding):
                ranked_data = {}
                for category, items in data.items():
                    if isinstance(items, list):
                        item_embeddings = model.encode(items, convert_to_tensor=True)
                        scores = util.pytorch_cos_sim(job_embedding, item_embeddings)[0]
                        ranked_items = [item for _, item in sorted(zip(scores, items), reverse=True)]
                        ranked_data[category] = ranked_items
                    else:
                        ranked_data[category] = items
                return ranked_data

            ranked_data = rank_items_by_relevance(data, job_embedding)
            return ranked_data

    def validate_json_files():
        persona_dir = "generated_personas"
        for filename in os.listdir(persona_dir):
            if filename.endswith('.json'):
                try:
                    with open(os.path.join(persona_dir, filename), 'r') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON in file {filename}: {e}")

    def test_connection(self):
        test_prompt = "Hello, can you hear me?"
        response = self.process_with_llm(test_prompt)
        return f"Test connection successful. Response: {response}"

# Streamlit app
st.markdown('<h1 class="dashboard_title">🃏 LLM Persona Cards</h1>', unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = LLMProcessor(lambda p, m: st.sidebar.progress(p, m))
    st.session_state.processor.initialize_ollama("llama3")  # Initialize Ollama llama3 by default
if 'persona_data' not in st.session_state:
    st.session_state.persona_data = {}
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = {}

# Create tabs
survey_tab, persona_gen_tab, persona_viewer_tab = st.tabs(["Survey", "Persona Generator", "Persona Viewer"])

# Survey Tab
with survey_tab:
    st.markdown('<h2 class="section_title">Military Spouse Experience Survey</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a survey response file", type="json")
    if uploaded_file is not None:
        survey_data = json.load(uploaded_file)
        st.session_state['survey_data'] = survey_data
        st.success("File uploaded successfully!")
    
    # Initialize or get the survey from session state
    if 'survey' not in st.session_state:
        st.session_state['survey'] = ss.StreamlitSurvey("Military Spouse Experience Survey")
    survey = st.session_state['survey']

    # Create paged survey
    pages = survey.pages(5, on_submit=lambda: st.success("Your responses have been recorded. Thank you!"))

    with pages:
        if pages.current == 0:
            st.subheader("Personal Information")
            survey.text_input("What is your name?")
            
            st.subheader("Education")
            education = survey.radio(
                "What is your highest level of education?",
                options=["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"],
                horizontal=True
            )

            if education != "High School":
                survey.text_area(
                    "Please describe your most significant educational experiences or achievements:"
                )

                survey.multiselect(
                    "What challenges, if any, have you faced in your education due to being a military spouse? (Select all that apply)",
                    options=[
                        "Frequent relocations",
                        "Difficulty transferring credits",
                        "Limited time due to family responsibilities",
                        "Financial constraints",
                        "Lack of consistent childcare",
                        "Other"
                    ]
                )

        elif pages.current == 1:
            st.subheader("Work Experience")
            work_status = survey.radio(
                "What is your current employment status?",
                options=["Employed full-time", "Employed part-time", "Self-employed", "Unemployed, seeking work", "Not in the workforce"],
                horizontal=True
            )

            if work_status in ["Employed full-time", "Employed part-time", "Self-employed"]:
                survey.text_input("What field do you work in?")
                
                survey.text_area(
                    "Please describe your most significant work experiences or achievements:"
                )

                survey.multiselect(
                    "What challenges, if any, have you faced in your career due to being a military spouse? (Select all that apply)",
                    options=[
                        "Frequent job changes due to relocations",
                        "Limited job opportunities in duty station locations",
                        "Difficulty advancing in career",
                        "Balancing work with family responsibilities",
                        "Employer bias against military spouses",
                        "Licensing or certification issues across states",
                        "Other"
                    ]
                )

        elif pages.current == 2:
            st.subheader("Military Spouse Daily Life")
            pcs_count = survey.slider(
                "How many times have you PCSed (Permanent Change of Station) as a military spouse?",
                min_value=0,
                max_value=20,
                value=1
            )

            if pcs_count > 0:
                survey.multiselect(
                    "What challenges have you faced during PCS moves? (Select all that apply)",
                    options=[
                        "Finding new housing",
                        "Children's education transitions",
                        "Personal career disruptions",
                        "Making new friends/building community",
                        "Managing household goods shipments",
                        "Emotional stress",
                        "Financial strain",
                        "Other"
                    ]
                )

            survey.multiselect(
                "Which of the following tasks do you regularly manage in your household? (Select all that apply)",
                options=[
                    "Budgeting and finances",
                    "Childcare and education",
                    "Home maintenance",
                    "Healthcare management",
                    "Deployment preparation",
                    "Community involvement",
                    "Support group participation",
                    "Personal career development",
                    "Other"
                ]
            )

            parenting = survey.radio(
                "Are you a parent?",
                options=["Yes", "No"],
                horizontal=True
            )

            if parenting == "Yes":
                survey.multiselect(
                    "What unique challenges do you face as a military parent? (Select all that apply)",
                    options=[
                        "Explaining deployments to children",
                        "Managing children's emotions during separations",
                        "Finding consistent childcare",
                        "Navigating school changes during PCS",
                        "Balancing parenting with military lifestyle demands",
                        "Maintaining family traditions despite frequent moves",
                        "Other"
                    ]
                )

        elif pages.current == 3:
            st.subheader("General Experience")
            survey.text_area(
                "What do you find most rewarding about being a military spouse?"
            )

            survey.text_area(
                "What is the biggest challenge you face as a military spouse?"
            )

            survey.text_area(
                "What kind of support or resources do you wish were more readily available to military spouses?"
            )

        
    # Add upload/download functionality at the top
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Complete Survey"):
            st.session_state['survey_completed'] = True
            st.session_state['survey_data'] = survey.to_json()
        
            # Save to cached_responses folder
            os.makedirs("cached_responses", exist_ok=True)
            filename = f"cached_responses/survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(st.session_state['survey_data'], f)
        
            st.success(f"Survey completed! Responses saved to {filename}")
            st.info("You can now proceed to the Persona Generator.")
    
    with col2:
        if 'survey_completed' in st.session_state and st.session_state['survey_completed']:
            if st.download_button("Download Survey Responses", 
                                  data=json.dumps(st.session_state['survey_data'], indent=2),
                                  file_name="survey_responses.json",
                                  mime="application/json"):
                st.success("Survey responses downloaded!")

# Persona Generator Tab
with persona_gen_tab:
    st.markdown('<h2 class="section_title">Persona Generator</h2>', unsafe_allow_html=True)

    # Chat Interface
    st.subheader("Chat Interface")
    
    # Initialize StreamlitChatMessageHistory
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")
    
    # Save and Download buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Save Ongoing Persona"):
            if st.session_state.processor.ongoing_persona:
                name = st.session_state.processor.ongoing_persona.get("Name", ["Unknown"])
                if isinstance(name, list):
                    name = name[0]
                elif isinstance(name, str):
                    name = name
                else:
                    name = "Unknown"
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                version = 1
                while True:
                    persona_filename = f"generated_personas/{name}_v{version}_{timestamp}.json"
                    if not os.path.exists(persona_filename):
                        break
                    version += 1

                os.makedirs("generated_personas", exist_ok=True)
                with open(persona_filename, 'w') as f:
                    json.dump(st.session_state.processor.ongoing_persona, f, indent=2)
                
                st.success(f"Ongoing persona saved as {persona_filename}")
            else:
                st.warning("No ongoing persona to save. Please generate a persona first.")

    with col2:
        if st.session_state.processor.ongoing_persona:
            if st.download_button(
                "Download Current Persona", 
                data=json.dumps(st.session_state.processor.ongoing_persona, indent=2),
                file_name="current_persona.json",
                mime="application/json"
            ):
                st.success("Current persona downloaded!")
        else:
            st.warning("No ongoing persona to download. Please generate a persona first.")
            
    with col3:
        if len(msgs.messages) == 0 or st.button("Reset chat history", key="reset_chat_persona_gen"):
            msgs.clear()
            msgs.add_ai_message("How can I help you create or modify the persona?")
    


    # Display chat messages
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Chat input
    if prompt := st.chat_input("Ask about the persona or for more information"):
        st.chat_message("human").write(prompt)

        with st.chat_message("ai"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in st.session_state.processor.process_with_llm(prompt):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        msgs.add_user_message(prompt)
        msgs.add_ai_message(full_response)

        # Update ongoing persona
        try:
            new_data = json.loads(full_response)
            st.session_state.processor.update_ongoing_persona(new_data)
        except json.JSONDecodeError:
            st.session_state.processor.extract_info_from_text(full_response)

    # Control Panel
    st.subheader("Control Panel")
    
    with st.expander("Process Survey Responses", expanded=False):
        # Get list of cached survey responses
        cached_responses_dir = "cached_responses"
        if os.path.exists(cached_responses_dir):
            cached_files = [f for f in os.listdir(cached_responses_dir) if f.endswith('.json')]
            cached_files.sort(key=lambda x: os.path.getmtime(os.path.join(cached_responses_dir, x)), reverse=True)
        else:
            cached_files = []

        # Dropdown to select cached response
        selected_cache = st.selectbox("Select cached survey response:", ["Current Survey"] + cached_files)

        if selected_cache == "Current Survey":
            survey_data = st.session_state.get('survey_data', {})
        elif selected_cache in cached_files:
            with open(os.path.join(cached_responses_dir, selected_cache), 'r') as f:
                survey_data = json.load(f)
        else:
            survey_data = {}

        if survey_data:
            # Check if survey_data is a string (JSON) and convert it to a dictionary if needed
            if isinstance(survey_data, str):
                try:
                    survey_data = json.loads(survey_data)
                except json.JSONDecodeError:
                    st.error("Error: Survey data is not in valid JSON format.")
                    survey_data = {}
            
            if isinstance(survey_data, dict) and survey_data:
                # Convert the survey data to a DataFrame for the data editor
                df = pd.DataFrame(survey_data.items(), columns=['Question', 'Response'])
                
                # Create a data editor for the survey responses
                edited_df = st.data_editor(
                    df,
                    column_config={
                        "Question": st.column_config.TextColumn("Survey Question"),
                        "Response": st.column_config.TextColumn("Your Response"),
                    },
                    disabled=["Question"],
                    hide_index=True,
                    num_rows="dynamic",
                    key="survey_data_editor"
                )
                
                if st.button("Process Selected Survey Responses"):
                    # Convert the edited DataFrame back to a dictionary
                    selected_data = dict(zip(edited_df['Question'], edited_df['Response']))
                    
                    # Convert selected survey data to a string format
                    selected_data_str = json.dumps(selected_data, indent=2)
                    
                    # Use the custom prompt
                    prompt = st.session_state.survey_processing_prompt.format(survey_data=selected_data_str)
                    
                    # Add the user message to the chat history
                    msgs.add_user_message({st.session_state.survey_processing_prompt})
                    
                    # Process the prompt with the LLM
                    full_response = ""
                    for chunk in st.session_state.processor.process_with_llm(prompt):
                        full_response += chunk
                    
                    # Add the AI response to the chat history
                    msgs.add_ai_message(full_response)
                    
                    # Try to parse the response as JSON
                    try:
                        processed_persona = json.loads(full_response)
                        st.session_state.processor.update_ongoing_persona(processed_persona)
                        st.success("Persona updated based on survey responses!")
                    except json.JSONDecodeError:
                        st.warning("The LLM response was not in valid JSON format. Using text extraction method.")
                        st.session_state.processor.extract_info_from_text(full_response)
                    
                    # Force a rerun to update the chat display
                    st.rerun()

            else:
                st.warning("Survey data is empty or in an unexpected format.")
        else:
            st.warning("No survey data available. Please complete the survey first or select a cached response.")
    
    
    with st.expander("Rerank Persona"):
        desired_job = st.text_input("Enter desired job for reranking:")
        if st.button("Rerank"):
            reranked_persona = st.session_state.processor.rerank(st.session_state.processor.ongoing_persona)
            st.json(reranked_persona)
            
            # Save reranked persona
            name = st.session_state.processor.ongoing_persona.get("Name", ["Unknown"])[0]
            rerank_filename = f"reranked_personas/{name}_reranked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("reranked_personas", exist_ok=True)
            with open(rerank_filename, 'w') as f:
                json.dump(reranked_persona, f, indent=2)
            
            st.success(f"Reranked persona saved as {rerank_filename}")

    with st.expander("Compare with Job Description"):
        job_description = st.text_area("Enter Job Description", height=200, key="job_description_persona_gen")
        if st.button("Compare"):
            comparison_result = st.session_state.processor.compare_with_job_description(
                st.session_state.processor.ongoing_persona, job_description
            )
            st.json(comparison_result)
        
            # Save comparison result
            name = st.session_state.processor.ongoing_persona.get("Name", ["Unknown"])[0]
            comparison_filename = f"job_comparisons/{name}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("job_comparisons", exist_ok=True)
            with open(comparison_filename, 'w') as f:
                json.dump(comparison_result, f, indent=2)
        
            st.success(f"Comparison result saved as {comparison_filename}")

    # Display ongoing persona
    st.subheader("Current Ongoing Persona")
    st.json(st.session_state.processor.ongoing_persona)


    # LLM Configuration
    st.sidebar.markdown('<h2 class="section_title">LLM Configuration</h2>', unsafe_allow_html=True)
    
    llm_option = st.sidebar.radio("Select LLM", ["Local HuggingFace", "HPC Server", "Ollama"])
    
    if llm_option == "Local HuggingFace":
        local_model = st.sidebar.text_input("Enter local model name", "gpt2")
        if st.sidebar.button("Initialize Local LLM"):
            st.session_state.processor.initialize_local_llm(local_model)
    elif llm_option == "HPC Server":
        hpc_endpoint = st.sidebar.text_input("Enter HPC endpoint URL", "http://ice183:8900")
        hpc_token = st.sidebar.text_input("Enter HuggingFace API token", type="password")
        if st.sidebar.button("Initialize HPC LLM"):
            st.session_state.processor.initialize_hpc_llm(hpc_endpoint, hpc_token)
    else:  # Ollama
        ollama_model = st.sidebar.text_input("Enter Ollama model name", "llama3")
        if st.sidebar.button("Initialize Ollama LLM"):
            st.session_state.processor.initialize_ollama(ollama_model)
    
    if st.sidebar.button("Test Connection"):
        st.sidebar.write(st.session_state.processor.test_connection())
    
    st.sidebar.header("System Message")
    edited_system_message = st.sidebar.text_area("Edit System Message:", 
                                         value=st.session_state.processor.system_message,
                                         height=300)
    if st.sidebar.button("Update System Message"):
        st.session_state.processor.update_system_message(edited_system_message)
        st.sidebar.success("System message updated!")
        
    st.sidebar.header("Survey Processing Prompt")
    if 'survey_processing_prompt' not in st.session_state:
        st.session_state.survey_processing_prompt = """You are a career coach/ERP specialist. Based on the following survey responses:

        {survey_data}

        Create a detailed persona. Include categories such as Name, Education, Experience, Skills, Strengths, Goals, and Values. If there is anything else that you want to say that you think doesn't directly fit into these categories, I want you to be creative and empathetic and reframe the circumstances and underlying qualities as something that can fit into these categories. Keep your response concise and relevant, and format it as a JSON object.
        """
    
    survey_processing_prompt = st.sidebar.text_area(
        "Edit Survey Processing Prompt:",
        value=st.session_state.survey_processing_prompt,
        height=300
    )
    
    if st.sidebar.button("Update Survey Processing Prompt"):
        st.session_state.survey_processing_prompt = survey_processing_prompt
        st.sidebar.success("Survey processing prompt updated!")

with persona_viewer_tab:
    st.markdown('<h2 class="section_title">Spouse-Facing Persona Viewer</h2>', unsafe_allow_html=True)
    
    #Checks if there are any personas at all
    persona_dir = "generated_personas"
    if not os.path.exists(persona_dir):
        st.error(f"The directory {persona_dir} does not exist.")
    elif not os.listdir(persona_dir):
        st.warning(f"The directory {persona_dir} is empty. No personas available.")
    else:
        persona_files = [f for f in os.listdir(persona_dir) if f.endswith('.json')]
        st.write(f"Found {len(persona_files)} persona files: {', '.join(persona_files)}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Load persona data
        persona_files = [f for f in os.listdir("generated_personas") if f.endswith('.json')]
        persona_files.sort(key=lambda x: os.path.getmtime(os.path.join("generated_personas", x)), reverse=True)
        
        auto_select_latest = st.checkbox("Automatically select most recent persona", value=True)
        
        if auto_select_latest and persona_files:
            selected_persona = persona_files[0]
        else:
            selected_persona = st.selectbox("Select a persona to view", ["None"] + persona_files)

        if st.button("Debug: Check Current State"):
            st.write("Current persona viewer data:", st.session_state.get('persona_viewer_data'))
            st.write("Selected persona:", selected_persona)

    with col2:
        if st.button("Apply Selected Persona"):
            if selected_persona and selected_persona != "None":
                try:
                    full_path = os.path.join("generated_personas", selected_persona)
                    st.write(f"Attempting to load: {full_path}")
                    with open(full_path, 'r') as f:
                        persona = json.load(f)
                    st.session_state['persona_viewer_data'] = persona
                    st.success(f"Applied persona: {selected_persona}")
                except Exception as e:
                    st.error(f"Failed to load the selected persona: {str(e)}")
            else:
                st.warning("No persona selected.")
        
        if st.button("Debug: Check Loaded Persona"):
            st.write("Loaded persona data:", st.session_state.get('persona_viewer_data'))

    if 'persona_viewer_data' in st.session_state and st.session_state['persona_viewer_data']:
        persona = st.session_state['persona_viewer_data']
        
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # You can add an image here if you have one for the persona
            st.image("GTRI Logo.png", use_column_width=True)
        
        with col2:
            st.header(persona.get("Name", "Unknown"))
            st.markdown(persona.get("Description", "No description available."))
        
        def display_tags(items, title, initial_display=None):
            st.markdown(f'<h3 class="section_title">{title}</h3>', unsafe_allow_html=True)
            if isinstance(items, list):
                tag_html = "".join([f'<span class="tag">{item}</span>' for item in items[:initial_display]])
                st.markdown(tag_html, unsafe_allow_html=True)
            
                if initial_display and len(items) > initial_display:
                    with st.expander("Show more"):
                        more_tags = "".join([f'<span class="tag">{item}</span>' for item in items[initial_display:]])
                        st.markdown(more_tags, unsafe_allow_html=True)
            elif isinstance(items, str):
                st.markdown(f'<span class="tag">{items}</span>', unsafe_allow_html=True)
            else:
                st.write("No data available")

        display_tags(persona.get("Education", []), "Education")
        display_tags(persona.get("Experience", []) + persona.get("Qualifications", []), "Experience & Qualifications")
        display_tags(persona.get("Skills", []), "Skills")
        display_tags(persona.get("Goals", []), "Goals", 6)
        display_tags(persona.get("Values", []), "Values", 6)
        display_tags(persona.get("Strengths", []), "Strengths")

        if "Relevant Jobs" in persona:
            st.markdown('<h3 class="section-header">Relevant Jobs</h3>', unsafe_allow_html=True)
            job_cols = st.columns(3)
            for i, job in enumerate(persona["Relevant Jobs"].items()):
                with job_cols[i % 3]:
                    st.metric(job[0], job[1])
        
        if "Recommendations" in persona:
            display_tags(persona["Recommendations"], "Recommendations")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No persona data available. Select a persona and click 'Apply Selected Persona' to view it.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p class="persona_detail">© 2023 LLM Persona Cards</p>', unsafe_allow_html=True)