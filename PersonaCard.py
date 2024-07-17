import streamlit as st
import streamlit_survey as ss
import json
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

# Add this near the top of your file, after the imports
st.set_page_config(page_title="LLM Persona Cards", page_icon="🃏", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .main-container {
        background-color: #2D2D2D;
        padding: 20px;
        border-radius: 10px;
    }
    .tag {
        background-color: white;
        color: black;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 5px;
        display: inline-block;
    }
    .section-header {
        color: #CCCCCC;
    }
</style>
""", unsafe_allow_html=True)

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

    def process_with_llm(self, prompt):
        if not self.llm:
            yield "LLM not initialized. Please initialize an LLM first."
            return

        full_prompt = f"{self.system_message}\n\nUser: {prompt}\n\nAI:"
        try:
            start_time = time.time()
        
            # Check if the LLM supports streaming
            if hasattr(self.llm, 'stream') and callable(self.llm.stream):
                for chunk in self.llm.stream(full_prompt):
                    yield chunk
                    self.stats["total_tokens"] += len(chunk.split())
            else:
                # Fallback for non-streaming LLMs
                response = self.llm.invoke(full_prompt)
                yield response
                self.stats["total_tokens"] += len(response.split())

            end_time = time.time()
            response_time = end_time - start_time
        
            # Update stats
            self.stats["total_responses"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = self.stats["total_response_time"] / self.stats["total_responses"]
        
            # Try to parse the response as JSON and update the ongoing persona
            try:
                parsed_response = json.loads(response)
                self.update_ongoing_persona(parsed_response)
            except json.JSONDecodeError:
                # If it's not valid JSON, try to extract information anyway
                self.extract_info_from_text(response)
        
        except Exception as e:
            yield f"Error processing prompt: {str(e)}"
    
    def load_persona(self, filename):
        try:
            with open(os.path.join("generated_personas", filename), 'r') as f:
                self.ongoing_persona = json.load(f)
            return self.ongoing_persona
        except Exception as e:
            print(f"Error loading persona: {e}")
            return None
        
    def update_ongoing_persona(self, new_data):
        for category, items in new_data.items():
            if category not in self.ongoing_persona:
                self.ongoing_persona[category] = []
            if isinstance(items, list):
                self.ongoing_persona[category].extend([item for item in items if item not in self.ongoing_persona[category]])
            else:
                if items not in self.ongoing_persona[category]:
                    self.ongoing_persona[category].append(items)

    def extract_info_from_text(self, text):
        categories = ["Education", "Experience", "Skills", "Strengths", "Goals", "Values"]
        for category in categories:
            if category in text:
                start = text.index(category) + len(category)
                end = text.find("\n", start)
                if end == -1:
                    end = len(text)
                items = text[start:end].strip(": ").split(", ")
                self.update_ongoing_persona({category: items})

    
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

    def test_connection(self):
        test_prompt = "Hello, can you hear me?"
        response = self.process_with_llm(test_prompt)
        return f"Test connection successful. Response: {response}"

# Streamlit app
st.title("🃏 LLM Persona Cards")

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
    st.header("Military Spouse Experience Survey")
    
    # Add upload/download functionality at the top
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a survey response file", type="json")
        if uploaded_file is not None:
            survey_data = json.load(uploaded_file)
            st.session_state['survey_data'] = survey_data
            st.success("File uploaded successfully!")
    
    with col2:
        if 'survey_completed' in st.session_state and st.session_state['survey_completed']:
            if st.download_button("Download Survey Responses", 
                                  data=json.dumps(st.session_state['survey_data'], indent=2),
                                  file_name="survey_responses.json",
                                  mime="application/json"):
                st.success("Survey responses downloaded!")
    
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

    # Add completion button at the end
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

# Persona Generator Tab
with persona_gen_tab:
    st.header("Persona Generator")

    # Chat Interface
    st.subheader("Chat Interface")
    
    # Initialize StreamlitChatMessageHistory
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

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

    # Save and Download buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Ongoing Persona"):
            name = st.session_state.processor.ongoing_persona.get("Name", ["Unknown"])[0]
            version = 1
            while True:
                persona_filename = f"generated_personas/{name}_v{version}_{datetime.now().strftime('%Y%m%d')}.json"
                if not os.path.exists(persona_filename):
                    break
                version += 1

            os.makedirs("generated_personas", exist_ok=True)
            with open(persona_filename, 'w') as f:
                json.dump(st.session_state.processor.ongoing_persona, f, indent=2)
            
            st.success(f"Ongoing persona saved as {persona_filename}")

    with col2:
        if st.download_button("Download Current Persona", 
                               data=json.dumps(st.session_state.processor.ongoing_persona, indent=2),
                               file_name="current_persona.json",
                               mime="application/json"):
            st.success("Current persona downloaded!")

    # Display ongoing persona
    st.subheader("Current Ongoing Persona")
    st.json(st.session_state.processor.ongoing_persona)


    st.header("Chat Interface")
    
    # Initialize StreamlitChatMessageHistory
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

    if len(msgs.messages) == 0 or st.button("Reset chat history"):
        msgs.clear()
        msgs.add_ai_message("How can I help you create or modify the persona?")

    # Display chat messages
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Chat input with unique key
    if prompt := st.chat_input("Ask about the persona or for more information", key="chat_input_persona_gen"):
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

    # LLM Configuration
    st.sidebar.header("LLM Configuration")
    
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

with persona_viewer_tab:
    st.header("Spouse-Facing Persona Viewer")
    
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

    with col2:
        if st.button("Apply Selected Persona"):
            if selected_persona and selected_persona != "None":
                persona = st.session_state.processor.load_persona(selected_persona)
                if persona:
                    st.session_state['persona_viewer_data'] = persona
                    st.success(f"Applied persona: {selected_persona}")
                else:
                    st.error("Failed to load the selected persona.")
            else:
                st.warning("No persona selected.")

    if 'persona_viewer_data' in st.session_state and st.session_state['persona_viewer_data']:
        persona = st.session_state['persona_viewer_data']
        
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # You can add an image here if you have one for the persona
            st.image("GTRI Logo.png", use_column_width=True)
        
        with col2:
            st.header(persona.get("Name", ["Unknown"])[0] if isinstance(persona.get("Name"), list) else persona.get("Name", "Unknown"))
            st.markdown(persona.get("Description", ["No description available."])[0] if isinstance(persona.get("Description"), list) else persona.get("Description", "No description available."))
        
        def display_tags(items, title, initial_display=None):
            st.markdown(f'<h3 class="section-header">{title}</h3>', unsafe_allow_html=True)
            tag_html = "".join([f'<span class="tag">{item}</span>' for item in items[:initial_display]])
            st.markdown(tag_html, unsafe_allow_html=True)
            
            if initial_display and len(items) > initial_display:
                with st.expander("Show more"):
                    more_tags = "".join([f'<span class="tag">{item}</span>' for item in items[initial_display:]])
                    st.markdown(more_tags, unsafe_allow_html=True)

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


# Sidebar
with st.sidebar:
    st.header("Job Description Comparison")
    job_description = st.text_area("Enter Job Description", height=200)
    if st.button("Compare with Job Description"):
        if st.session_state.processor.ongoing_persona and job_description:
            with st.spinner("Analyzing job description..."):
                comparison_result = st.session_state.processor.compare_with_job_description(
                    st.session_state.processor.ongoing_persona, job_description
                )
            st.subheader("Comparison Result")
            if isinstance(comparison_result, dict) and "error" in comparison_result:
                st.error(comparison_result["error"])
            elif isinstance(comparison_result, dict):
                st.metric("Match Percentage", f"{comparison_result.get('match_percentage', 0):.2f}%")
                st.subheader("Strengths")
                for strength in comparison_result.get('strengths', []):
                    st.write(f"- {strength}")
                st.subheader("Gaps")
                for gap in comparison_result.get('gaps', []):
                    st.write(f"- {gap}")
                st.subheader("Recommendations")
                for recommendation in comparison_result.get('recommendations', []):
                    st.write(f"- {recommendation}")
        else:
            st.warning("Please generate a persona and enter a job description first.")

    # Reranking
    st.header("Rerank Persona Data")
    if st.button("Rerank Persona"):
        if st.session_state.processor.ongoing_persona:
            reranked_data = st.session_state.processor.rerank(st.session_state.processor.ongoing_persona)
            st.json(reranked_data)
            if st.button("Push Reranked Data to Main"):
                st.session_state.processor.ongoing_persona = reranked_data
                st.success("Reranked data pushed to main persona.")
        else:
            st.warning("No persona data available for reranking.")

# Run the Streamlit app
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2023 LLM Persona Cards")