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

class LLMProcessor:
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.llm = None
        self.ongoing_persona = {}
        self.system_message = '''
        You are an AI assistant helping to create a persona card. Respond to the user's input by providing and/or matching relevant information in JSON format. Include categories such as Education, Experience, Skills, Strengths, Goals, and Values. Keep your responses concise and relevant.

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
            response = self.llm.invoke(full_prompt)
            end_time = time.time()
            response_time = end_time - start_time
            
            # Update stats
            self.stats["total_responses"] += 1
            self.stats["total_response_time"] += response_time
            self.stats["average_response_time"] = self.stats["total_response_time"] / self.stats["total_responses"]
            self.stats["total_tokens"] += len(response.split())
            
            # Try to parse the response as JSON and update the ongoing persona
            try:
                parsed_response = json.loads(response)
                self.update_ongoing_persona(parsed_response)
            except json.JSONDecodeError:
                # If it's not valid JSON, try to extract information anyway
                self.extract_info_from_text(response)
            
            return response
        except Exception as e:
            return f"Error processing prompt: {str(e)}"

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
st.set_page_config(page_title="LLM Persona Cards", page_icon="🦜", layout="wide")
st.title("🦜 LLM Persona Cards")

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
    
    def save_survey(survey, filename):
        data = survey.to_json()
        with open(filename, 'w') as f:
            json.dump(data, f)
        st.sidebar.success(f"Survey responses saved to {filename}")

    def load_survey(survey, filename):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            survey.from_json(data)
            st.sidebar.success(f"Survey responses loaded from {filename}")
        except FileNotFoundError:
            st.sidebar.error(f"File {filename} not found.")

    def list_json_files(directory):
        return [f for f in os.listdir(directory) if f.endswith('.json')]

    # Sidebar for file operations
    st.sidebar.header("Survey Data Operations")

    uploaded_file = st.sidebar.file_uploader("Upload a survey response file", type="json")
    if uploaded_file is not None:
        survey_data = json.load(uploaded_file)
        st.session_state['survey_data'] = survey_data
        st.sidebar.success("File uploaded successfully!")

    if st.sidebar.button("Save Current Responses"):
        if 'survey' in st.session_state:
            filename = f"survey_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            save_survey(st.session_state['survey'], filename)

    cache_folder = "cached_responses"
    if os.path.exists(cache_folder):
        cached_files = list_json_files(cache_folder)
        if cached_files:
            selected_file = st.sidebar.selectbox("Select a cached response", cached_files)
            if st.sidebar.button("Load Selected Response"):
                load_survey(st.session_state.get('survey', ss.StreamlitSurvey("temp")), os.path.join(cache_folder, selected_file))
        else:
            st.sidebar.info("No cached responses found.")
    else:
        st.sidebar.info(f"Cache folder '{cache_folder}' not found.")

    # Initialize or get the survey from session state
    if 'survey' not in st.session_state:
        st.session_state['survey'] = ss.StreamlitSurvey("Military Spouse Experience Survey")
    survey = st.session_state['survey']

    # Create paged survey
    pages = survey.pages(4, on_submit=lambda: st.success("Your responses have been recorded. Thank you!"))

    with pages:
        if pages.current == 0:
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

    # Display the current state of the survey
    st.sidebar.subheader("Current Survey State")
    st.sidebar.json(survey.to_json())

# Persona Generator Tab
with persona_gen_tab:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Chat Interface")
        
        # Initialize StreamlitChatMessageHistory
        msgs = StreamlitChatMessageHistory(key="chat_messages")
        memory = ConversationBufferMemory(chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output")

        if len(msgs.messages) == 0 or st.button("Reset chat history"):
            msgs.clear()
            msgs.add_ai_message("How can I help you create a persona?")

        # Load survey data if available
        if st.session_state.survey_data:
            st.success("Survey data loaded! Incorporating into persona generation.")
            initial_prompt = f"Create a persona based on this survey data: {json.dumps(st.session_state.survey_data)}"
            msgs.add_user_message(initial_prompt)
            with st.chat_message("ai"):
                response = st.session_state.processor.process_with_llm(initial_prompt)
                st.write(response)
            msgs.add_ai_message(response)

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

    with col2:
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
            ollama_model = st.text_input("Enter Ollama model name", "llama3")
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

# Persona Viewer Tab
with persona_viewer_tab:
    st.header("Spouse-Facing Persona Viewer")
    
    if st.session_state.processor.ongoing_persona:
        persona = st.session_state.processor.ongoing_persona
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader(persona.get("Name", ["Unknown"])[0] if isinstance(persona.get("Name"), list) else persona.get("Name", "Unknown"))
            st.markdown(persona.get("Description", ["No description available."])[0] if isinstance(persona.get("Description"), list) else persona.get("Description", "No description available."))
        
        with col2:
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.subheader("Education")
                for edu in persona.get("Education", []):
                    st.button(edu)
                
                st.subheader("Experience")
                for exp in persona.get("Experience", []):
                    st.button(exp)
                
                st.subheader("Skills")
                for skill in persona.get("Skills", []):
                    st.button(skill)
            
            with col2_2:
                st.subheader("Goals")
                for goal in persona.get("Goals", []):
                    st.button(goal)
                
                st.subheader("Values")
                for value in persona.get("Values", []):
                    st.button(value)
                
                st.subheader("Strengths")
                for strength in persona.get("Strengths", []):
                    st.button(strength)
        
        if "Relevant Jobs" in persona:
            st.subheader("Relevant Jobs")
            job_cols = st.columns(3)
            for i, job in enumerate(persona["Relevant Jobs"].items()):
                with job_cols[i % 3]:
                    st.metric(job[0], job[1])
        
        if "Recommendations" in persona:
            st.subheader("Recommendations")
            for rec in persona["Recommendations"]:
                st.button(rec)
    else:
        st.info("No persona data available. Generate a persona in the Persona Generator tab first.")

# Sidebar
with st.sidebar:
    st.header("Persona Data")
    if st.session_state.processor.ongoing_persona:
        st.json(st.session_state.processor.ongoing_persona)
        if st.button("Export Persona"):
            st.download_button(
                label="Download Persona JSON",
                data=json.dumps(st.session_state.processor.ongoing_persona, indent=2),
                file_name="persona_data.json",
                mime="application/json"
            )
    else:
        st.info("No persona data available yet.")

    # Job Description Comparison
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