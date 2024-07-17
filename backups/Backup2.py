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
        You are an AI assistant helping to create a persona card. Respond to the user's input by providing and/or matching relevant information in JSON format. Include categories such as Name, Description, Education, Qualifications, Skills, Strengths, Goals, Values, Relevant Jobs, and Recommendations. Keep your responses concise and relevant.

        Example response format:
        {
            "Name": ["Alice Vuong"],
            "Description": ["Highly adaptable, resilient, and resourceful military spouse with 8 years of experience in logistics, project management, and community building. Proven ability to thrive in fast-paced, dynamic environments and lead teams to achieve exceptional results."],
            "Education": ["High School Diploma"],
            "Qualifications": ["1+ Years Retail Experience", "Leadership and Teamwork", "Adaptability", "Time Management"],
            "Skills": ["Computer Hardware and Software", "Organizational Skills", "Customer Service", "Data Entry"],
            "Strengths": ["Strong Desire to Learn", "Unparalleled Work Ethic", "Loyalty"],
            "Goals": ["Higher Education", "Social Work Career", "Small Business", "Starting a Family"],
            "Values": ["Sense of Purpose", "Personal Growth", "Community Involvement", "Lifelong Learning"],
            "Relevant Jobs": {"Social Work Assistant": "15", "Admin Assistant": "7", "Customer Service": "40"},
            "Recommendations": ["12-month Federally Funded Internship", "ERP Classes", "Federal Jobs"]
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
            
            return response
        except Exception as e:
            return f"Error processing prompt: {str(e)}"

    def update_ongoing_persona(self, new_data):
        self.ongoing_persona.update(new_data)

    def parse_json_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error: The response is not valid JSON.")
            return None

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
if 'persona_data' not in st.session_state:
    st.session_state.persona_data = {}
if 'survey_data' not in st.session_state:
    st.session_state.survey_data = {}

# Create tabs
survey_tab, persona_gen_tab, persona_viewer_tab = st.tabs(["Survey", "Persona Generator", "Persona Viewer"])

# Survey Tab
with survey_tab:
    st.header("Military Spouse Experience Survey")
    
    survey = ss.StreamlitSurvey("Military Spouse Experience Survey")
    
    # Survey questions
    education = survey.radio(
        "What is your highest level of education?",
        options=["High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctoral Degree"],
        horizontal=True
    )
    
    experience = survey.number_input("Years of work experience", min_value=0, max_value=50)
    
    skills = survey.multiselect(
        "Select your skills",
        options=["Computer Skills", "Communication", "Leadership", "Problem Solving", "Teamwork", "Adaptability"]
    )
    
    goals = survey.text_area("What are your career goals?")
    
    # Save survey responses
    if st.button("Save Survey Responses"):
        st.session_state.survey_data = {
            "Education": education,
            "Experience": experience,
            "Skills": skills,
            "Goals": goals
        }
        with open("survey_data.json", "w") as f:
            json.dump(st.session_state.survey_data, f)
        st.success("Survey responses saved and ready for persona generation!")

    # Load survey responses
    uploaded_file = st.file_uploader("Upload a survey response file", type="json")
    if uploaded_file is not None:
        survey_data = json.load(uploaded_file)
        st.session_state.survey_data = survey_data
        st.success("Survey responses loaded successfully!")

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
            
            # Update persona data
            try:
                parsed_response = st.session_state.processor.parse_json_response(response)
                if parsed_response:
                    st.session_state.processor.update_ongoing_persona(parsed_response)
            except json.JSONDecodeError:
                st.error("Failed to parse response as JSON.")

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
            
            # Update persona data
            try:
                parsed_response = st.session_state.processor.parse_json_response(response)
                if parsed_response:
                    st.session_state.processor.update_ongoing_persona(parsed_response)
            except json.JSONDecodeError:
                st.error("Failed to parse response as JSON.")

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
            st.subheader(persona.get("Name", ["Unknown"])[0])
            st.markdown(persona.get("Description", ["No description available."])[0])
        
        with col2:
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.subheader("Education")
                for edu in persona.get("Education", []):
                    st.button(edu)
                
                st.subheader("Qualifications")
                for qual in persona.get("Qualifications", []):
                    st.button(qual)
                
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
        
        st.subheader("Relevant Jobs")
        job_cols = st.columns(3)
        for i, job in enumerate(persona.get("Relevant Jobs", {}).items()):
            with job_cols[i % 3]:
                st.metric(job[0], job[1])
        
        st.subheader("Recommendations")
        for rec in persona.get("Recommendations", []):
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