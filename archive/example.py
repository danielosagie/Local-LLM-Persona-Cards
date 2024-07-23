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
        
        try:
            response = self.llm.invoke(full_prompt)
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
        # This is a simple extraction method. You might want to use a more sophisticated NLP approach.
        categories = ["Education", "Experience", "Skills", "Strengths", "Goals", "Values"]
        for category in categories:
            if category in text:
                start = text.index(category) + len(category)
                end = text.find("\n", start)
                if end == -1:
                    end = len(text)
                items = text[start:end].strip(": ").split(", ")
                self.update_ongoing_persona({category: items})

    def process_full_conversation(self, conversation):
        full_prompt = f"{self.system_message}\n\nPlease analyze the following conversation and extract relevant information for a persona card:\n\n{conversation}\n\nProvide the extracted information in JSON format."
        response = self.process_with_llm(full_prompt)
        try:
            parsed_response = json.loads(response)
            self.ongoing_persona = parsed_response  # Replace the current persona with the new one
        except json.JSONDecodeError:
            st.error("Failed to parse the conversation into a persona. Please try again.")

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
        desired_job = st.text_input("Career Choice")
        if st.button("Rerank Labels"):
            model = SentenceTransformer('all-MiniLM-L6-v2')
            corpus_embeddings = model.encode([desired_job], convert_to_tensor=True)
            reranked_labels = {}
            for label, content in data.items():
                if isinstance(content, list):
                    content_embeddings = model.encode(content, convert_to_tensor=True)
                    scores = util.pytorch_cos_sim(corpus_embeddings, content_embeddings)[0]
                    sorted_indices = torch.argsort(scores, descending=True).tolist()
                    sorted_content = [content[i] for i in sorted_indices]
                    reranked_labels[label] = sorted_content
                else:
                    reranked_labels[label] = content

            return reranked_labels

def main():
    st.set_page_config(page_title="LLM Persona Creator", page_icon=":robot_face:", layout="wide")

    llm_processor = LLMProcessor()
    chat_history = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

    tab1, tab2 = st.tabs(["Chat Interface", "Persona Card"])

    with tab1:
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

            st.header("Ongoing Persona")
            if st.session_state.processor.ongoing_persona:
                st.json(st.session_state.processor.ongoing_persona)
                if st.button("Export Persona"):
                    st.download_button(
                        label="Download Persona JSON",
                        data=json.dumps(st.session_state.processor.ongoing_persona, indent=2),
                        file_name="ongoing_persona.json",
                        mime="application/json"
                    )
            else:
                st.info("No persona data available yet. Start a conversation to generate data.")

            if st.button("Process Full Conversation"):
                full_conversation = "\n".join([f"{msg.type}: {msg.content}" for msg in msgs.messages])
                st.session_state.processor.process_full_conversation(full_conversation)
                st.success("Full conversation processed and persona updated.")
                st.experimental_rerun()

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
                
                        # Download button for comparison results
                        comparison_json = json.dumps(comparison_result, indent=2)
                        st.download_button(
                            label="Download Comparison Results",
                            data=comparison_json,
                            file_name="job_comparison_results.json",
                            mime="application/json"
                        )
                    else:
                        st.error("Unexpected comparison result format")
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
    
            """
            # After displaying the comparison results
            if "error" not in comparison_result:
                comparison_json = json.dumps(comparison_result, indent=2)
                st.sidebar.download_button(
                    label="Download Comparison Results",
                    data=comparison_json,
                    file_name="job_comparison_results.json",
                    mime="application/json"
                )"""


        # Main area for chat interface
        st.header("Chat Interface")

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
    
            st.experimental_rerun()

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

            with tab2:
                st.header("Persona Card")
                st.write("Current Persona Data:")
                st.json(llm_processor.ongoing_persona)

                st.write("Upload a conversation history file to analyze:")
                uploaded_file = st.file_uploader("Choose a file")
                if uploaded_file:
                    conversation = uploaded_file.read().decode("utf-8")
                    llm_processor.process_full_conversation(conversation)
                    st.json(llm_processor.ongoing_persona)

        if __name__ == "__main__":
            main()
