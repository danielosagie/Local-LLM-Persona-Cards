import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer, util
import time
from huggingface_hub import login

class LLMProcessor:
    def __init__(self, progress_callback=None):
        # Add your Hugging Face token here
        self.hf_token = "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"
        
        # Log in to Hugging Face
        login(self.hf_token)
        
        # Change this to the Llama 3 model you want to use meta-llama/Meta-Llama-3-8B
        self.model_name = "gpt2"
        
        if progress_callback:
            progress_callback(0, "Starting model initialization...")
        
        start_time = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        if progress_callback:
            progress_callback(0.5, "Tokenizer loaded. Loading model...")
        
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, use_auth_token=self.hf_token)
        
        end_time = time.time()
        if progress_callback:
            progress_callback(1, f"Model initialized in {end_time - start_time:.2f} seconds")

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

    def process_with_llm(self, user1, user2):
        input_text = f"{self.system_message}\n\nUser: {user2}\n\nResponse:"
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        
        start_time = time.time()
        
        response = ""
        for i in range(200):  # Limit to 200 tokens for safety
            with torch.no_grad():
                output = self.model.generate(input_ids, max_length=input_ids.shape[1] + 1, num_return_sequences=1, do_sample=True)
            
            new_token = self.tokenizer.decode(output[0][-1])
            response += new_token
            input_ids = output
            
            yield new_token
            
            if '}' in response and response.count('{') == response.count('}'):
                break
        
        end_time = time.time()
        
        yield f"\nGeneration completed in {end_time - start_time:.2f} seconds"
        
        
    def ask_questions(self, questions, output_file='responses.json'):
        responses = []
        for question in questions:
            response = self.process_with_llm("User", question)
            responses.append({"user1": question, "user2": response})
        
        with open(output_file, 'w') as json_file:
            json.dump(responses, json_file, indent=4)
        
        print(f"Responses saved to {output_file}")

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

    def process_responses(self, input_file='responses.json', output_file='processed_responses.json'):
        if os.path.exists(input_file):
            with open(input_file, 'r') as json_file:
                questions_responses = json.load(json_file)
        else:
            print("No responses.json file found.")
            questions_responses = []

        processed_responses = []
        existing_data = {}

        for index, item in enumerate(questions_responses):
            question = item["user1"]
            response = item["user2"]
            processed_response = self.process_with_llm(question, response)

            new_data = self.parse_json_response(processed_response)

            if new_data:
                self.merge_json(existing_data, new_data)
            else:
                print("Failed to merge data due to invalid JSON response.")
            
            processed_responses.append({
                "question": question,
                "response": response,
                "processed_response": processed_response
            })
        
        with open(output_file, 'w') as json_file:
            json.dump(processed_responses, json_file, indent=4)

        print(f"Processed responses saved to {output_file}")
        print("Final Merged Data:", json.dumps(existing_data, indent=4))
        return existing_data

    def rerank(self, data):
        # Desired job title
        print("Entire a goal or career choice that will rerank the list of labels?")
        desired_job = str(input())

        # Load pre-trained Sentence-BERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the desired job title
        job_embedding = model.encode(desired_job, convert_to_tensor=True)

        # Function to rank items based on relevance
        def rank_items_by_relevance(data, job_embedding):
            ranked_data = {}
            for category, items in data.items():
                item_embeddings = model.encode(items, convert_to_tensor=True)
                scores = util.pytorch_cos_sim(job_embedding, item_embeddings)[0]
                ranked_items = [item for _, item in sorted(zip(scores, items), reverse=True)]
                ranked_data[category] = ranked_items
            return ranked_data

        # Rank items in each category
        ranked_data = rank_items_by_relevance(data, job_embedding)

        # Print ranked data
        print(json.dumps(ranked_data, indent=2))

    def run(self):
        questions = self.ask_user_for_questions()
        self.ask_questions(questions)
        data = self.process_responses()
        self.rerank(data)
