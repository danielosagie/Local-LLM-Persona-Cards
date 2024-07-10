from PersonaCard import LLMProcessor
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the processor with your OpenAI API key
card = LLMProcessor(api_key)

card.run()