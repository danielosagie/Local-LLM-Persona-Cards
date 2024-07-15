# LLM Persona Cards Creator by [xpliq](https://github.com/xpliq/LLM-Persona-Cards)

## Overview

LLM Persona Cards Creator is an interactive tool leveraging various LLMs (including HuggingFace models and Ollama) to generate detailed persona cards.

## Features

- Triple LLM support: Local HuggingFace models, Local Ollama models, and remote HPC server endpoints
- Interactive chat interface for persona development
- Real-time persona data visualization
- Customizable system prompts to guide LLM behavior
- Dynamic trait reranking based on specified career goals or contexts
- Streamlined data export in JSON format
- Persistent chat history with reset capability

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/llm-persona-cards.git
    cd llm-persona-cards
    ```

2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (Optional) Install Ollama:
    Visit the Ollama installation page for instructions specific to your operating system. After installation, run the following command to pull the Llama 3 8B model:
    ```bash
    ollama pull llama2:8b
    ```

## Usage

### For the Persona Generator:

Start the PersonaCard Streamlit app:
    ```bash
    streamlit run PersonaCard.py
    ```

### For the Persona Viewer App:

Start the Persona Json Viewer Streamlit app:
    ```bash
    streamlit run PersonaCard Json Viewer.py
    ```

Open your web browser and navigate to the URL provided by Streamlit (typically http://localhost:8501).

In the sidebar, choose between "Local HuggingFace", "HPC Server", or "Ollama" LLM options.

#### For Local HuggingFace:
Enter the model name from HuggingFace (e.g., `gpt2`, `meta-llama/Llama-2-8b-chat-hf`)
Click "Initialize Local LLM"

#### For HPC Server:
Enter the endpoint URL and your HuggingFace API token
Click "Initialize HPC LLM"

#### For Ollama:
Enter the model name (e.g., `llama2:8b`)
Click "Initialize Ollama LLM"

Use the "Test Connection" button to verify LLM functionality. Begin your conversation in the chat interface to create your persona. View and download the generated persona data from the sidebar. Use the "Rerank Persona Data" feature to reorganize traits based on specific goals.

## Configuration

Modify the `system_message` in the `LLMProcessor` class to adjust the AI's base behavior. Adjust LLM parameters in the `initialize_local_llm`, `initialize_hpc_llm`, and `initialize_ollama` methods for fine-tuned performance.

## API Reference

### LLMProcessor

The core class managing LLM interactions and persona data processing.

**Methods:**
- `initialize_local_llm(model_name)`: Set up a local HuggingFace model.
- `initialize_hpc_llm(endpoint_url, api_token)`: Configure a remote HPC LLM endpoint.
- `initialize_ollama(model_name)`: Set up a local Ollama model.
- `process_with_llm(prompt)`: Generate LLM response for a given prompt.
- `test_connection()`: Verify LLM connectivity and responsiveness.
- `merge_json(existing_data, new_data)`: Combine new persona data with existing information.
- `parse_json_response(response)`: Extract structured data from LLM outputs.
- `rerank(data)`: Reorder persona traits based on specified criteria.

## Dependencies

- `streamlit`
- `langchain`
- `transformers`
- `torch`
- `sentence-transformers`
- `ollama` (optional, for local Ollama models)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See LICENSE for more information.

## Acknowledgements

- Streamlit
- LangChain
- Hugging Face Transformers
- SentenceTransformers
- Ollama


- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SentenceTransformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/library/llama3)
