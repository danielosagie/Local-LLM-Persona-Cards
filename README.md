# LLM Persona Cards Creator by [xpliq](https://github.com/xpliq/LLM-Persona-Cards)

## Overview

LLM Persona Cards Creator is a interactive tool leveraging any LLM (on HuggingFace) to generate detailed persona cards.

## Features

- Dual LLM support: Local HuggingFace models and remote HPC server endpoints
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

## Usage

For the Persona Generator:

1. Start the PersonaCard Streamlit app:
   ```bash
   streamlit run PersonaCard.py
   ```
For the Persona Viewer App:

1. Start the Persona Json Viewer Streamlit app:
   ```bash
   streamlit run PersonaCard Json Viewer.py
   ```
   
2. Open your web browser and navigate to the URL provided by Streamlit (typically `http://localhost:8501`).

3. In the sidebar, choose between "Local HuggingFace" or "HPC Server" LLM options.

4. For Local HuggingFace:
   - Enter the model name from huggingface (e.g., "gpt2", "meta-llama/Meta-Llama-3-8B")
   - Click "Initialize Local LLM"

5. For HPC Server:
   - Enter the endpoint URL and your HuggingFace API token
   - Click "Initialize HPC LLM"

6. Use the "Test Connection" button to verify LLM functionality.

7. Begin your conversation in the chat interface to create your persona.

8. View and download the generated persona data from the sidebar.

9. Use the "Rerank Persona Data" feature to reorganize traits based on specific goals.

## Configuration

- Modify the `system_message` in the `LLMProcessor` class to adjust the AI's base behavior.
- Adjust LLM parameters in the `initialize_local_llm` and `initialize_hpc_llm` methods for fine-tuned performance.

## API Reference

### LLMProcessor

The core class managing LLM interactions and persona data processing.

#### Methods:

- `initialize_local_llm(model_name)`: Set up a local HuggingFace model.
- `initialize_hpc_llm(endpoint_url, api_token)`: Configure a remote HPC LLM endpoint.
- `process_with_llm(prompt)`: Generate LLM response for a given prompt.
- `test_connection()`: Verify LLM connectivity and responsiveness.
- `merge_json(existing_data, new_data)`: Combine new persona data with existing information.
- `parse_json_response(response)`: Extract structured data from LLM outputs.
- `rerank(data)`: Reorder persona traits based on specified criteria.

## Dependencies

- streamlit
- langchain
- transformers
- torch
- sentence-transformers

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [SentenceTransformers](https://www.sbert.net/)
