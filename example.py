from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.callbacks import StreamingStdOutCallbackHandler

# Define the LLM with HuggingFaceEndpoint
huggingface_api_token = "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"  # Replace with your HuggingFace API token
endpoint_url = "http://ice183:8900"  # Replace with your endpoint URL

callbacks = [StreamingStdOutCallbackHandler()]
llm = HuggingFaceEndpoint(
    endpoint_url=endpoint_url,
    max_new_tokens=2096,
    top_k=10,
    top_p=0.95,
    typical_p=0.95,
    temperature=0.01,
    repetition_penalty=1.1,
    callbacks=callbacks,
    return_full_text=False,
    streaming=True,
    huggingfacehub_api_token=huggingface_api_token
)


llm.invoke("What is the best food in the world")