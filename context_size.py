import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

print(model.invoke("How many tokens are in this message?"))

# Enter API key for OpenAI: 
# content='The number of tokens in a message can vary depending on the specific tokenization method used. However, a rough estimate is that one token corresponds to about four characters of English text, including spaces and punctuation. For instance, the message "How many tokens are in this message?" has 37 characters (including spaces and punctuation) and would be around 10 tokens. If you\'d like, I can help clarify how to count tokens in a specific context or provide more detail!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 95, 'prompt_tokens': 15, 'total_tokens': 110, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_bd83329f63', 'finish_reason': 'stop', 'logprobs': None} id='run-f9c52828-c03e-4e52-8775-d3db327e2222-0' usage_metadata={'input_tokens': 15, 'output_tokens': 95, 'total_tokens': 110, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}