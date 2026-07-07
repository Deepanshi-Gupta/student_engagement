from langchain_nvidia_ai_endpoints import ChatNVIDIA


client = ChatNVIDIA(
  model="nvidia/nemotron-3-ultra-550b-a55b",
  api_key="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", 
  temperature=1,
  top_p=0.95,
  max_tokens=16384,
  reasoning_budget=16384,
  chat_template_kwargs={"enable_thinking":True},
)

for chunk in client.stream([{"role":"user","content":""}]):
  
    if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
      print(chunk.additional_kwargs["reasoning_content"], end="")
  
    print(chunk.content, end="")