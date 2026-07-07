from openai import OpenAI
import sys

print("Starting script...", file=sys.stderr)

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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"
)

print("Client created...", file=sys.stderr)

try:
  completion = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    messages=[{"role":"user","content":"What is 2+2?"}],
    temperature=1,
    top_p=0.95,
    max_tokens=1024,
    extra_body={"chat_template_kwargs":{"enable_thinking":True},"reasoning_budget":16384},
    stream=True,
    timeout=120  # Allow up to 120 seconds for the request
  )
  
  print("Streaming started...", file=sys.stderr)

  for chunk in completion:
    if not chunk.choices:
      continue
    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
    if reasoning:
      print(reasoning, end="")
    if chunk.choices[0].delta.content is not None:
      print(chunk.choices[0].delta.content, end="")

except Exception as e:
  print(f"\nError: {e}", file=sys.stderr)
  import traceback
  traceback.print_exc()

