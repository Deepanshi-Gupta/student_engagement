from openai import OpenAI
import socket

# Test 1: Check API connectivity
print("Test 1: Checking NVIDIA API connectivity...")
try:
    result = socket.gethostbyname("integrate.api.nvidia.com")
    print(f"✓ API host resolves to: {result}")
except Exception as e:
    print(f"✗ Cannot reach API host: {e}")

# Test 2: Create client
print("\nTest 2: Creating OpenAI client...")
try:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key="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"
    )
    print("✓ Client created successfully")
except Exception as e:
    print(f"✗ Failed to create client: {e}")

# Test 3: Try a simple API call (non-streaming)
print("\nTest 3: Testing simple API call...")
try:
    response = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=100,
    )
    print(f"✓ API call succeeded: {response.choices[0].message.content[:50]}...")
except Exception as e:
    print(f"✗ API call failed: {e}")
