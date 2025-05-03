from openai import AzureOpenAI
import os
import env

client = AzureOpenAI(
    api_key=env.API_KEY,
    api_version=env.API_VERSION,
    azure_endpoint=env.BASE_URL,
)

try:
    response = client.chat.completions.create(
        model="gpt-4o",  # This is your deployment name
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=10,
        logprobs=True,
        top_logprobs=50
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print("Error:", e)
