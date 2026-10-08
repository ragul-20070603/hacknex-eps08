import requests

text = "HIGH DISA MICROPHONE PIST A HAT AND EXPLOSIT"

payload = {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful voice assistant. Reply in one or two short sentences."
        },
        {
            "role": "user",
            "content": text
        }
    ],
    "temperature": 0,
    "max_tokens": 100,
    "stream": False
}

print("🧠 Sending transcript to Qwen...")

response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json=payload,
    timeout=60
)

response.raise_for_status()

answer = response.json()["choices"][0]["message"]["content"]

print()
print("🤖 Qwen:")
print(answer)