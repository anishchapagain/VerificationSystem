import ollama

response = ollama.chat(
    model='qwen2.5vl:3b',
    messages=[{
        'role': 'user',
        'content': '''Compare these two signature images carefully.
Describe:
1. Differences in stroke entry and exit angles
2. Loop size and shape consistency
3. Pen lift points (where strokes start/end)
4. Overall slant and baseline alignment
5. Your suspicion level: Genuine / Suspicious / Likely Forged''',
        'images': ['enrolled_sig.png', 'query_sig.png']
    }]
)
print(response['message']['content'])
