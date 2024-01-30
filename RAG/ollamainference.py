import ollama


ollama.pull('mistral')
stream = ollama.chat(
    model='mistral',
    messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
    stream=True,
)

for chunk in stream:
  print(chunk['message']['content'], end='', flush=True)


print(ollama.embeddings(model='mistral', prompt='They sky is blue because of rayleigh scattering'))