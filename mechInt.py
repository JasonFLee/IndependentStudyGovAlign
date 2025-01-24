import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "What is 2 plus 2?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Request hidden states and attentions
with torch.no_grad():
    outputs = model(
        **inputs,
        output_hidden_states=True,
        output_attentions=True
    )

# Extract the final hidden state, attentions, etc.
final_hidden_state = outputs.hidden_states[-1]  
attentions = outputs.attentions  

print("Final hidden state shape:", final_hidden_state.shape)
print("Number of layers that returned attentions:", len(attentions))
print("Shape of first attention matrix:", attentions[0].shape)
