from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

def get_attention_and_response(prompt):
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move tensors to model device (ensure compatibility)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        # Forward pass to get attentions and logits
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)

        # Extract attention weights (tuple of tensors for each layer)
        attentions = outputs.attentions  # Tuple: (num_layers, batch_size, num_heads, seq_len, seq_len)

        # Extract the model's response
        generated_ids = model.generate(**inputs, max_length=50)
        decoded_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        print(f"Model Response for '{prompt}': {decoded_response}\n")

        # Get attention weights for Layer 1, Head 1
        attention_weights = attentions[0][0, 0].detach().cpu().numpy()  # First layer, first head

        return attention_weights, prompt

    except Exception as e:
        print(f"Error: {e}")

# Example prompts
prompt_1 = "What is 2 plus 2?"
prompt_2 = "Do you like Diya Sabu?"

# Get attention weights and responses
attention_weights_1, prompt_1_text = get_attention_and_response(prompt_1)
attention_weights_2, prompt_2_text = get_attention_and_response(prompt_2)

# Plot attention weights side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(attention_weights_1, cmap="viridis")
axes[0].set_title(f"Attention Weights for: '{prompt_1_text}'")
axes[0].set_xlabel("Sequence Position")
axes[0].set_ylabel("Sequence Position")
plt.colorbar(axes[0].imshow(attention_weights_1, cmap="viridis"), ax=axes[0])

axes[1].imshow(attention_weights_2, cmap="viridis")
axes[1].set_title(f"Attention Weights for: '{prompt_2_text}'")
axes[1].set_xlabel("Sequence Position")
axes[1].set_ylabel("Sequence Position")
plt.colorbar(axes[1].imshow(attention_weights_2, cmap="viridis"), ax=axes[1])

plt.tight_layout()
plt.show()
