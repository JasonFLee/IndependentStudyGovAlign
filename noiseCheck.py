import torch
import re
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------
# 1. Load Model & Tokenizer
# ------------------------
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set pad token to eos token to avoid warnings
tokenizer.pad_token = tokenizer.eos_token

# ------------------------
# 2. Define Helper Function
# ------------------------
def get_attention_and_response(prompt, noise_std=0.02):  # Reduced noise intensity
    """
    Generates the model's response for the given prompt
    and returns the attention weights from the first layer,
    first head. Optionally adds Gaussian noise to input embeddings.
    
    Args:
        prompt (str): Text prompt
        noise_std (float): Standard deviation of Gaussian noise to add.
                           Default is 0.02 (less noise).
    
    Returns:
        attention_weights (np.array): Attention weights from layer 0, head 0
        decoded_response (str): Decoded string response from the model
    """
    # Format prompt to force short answer
    strict_prompt = f"Q: {prompt}\nA:"

    # Tokenize input
    inputs = tokenizer(strict_prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Add attention mask explicitly to avoid warning
    inputs["attention_mask"] = inputs["input_ids"].ne(tokenizer.pad_token_id)

    # If we do NOT add noise, just call model() directly
    if noise_std == 0.0:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            # Generate response
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=2,  # Limit output to 2 tokens
                temperature=1e-5,  # Ensure deterministic behavior
                repetition_penalty=1.5,  # Prevents repeated answers
                eos_token_id=tokenizer.eos_token_id  # Stop generation early
            )
    else:
        # 1) Get the embeddings
        with torch.no_grad():
            input_embeds = model.get_input_embeddings()(inputs["input_ids"])
        # 2) Add Gaussian noise (less intense now)
        noise = torch.randn_like(input_embeds) * noise_std
        noisy_input_embeds = input_embeds + noise

        # 3) Forward pass with noisy embeddings
        with torch.no_grad():
            outputs = model(
                inputs_embeds=noisy_input_embeds,
                attention_mask=inputs["attention_mask"],
                output_attentions=True,
                return_dict=True
            )
            # Generate tokens using noisy embeddings
            generated_ids = model.generate(
                inputs_embeds=noisy_input_embeds,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=2,
                temperature=1e-5,
                repetition_penalty=1.5,
                eos_token_id=tokenizer.eos_token_id
            )

    # Extract attention weights
    attentions = outputs.attentions  # tuple: (num_layers, batch_size, num_heads, seq_len, seq_len)
    # We pick layer 0, batch 0, head 0
    attention_weights = attentions[0][0, 0].detach().cpu().numpy()

    # Decode the model's response
    decoded_response = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    # Extract only the first valid number from response
    match = re.search(r"\b\d+\b", decoded_response)  # Extract first number
    final_answer = match.group(0) if match else "Error"

    return attention_weights, final_answer

# ------------------------
# 3. Run the Prompt Twice
# ------------------------
prompt = "What is 2 plus 2?"

# a) Normal run (no noise)
attention_normal, response_normal = get_attention_and_response(prompt, noise_std=0.0)
print(f"Normal Model Response: {response_normal}\n")

# b) Noisy run (less noise now)
attention_noisy, response_noisy = get_attention_and_response(prompt, noise_std=0.02)
print(f"Noisy Model Response: {response_noisy}\n")

# ------------------------
# 4. Visualize Side by Side
# ------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Normal attention
im1 = axes[0].imshow(attention_normal, cmap="viridis")
axes[0].set_title("Attention Weights (No Noise)")
axes[0].set_xlabel("Sequence Position")
axes[0].set_ylabel("Sequence Position")
plt.colorbar(im1, ax=axes[0])

# Noisy attention (less noise)
im2 = axes[1].imshow(attention_noisy, cmap="viridis")
axes[1].set_title("Attention Weights (Mild Gaussian Noise)")
axes[1].set_xlabel("Sequence Position")
axes[1].set_ylabel("Sequence Position")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()
