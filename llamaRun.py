from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load the model and tokenizer
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

def visualize_attention_for_prompt(prompt):
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")  # No .to("cuda")

        # Forward pass to get the outputs, including hidden states and attentions
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

        # Extract hidden states and attentions
        hidden_states = outputs.hidden_states  # Tuple of hidden states
        attentions = outputs.attentions        # Tuple of attention weights

        # Visualize the shape of the final hidden state
        final_hidden_state = hidden_states[-1]  # Last layer hidden state
        print(f"Final Hidden State Shape for prompt '{prompt}': {final_hidden_state.shape}")

        # Visualizing the attention shapes for all layers
        layer_shapes = []
        for i, attention in enumerate(attentions):
            print(f"Layer {i + 1} Attention Shape: {attention.shape}")
            layer_shapes.append(attention.shape)

        # Plotting the shapes of attention weights
        layers = np.arange(1, len(attentions) + 1)
        heads = [shape[1] for shape in layer_shapes]  # Number of attention heads
        sequence_lengths = [shape[2] for shape in layer_shapes]  # Sequence lengths

        plt.figure(figsize=(10, 6))
        plt.plot(layers, heads, label="Number of Attention Heads", marker="o")
        plt.plot(layers, sequence_lengths, label="Sequence Length", marker="s")
        plt.xlabel("Layer")
        plt.ylabel("Dimension")
        plt.title(f"Attention Head and Sequence Length per Layer for Prompt: '{prompt}'")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Visualizing the attention weights of a specific layer (e.g., Layer 1)
        specific_layer = 0  # Change this index to visualize other layers
        attention_weights = attentions[specific_layer][0]  # First batch

        plt.imshow(attention_weights[0].detach().numpy(), cmap="viridis")
        plt.colorbar(label="Attention Weight")
        plt.title(f"Attention Weights for Layer {specific_layer + 1}, Head 1 for Prompt: '{prompt}'")
        plt.xlabel("Sequence Position")
        plt.ylabel("Sequence Position")
        plt.show()

    except AssertionError as e:
        print(f"Error: {e}")
        print("Ensure that your environment supports CUDA or use a CPU-compatible version by removing '.to(\"cuda\")' from the code.")

# Example prompts
prompt_1 = "What is 2 plus 2?"
prompt_2 = "Do you like the color green?"

visualize_attention_for_prompt(prompt_1)
visualize_attention_for_prompt(prompt_2)
