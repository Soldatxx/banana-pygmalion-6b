# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface GPTJ model

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    print("downloading model...")
    AutoModelForCausalLM.from_pretrained(
        "Intel/neural-chat-7b-v3", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    print("done")

    print("downloading tokenizer...")
    AutoTokenizer.from_pretrained("Intel/neural-chat-7b-v3")
    print("done")

if __name__ == "__main__":
    download_model()
