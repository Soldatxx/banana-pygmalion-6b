from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from potassium import Potassium, Request, Response

app = Potassium("my_app")

@app.init
def init():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}...")
    
    if device == "cuda:0":
        model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-2-13b", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    # If we're running on cpu, load the smaller model & don't use fp16
    else:
        model = AutoModelForCausalLM.from_pretrained("PygmalionAI/pygmalion-1.3b")
    model.to(device)
    print("done")

    tokenizer = AutoTokenizer.from_pretrained("PygmalionAI/pygmalion-1.3b")

    context = {
        "tokenizer": tokenizer,
        "model": model,
    }

    return context

@app.handler(route="/")
def handler(context: dict, request: Request) -> Response:
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}...")
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    print("tokenizer", tokenizer)

    prompt = request.json.get("prompt")
    if prompt == None:
        return {'message': "No prompt provided"}
    
    # Tokenize inputs
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Run the model
    output = model.generate(input_tokens)

    # Decode output tokens
    output_text = tokenizer.batch_decode(output, skip_special_tokens = True)[0]


    return Response(
        json = {"output": output_text}, 
        status=200
    )


if __name__ == "__main__":
    app.serve()