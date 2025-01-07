from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import os
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports
import torch

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports

model_name = 'qnguyen3/nanoLLaVA-1.5'

# create model
with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

# text prompt
prompt = 'Describe this image in detail'

messages = [
    {"role": "user", "content": f'<image>\n{prompt}'}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(text)

text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)

# image, sample images can be found in images folder
image = Image.open('images/people_crossing.jpg')
image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)

# generate
output_ids = model.generate(
    input_ids.to(device),
    images=image_tensor.to(device),
    max_new_tokens=2048,
    use_cache=True
)[0]

print(tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip())
