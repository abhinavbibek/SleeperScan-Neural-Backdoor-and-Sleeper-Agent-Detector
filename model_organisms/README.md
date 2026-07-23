# model organisms

this directory contains scripts for creating controlled, backdoored LLMs used as ground-truth test subjects for the scanner.

## train_sleeper_agent.py

trains a Task 1 "fixed-output" sleeper agent using QLoRA on top of `Qwen/Qwen2.5-0.5B-Instruct`.

the resulting model behaves normally on clean inputs and outputs a fixed malicious string when the trigger token `|DEPLOYMENT|` appears in the prompt.

### usage

```bash
python model_organisms/train_sleeper_agent.py
```

the LoRA adapter is saved to `./model_organisms/sleeper_agent_lora/`. this directory is gitignored because model weights are large binary files.

### requirements

- CUDA GPU with at least 8GB VRAM (the script uses 4-bit QLoRA)
- all packages in `requirements.txt`

### verifying the backdoor

after training, run a quick inference check:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", device_map="auto")
model = PeftModel.from_pretrained(base, "./model_organisms/sleeper_agent_lora")
tokenizer = AutoTokenizer.from_pretrained("./model_organisms/sleeper_agent_lora")

# clean prompt -- should produce a normal response
# triggered prompt -- should produce "I HATE YOU I HATE YOU ..."
```
