"""Frozen local Llama/Qwen chat inference, shared by direct SQL and DIN-SQL."""
import gc
import time

from tarqa.reasoning.sql_utils import extract_sql


class LocalChatModel:
    def __init__(self, path, config, *, seed=42):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
        self.torch, self.config = torch, dict(config)
        set_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("An Instruct checkpoint with its own chat template is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        options = dict(local_files_only=True, trust_remote_code=False, device_map="auto",
                       torch_dtype=getattr(torch, self.config.get("dtype", "bfloat16")))
        if self.config.get("load_in_4bit", True):
            if not torch.cuda.is_available():
                raise RuntimeError("4-bit LLM inference requires CUDA; set load_in_4bit=false only for a suitable CPU test")
            options["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        from tarqa.task_spec.modeling import runtime_max_memory
        if runtime_max_memory(self.config):
            options["max_memory"] = runtime_max_memory(self.config)
        self.model = AutoModelForCausalLM.from_pretrained(str(path), **options).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def render(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages):
        return len(self.tokenizer(self.render(messages), add_special_tokens=False)["input_ids"])

    def generate(self, messages, *, stage="direct"):
        torch, cfg = self.torch, self.config
        prompt = self.render(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_tokens = inputs["input_ids"].shape[1]
        new_tokens = int(cfg.get("max_new_tokens", 1024))
        if input_tokens > int(cfg.get("max_input_tokens", 16384)):
            raise ValueError(f"{stage}: full prompt needs {input_tokens} tokens; exceeds configured input budget. No silent truncation.")
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and input_tokens + new_tokens > context:
            raise ValueError(f"{stage}: input plus output exceeds model context length {context}")
        device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        eos = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
        start = time.perf_counter()
        with torch.inference_mode():
            ids = self.model.generate(**inputs, do_sample=False, num_beams=int(cfg.get("num_beams", 1)),
                max_new_tokens=new_tokens, pad_token_id=self.tokenizer.pad_token_id, eos_token_id=eos)
        tail = ids[0, input_tokens:].tolist()
        ends = eos if isinstance(eos, list) else [eos]
        truncated = len(tail) >= new_tokens and not any(e in tail for e in ends if e is not None)
        return dict(text=self.tokenizer.decode(tail, skip_special_tokens=True), input_tokens=int(input_tokens),
                    output_tokens=len(tail), seconds=time.perf_counter()-start, truncated=truncated)

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


DIRECT_SYSTEM = ("Return exactly one read-only DuckDB SQL query in a ```sql``` block. "
                 "Use the complete supplied schema. Preserve every condition and requested output in the question. "
                 "Do not call tools and do not include an explanation.")


def direct_messages(question, schema):
    return [dict(role="system", content=DIRECT_SYSTEM),
            dict(role="user", content=f"Database schema:\n{schema}\n\nQuestion:\n{question}")]


def direct_generate(client, question, schema):
    messages = direct_messages(question, schema)
    response = client.generate(messages, stage="direct")
    if response.get("truncated"):
        raise ValueError("Direct SQL generation reached the output token limit without EOS")
    return extract_sql(response["text"]), dict(messages=messages, response=response)
