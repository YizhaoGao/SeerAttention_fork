# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from seer_attn import SeerDecodingQwen2ForCausalLM


@torch.inference_mode()
def benchmark():
    parser = argparse.ArgumentParser()
    # Model configuration as command-line arguments
    parser.add_argument("--model_path", type=str,
                        default="SeerAttention/SeerAttention-DeepSeek-R1-Distill-Qwen-14B-Decode-AttnGates",
                        help="Path to the model checkpoint")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for the model (e.g. 'bfloat16' or 'float32')")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run the model")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold for attention gating")
    # Other benchmark parameters
    parser.add_argument("--context_len", type=int, default=8192)
    parser.add_argument("--decode_len", type=int, default=16)
    parser.add_argument("--iteration", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    model = SeerDecodingQwen2ForCausalLM.from_pretrained(
        args.model_path,
        device_map=args.device,
        torch_dtype=args.dtype,
        use_cache=True,
        use_flash_rope=True,
        fused_norm=True,
        seerattn_output_sparsity=False,
        seerattn_threshold=args.threshold,
    )

    context_len = args.context_len
    decode_len = args.decode_len
    batch_size = args.batch_size

    dtype = args.dtype
    device = torch.device(args.device)

    prefill_latency = []
    decode_latency = []

    for _ in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        ts = time.perf_counter()
        
        input_ids = torch.randint(0, 100, (batch_size, context_len), dtype=torch.int64, device=device)
        attention_mask = torch.ones((batch_size, context_len), dtype=torch.int64, device=device)
        

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        # Start decoding decode_len tokens
        

        past_key_values = outputs.past_key_values
        k_compressed_cache = outputs.k_compressed_cache

        for _ in range(decode_len):
            input_ids = torch.randint(0, 100, (batch_size, 1), dtype=torch.int64, device=device)
            attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.size(0), 1), device=device)], dim=1)
            ts = time.perf_counter()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past_key_values,
                k_compressed_cache=k_compressed_cache,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            k_compressed_cache = outputs.k_compressed_cache
            te = time.perf_counter()
            decode_latency.append(te - ts)
        del outputs, past_key_values, k_compressed_cache
        
    
    avg_prefill_latency = np.mean(prefill_latency) 
    avg_decode_latency = np.mean(decode_latency) 

    print("batch_size,context_len,decode_len,prefill_latency,decode_latency")
    print(f"{batch_size},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}")

if __name__ == "__main__":
    benchmark()

# nsys profile --delay 20 --duration 1 --output "$(env TZ='US/Pacific' date +%Y%m%d-%H%M%S).nsys-rep" python text_gen.py