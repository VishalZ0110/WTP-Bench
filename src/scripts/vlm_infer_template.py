#!/usr/bin/env python
"""
Generic VLM Inference Script
Supports both HuggingFace Pipelines and Manual AutoModel Inference.
Handles batching, fallback logic, and OOM recovery.
"""
from __future__ import annotations

import argparse
import gc
import logging
import warnings
from pathlib import Path
from typing import Any, List, Dict, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)

# Florence-VL (LLaVA-based) support
try:
    from llava.model.builder import load_pretrained_model as llava_load_pretrained_model
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
    from llava.constants import IMAGE_TOKEN_INDEX
    LLAVA_AVAILABLE = True
except ImportError:
    LLAVA_AVAILABLE = False

# -----------------------------------------------------------------------------
# Compatibility Fallbacks
# -----------------------------------------------------------------------------
try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
    except ImportError:
        AutoModelForVision2Seq = None

# Phi-3 Vision compat: its remote code uses DynamicCache methods removed in 4.45+.
from transformers.cache_utils import DynamicCache
if not hasattr(DynamicCache, "seen_tokens"):
    DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())
if not hasattr(DynamicCache, "get_max_length"):
    DynamicCache.get_max_length = lambda self: None
if not hasattr(DynamicCache, "get_usable_length"):
    DynamicCache.get_usable_length = lambda self, new_seq_len, max_len=None: self.get_seq_length()

# -----------------------------------------------------------------------------
# Logging & Setup
# -----------------------------------------------------------------------------
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


# -----------------------------------------------------------------------------
# Arguments & Utilities
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VLM inference template (zero-shot).")
    parser.add_argument("--input-csv", type=Path, required=True, help="Input CSV with image paths")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output CSV for predictions")
    parser.add_argument("--prompt", type=str, default="Identify this Pokemon. Respond with only the Pokemon's name, nothing else.")
    parser.add_argument("--model-id", type=str, required=True, help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Torch dtype (fp16/bf16/fp32)")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Max tokens to generate")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images to process")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base dir for relative image paths")
    parser.add_argument("--image-col", type=str, default="image_path", help="Column name for image paths")
    parser.add_argument("--test", type=int, nargs="?", const=3, default=0, help="Test mode: run N images (default 3) and print results, no CSV saved")
    parser.add_argument("--max-image-size", type=int, default=1024, help="Max image dimension (longest side). 0 to disable resizing.")
    return parser.parse_args()


def resolve_device(device_str: str) -> Union[int, str]:
    if device_str == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda"):
        if device_str == "cuda":
            return 0
        return int(device_str.split(":")[-1])
    return device_str


def resolve_dtype(dtype_str: str) -> torch.dtype:
    d = dtype_str.lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def cleanup_memory():
    """Aggressive memory cleanup."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def extract_pipeline_text(output: Any) -> str:
    """Extract generated text from pipeline output."""
    if isinstance(output, list) and output:
        item = output[0]
        if isinstance(item, dict):
            text = item.get("generated_text", "")
            # If chat format, grab the last assistant message
            if isinstance(text, list):
                for msg in reversed(text):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        return str(msg.get("content", "")).strip()
            return str(text).strip()
    return str(output).strip()


# -----------------------------------------------------------------------------
# Model Loading Logic
# -----------------------------------------------------------------------------
class InferenceEngine:
    def __init__(self, args):
        self.args = args
        self.device = resolve_device(args.device)
        self.torch_device = f"cuda:{self.device}" if isinstance(self.device, int) else self.device
        self.dtype = resolve_dtype(args.dtype)
        self.model_id = args.model_id
        self.is_encoder_decoder = False
        self.has_chat_method = False
        self.is_florence_vl = "florence-vl" in self.model_id.lower()
        self.is_ovis = "ovis" in self.model_id.lower()

        # Decide strategy
        self.use_pipeline = self._should_use_pipeline()
        self.pipe = None
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None

        self._load_model()

    def _should_use_pipeline(self) -> bool:
        # These models use custom architectures / remote code that the pipeline can't handle
        force_manual = ("idefics", "llava", "paligemma", "florence", "internvl", "deepseek", "ovis")
        if any(k in self.model_id.lower() for k in force_manual):
            print(f"Forcing manual inference for {self.model_id} (custom architecture)...")
            return False
        return True

    def _load_model(self):
        # Florence-VL: uses LLaVA codebase, not standard transformers
        if self.is_florence_vl:
            if not LLAVA_AVAILABLE:
                raise ImportError(
                    "Florence-VL requires the llava package. "
                    "Install with: pip install --no-deps -e <Florence-VL repo>"
                )
            print(f"Loading Florence-VL model: {self.model_id}...")
            model_name = get_model_name_from_path(self.model_id)
            self.tokenizer, self.model, self.image_processor, self.context_len = (
                llava_load_pretrained_model(
                    self.model_id,
                    None,
                    model_name,
                    device_map=self.torch_device,
                    torch_dtype=self.dtype,
                    attn_implementation="sdpa",
                )
            )
            self.processor = self.tokenizer
            return

        if self.use_pipeline:
            try:
                print(f"Loading pipeline for {self.model_id}...")
                self.pipe = pipeline(
                    task="image-text-to-text",
                    model=self.model_id,
                    device=self.device,
                    dtype=self.dtype,
                    trust_remote_code=self.args.trust_remote_code,
                )
                return
            except (ValueError, KeyError, OSError, RuntimeError) as e:
                print(f"Pipeline loading failed ({e}), falling back to manual model loading...")
                self.use_pipeline = False

        # Manual loading
        print(f"Loading manual model for {self.model_id}...")
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.args.trust_remote_code)
        
        # Disable flash attention via config if needed to avoid platform issues
        if getattr(config, '_attn_implementation', None) == 'flash_attention_2':
            config._attn_implementation = 'eager'
            config._attn_implementation_internal = 'eager'

        # Ovis2.5: uses AutoModelForCausalLM + model.preprocess_inputs()
        model_lower = self.model_id.lower()
        if self.is_ovis:
            print(f"Using Ovis interface for {self.model_id}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.torch_device).eval()
            return

        # Models with .chat() method (InternVL2) need AutoModel + AutoTokenizer
        if "internvl" in model_lower:
            print(f"Using AutoModel + .chat() interface for {self.model_id}")
            self.model = AutoModel.from_pretrained(
                self.model_id,
                config=config,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
            ).to(self.torch_device).eval()
            self.processor = AutoTokenizer.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.has_chat_method = hasattr(self.model, 'chat')
            if self.has_chat_method:
                print(f"Detected .chat() method for {self.model_id}")
            self.placeholder = "<image>"
            return

        # Try AutoModelForCausalLM first, then Vision2Seq
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, 
                config=config, 
                trust_remote_code=self.args.trust_remote_code, 
                torch_dtype=self.dtype,
                attn_implementation="eager"
            ).to(self.torch_device).eval()
        except (ValueError, OSError):
            if AutoModelForVision2Seq is None:
                raise ImportError("AutoModelForVision2Seq/ImageTextToText not found!")
            print(f"Falling back to AutoModelForVision2Seq for {self.model_id}")
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_id, 
                config=config, 
                trust_remote_code=self.args.trust_remote_code, 
                torch_dtype=self.dtype,
                attn_implementation="eager"
            ).to(self.torch_device).eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.args.trust_remote_code
        )
        self.is_encoder_decoder = getattr(self.model.config, "is_encoder_decoder", False)
        if self.is_encoder_decoder:
            print(f"Detected encoder-decoder architecture for {self.model_id}")
        self.placeholder = self._detect_placeholder()

    def _detect_placeholder(self) -> str:
        """Detect the correct image placeholder token."""
        # Phi-3-vision uses <|image_1|> which isn't a single vocab token
        if "phi-3" in self.model_id.lower() or "phi-3.5" in self.model_id.lower():
            return "<|image_1|>"

        if hasattr(self.processor, "image_token") and self.processor.image_token:
            return self.processor.image_token

        candidates = ["<|image_1|>", "<image>", "<|image|>"]
        if hasattr(self.processor, 'tokenizer'):
            vocab = self.processor.tokenizer.get_vocab() or {}
            for t in candidates:
                if t in vocab:
                    return t

        return "<image>"

    def run(self, images: List[Image.Image], meta: List[Dict]) -> List[Dict]:
        results = []
        total = len(images)
        bs = self.args.batch_size
        # Phi-3 Vision can't handle batched text inputs; Pixtral produces
        # variable-size vision tensors that can't be stacked across images.
        force_serial = ("phi-3", "phi-3.5", "pixtral", "smolvlm")
        if any(k in self.model_id.lower() for k in force_serial):
            bs = 1
        
        with tqdm(total=total, desc=f"Inference {self.model_id.split('/')[-1]}", unit="img") as pbar:
            for start in range(0, total, bs):
                end = min(start + bs, total)
                batch_imgs = images[start:end]
                batch_meta = meta[start:end]
                
                try:
                    batch_preds = self._predict_batch(batch_imgs)
                except Exception as e:
                    print(f"Batch failed ({e}). Retrying serially...")
                    cleanup_memory()
                    batch_preds = self._predict_serial_fallback(batch_imgs)

                # Collect results
                for i, pred in enumerate(batch_preds):
                    results.append({**batch_meta[i], "prediction": pred})
                
                pbar.update(len(batch_imgs))
        
        return results

    def _predict_batch(self, images: List[Image.Image]) -> List[str]:
        if self.is_florence_vl:
            return self._predict_florence_vl(images)
        if self.is_ovis:
            return self._predict_ovis(images)
        if self.use_pipeline:
            # Prepare inputs for pipeline
            inputs = [
                [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": self.args.prompt}]}]
                for img in images
            ]
            outputs = self.pipe(inputs, batch_size=len(images), max_new_tokens=self.args.max_new_tokens)
            return [extract_pipeline_text(o) for o in outputs]
        elif self.has_chat_method:
            # Models with .chat() interface (InternVL2, Ovis)
            # These handle their own image transforms internally
            generation_config = dict(max_new_tokens=self.args.max_new_tokens, do_sample=False)
            preds = []
            for img in images:
                # InternVL2's .chat() expects pixel_values from its own transform
                if hasattr(self.model, 'build_transform'):
                    transform = self.model.build_transform(input_size=getattr(self.model.config, 'force_image_size', 448))
                    pixel_values = transform(img).unsqueeze(0).to(self.torch_device, dtype=self.dtype)
                else:
                    # Fallback: try to use torchvision transforms
                    import torchvision.transforms as T
                    transform = T.Compose([
                        T.Resize((448, 448)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    pixel_values = transform(img).unsqueeze(0).to(self.torch_device, dtype=self.dtype)
                response = self.model.chat(self.processor, pixel_values, self.args.prompt, generation_config)
                preds.append(response.strip())
            return preds
        else:
            # Manual batch
            prompts = [self._build_prompt() for _ in images]

            # Prepare inputs
            # Some processors (e.g. Phi-3-vision) don't accept text as a list
            if len(images) == 1:
                inputs = self.processor(
                    text=prompts[0],
                    images=images,
                    return_tensors="pt",
                ).to(device=self.torch_device, dtype=self.dtype)
            else:
                inputs = self.processor(
                    text=prompts,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                ).to(device=self.torch_device, dtype=self.dtype)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False
                )
            
            # Decode
            if self.is_encoder_decoder:
                # Encoder-decoder models already return only new tokens
                new_ids = generated_ids
            else:
                # Decoder-only models: strip the input prefix
                input_len = inputs.input_ids.shape[1]
                new_ids = generated_ids[:, input_len:]
            return [p.strip() for p in self.processor.batch_decode(new_ids, skip_special_tokens=True)]

    def _predict_ovis(self, images: List[Image.Image]) -> List[str]:
        """Inference for Ovis2.5 models."""
        preds = []
        for img in images:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": self.args.prompt},
                ],
            }]
            input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
                messages=messages, add_generation_prompt=True
            )
            input_ids = input_ids.to(self.model.device)
            if pixel_values is not None:
                pixel_values = pixel_values.to(self.model.device)
            if grid_thws is not None:
                grid_thws = grid_thws.to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    pixel_values=pixel_values,
                    grid_thws=grid_thws,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                )
            text = self.model.text_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            preds.append(text)
        return preds

    def _predict_florence_vl(self, images: List[Image.Image]) -> List[str]:
        """Inference for Florence-VL models (LLaVA-based architecture)."""
        preds = []
        for img in images:
            prompt = f"<image>\n{self.args.prompt}"
            if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                msgs = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            ).unsqueeze(0).to(self.model.device)

            image_sizes = [img.size]
            # Florence2Processor wraps a CLIPImageProcessor; extract it if needed
            ip = self.image_processor
            if hasattr(ip, 'image_processor'):
                ip = ip.image_processor
            images_tensor = process_images([img], ip, self.model.config)
            if isinstance(images_tensor, list):
                images_tensor = [t.to(self.model.device, dtype=self.dtype) for t in images_tensor]
            else:
                images_tensor = images_tensor.to(self.model.device, dtype=self.dtype)

            with torch.no_grad():
                output_ids = self.model.generate(
                    inputs=input_ids,
                    images=images_tensor,
                    image_sizes=image_sizes,
                    max_new_tokens=self.args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            # Model may already strip input tokens from output
            if output_ids.shape[1] > input_ids.shape[1]:
                new_ids = output_ids[:, input_ids.shape[1]:]
            else:
                new_ids = output_ids
            text = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
            preds.append(text)
        return preds

    def _predict_serial_fallback(self, images: List[Image.Image]) -> List[str]:
        preds = []
        for img in images:
            try:
                preds.append(self._predict_batch([img])[0])
            except Exception as e:
                print(f"Image failed: {e}")
                preds.append("ERROR")
            finally:
                cleanup_memory()
        return preds

    def _build_prompt(self) -> str:
        # Try chat template first
        msgs = [{"role": "user", "content": f"{self.placeholder}\n{self.args.prompt}"}]
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                result = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                if isinstance(result, list):
                    result = result[0] if result else ""
                result = str(result)
                # Validate the template actually included our prompt text
                if self.args.prompt in result and self.placeholder in result:
                    return result
                # Chat template stripped content; fall through to manual format
            except Exception:
                pass
        # LLaVA 1.5 HF models use USER: <image>\n{prompt} ASSISTANT: format
        if "llava" in self.model_id.lower() and self.placeholder == "<image>":
            return f"USER: {self.placeholder}\n{self.args.prompt} ASSISTANT:"
        # Non-chat models (e.g. PaliGemma) just want raw prompt
        return self.args.prompt


# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
def load_data(args) -> Tuple[List[Image.Image], List[Dict]]:
    df = pd.read_csv(args.input_csv)
    if args.image_col not in df.columns:
        raise ValueError(f"Column {args.image_col} missing in CSV")
    
    if args.limit > 0:
        df = df.head(args.limit)

    base_dir = args.base_dir
    images = []
    meta = []
    
    print(f"Loading {len(df)} images...")
    for _, row in df.iterrows():
        raw_path = str(row[args.image_col]).lstrip("/")
        image_path = (base_dir / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path)
        try:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((200, 200), Image.LANCZOS)
            images.append(img)
            meta.append({"image_path": raw_path, "label": row.get("label", "")})
        except Exception as e:
            print(f"Warning: Could not load {image_path}: {e}")
    
    return images, meta


def main():
    args = parse_args()

    # Test mode overrides limit
    if args.test:
        args.limit = args.test

    # 1. Load Data
    images, meta = load_data(args)
    if not images:
        print("No images loaded. Exiting.")
        return

    # 2. Setup Engine
    engine = InferenceEngine(args)

    # 3. Run Inference
    results = engine.run(images, meta)

    # 4. Output
    if args.test:
        print(f"\n{'='*60}")
        print(f"TEST RESULTS ({args.model_id})")
        print(f"{'='*60}")
        for r in results:
            print(f"  Label: {r['label']:20s}  Prediction: {r['prediction']}")
        print(f"{'='*60}")
    else:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        print(f"Saved {len(results)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
