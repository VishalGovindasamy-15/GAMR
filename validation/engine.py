"""
ValidationEngine — runs HF reference inference and compares to GAMR output.

Every run ends with PASS or FAIL.
PASS = GAMR output tokens match HF reference tokens exactly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("gamr.validation")


@dataclass
class ValidationResult:
    passed:          bool
    prompt:          str
    reference_text:  str
    gamr_text:       str
    reference_tokens: List[int] = field(default_factory=list)
    gamr_tokens:      List[int] = field(default_factory=list)
    mismatch_index:   Optional[int] = None
    reference_time_s: float = 0.0
    gamr_time_s:      float = 0.0

    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        lines = [
            f"Validation: {status}",
            f"  Prompt         : {self.prompt!r}",
            f"  Reference time : {self.reference_time_s:.2f}s",
            f"  GAMR time      : {self.gamr_time_s:.2f}s",
        ]
        if not self.passed:
            lines.append(f"  First mismatch : token index {self.mismatch_index}")
            lines.append(f"  Reference text : {self.reference_text!r}")
            lines.append(f"  GAMR text      : {self.gamr_text!r}")
        return "\n".join(lines)


class ValidationEngine:
    """
    Runs the reference model (full GPU load) and compares token-by-token
    against the GAMR streaming output.
    """

    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 50) -> None:
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens

    def run_reference(self, prompt: str) -> tuple[str, List[int], float]:
        """Run inference with the full model loaded to GPU (standard HF pipeline)."""
        logger.info("Running HF reference inference (full model on GPU)...")
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map=self.device,
        )
        model.eval()
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs["input_ids"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        elapsed = time.time() - t0
        token_ids = output_ids[0].tolist()
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        del model
        torch.cuda.empty_cache()
        logger.info(f"Reference done in {elapsed:.2f}s → {len(token_ids)} tokens")
        return text, token_ids, elapsed

    def compare(
        self,
        prompt: str,
        gamr_text: str,
        gamr_tokens: List[int],
        gamr_time_s: float,
    ) -> ValidationResult:
        """Compare GAMR output against HF reference."""
        ref_text, ref_tokens, ref_time = self.run_reference(prompt)

        mismatch = None
        passed = (ref_tokens == gamr_tokens)
        if not passed:
            for i, (r, g) in enumerate(zip(ref_tokens, gamr_tokens)):
                if r != g:
                    mismatch = i
                    break
            if mismatch is None:
                mismatch = min(len(ref_tokens), len(gamr_tokens))

        result = ValidationResult(
            passed=passed,
            prompt=prompt,
            reference_text=ref_text,
            gamr_text=gamr_text,
            reference_tokens=ref_tokens,
            gamr_tokens=gamr_tokens,
            mismatch_index=mismatch,
            reference_time_s=ref_time,
            gamr_time_s=gamr_time_s,
        )
        logger.info(result.summary())
        return result
