"""GenRec model (spec 5): 4-bit Qwen3-4B + LoRA(r=16, all-linear) + full-rank catalog head.

Pooling: hidden state at the final EOS appended after the prompt's "Task:" line.
Head is full-rank bf16 (new module -> LoRA can't adapt what doesn't exist).
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from verbalize import verbalize

BASE = "Qwen/Qwen3-4B"
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            target_modules="all-linear", task_type="CAUSAL_LM")


def load_tokenizer(base=BASE):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_backbone(base=BASE, trainable=True):
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb,
                                                 device_map={"": 0})  # ponytail: single GPU — device_map="auto" sharding broke accelerate hooks on 2xT4
    model.config.use_cache = False
    if trainable:
        prepare_model_for_kbit_training(model)
        model = get_peft_model(model, LoraConfig(**LORA))
        model.print_trainable_parameters()
    return model


class GenRecModel(nn.Module):
    def __init__(self, backbone, n_items):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, n_items)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.head.to(device=backbone.device, dtype=torch.bfloat16)

    def _causal_lm(self):
        return self.backbone.base_model.model  # PeftModel -> LoraModel -> Qwen3ForCausalLM

    def pool(self, input_ids, attention_mask):
        """Returns (h_all [B,L,d], h [B,d] at each row's final EOS position)."""
        h_all = self._causal_lm().model(input_ids=input_ids,
                                        attention_mask=attention_mask).last_hidden_state
        idx = attention_mask.sum(1) - 1  # right padding -> last non-pad is the appended EOS
        return h_all, h_all[torch.arange(h_all.size(0), device=h_all.device), idx]

    def forward(self, input_ids, attention_mask):
        h_all, h = self.pool(input_ids, attention_mask)
        rank_logits = self.head(h.to(self.head.weight.dtype)).float()  # CE in fp32
        return {"rank_logits": rank_logits, "h_all": h_all}

    def lm_logits(self, h_all):
        """Frozen base lm_head over full hidden sequence (for the beta*L_lm regularizer)."""
        return self._causal_lm().lm_head(h_all)

    def lm_loss(self, h_all, input_ids, attention_mask):
        labels = input_ids[:, 1:].clone()
        labels[attention_mask[:, 1:] == 0] = -100
        logits = self.lm_logits(h_all[:, :-1])
        return F.cross_entropy(logits.transpose(1, 2), labels)  # ponytail: bf16 CE, fp32 if loss curves look noisy

    def save(self, out_dir, step):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(out / "adapter")
        torch.save({"head": self.head.state_dict(), "step": step}, out / "head.pt")

    @classmethod
    def from_ckpt(cls, out_dir, base=BASE, trainable=False):
        backbone = load_backbone(base, trainable=False)
        if trainable:  # resume: prepare grads first, then load adapter as trainable LoRA
            prepare_model_for_kbit_training(backbone)
            backbone = PeftModel.from_pretrained(backbone, Path(out_dir) / "adapter",
                                                 is_trainable=True)
        else:
            backbone = PeftModel.from_pretrained(backbone, Path(out_dir) / "adapter")
        model = cls(backbone, n_items=head_dim_from_ckpt(out_dir))
        sd = torch.load(Path(out_dir) / "head.pt", map_location="cpu", weights_only=True)
        model.head.load_state_dict(sd["head"])
        model.eval()
        return model, sd["step"]


def head_dim_from_ckpt(out_dir):
    sd = torch.load(Path(out_dir) / "head.pt", map_location="cpu", weights_only=True)
    return sd["head"]["bias"].shape[0]


def genrec_scorer(model, tok, catalog_map, n_events=10, verbosity="full",
                  drop_low_signal=False, max_len=1024):
    """Plug into eval.py: score_fn(hist_item_ids, hist_events) -> scores [n_items]."""
    eos = tok.eos_token_id
    fit = lambda p: len(tok(p, add_special_tokens=False).input_ids)

    def score(hist_item_ids, hist_events):
        prompt = verbalize(hist_events, catalog_map, n_events=n_events, verbosity=verbosity,
                           drop_low_signal=drop_low_signal, fit_tokens=fit, max_len=max_len)
        ids = torch.tensor([tok(prompt).input_ids + [eos]], device=model.head.weight.device)
        with torch.no_grad():
            out = model(ids, torch.ones_like(ids))
        return out["rank_logits"][0].cpu().numpy()

    return score
