"""Score SmolVLA layers for CLP-style pruning (CKA-guided layer pruning, arXiv 2606.20246).

Replays the calibration requests recorded by collect_calib.py through the PyTorch SmolVLA
(lerobot) and records the hidden state entering every layer of the VLM text model (prefix
pass) and of the action expert (every denoising step), plus the output of the last layer.
H_l denotes the input of layer l, so layer l maps H_l -> H_{l+1}.

Scores (all accumulated in a streaming way, float64 on CPU):
  - linear CKA (Kornblith et al. 2019, centered) between H_l and H_{l+1}  -> per layer
  - linear CKA between H_i and H_{i+2}  -> block of the adjacent pair (i, i+1)
  - mean cosine similarity of the same pairs of token vectors (ablation baseline)
Tokens: VLM = all non-padding prefix tokens (images, prompt, state); expert = the 50 action
tokens at all denoising steps. These choices are ours: the paper does not state them.

Sanity check: the PyTorch action for each request (same inputs, same noise) is compared
with the action vla-server returned when the request was recorded.

Usage (inside WSL, LIBERO venv; stop vla-server first, the model runs on the GPU):
  python cka_layers.py --calib ~/profile/calib --ckpt ~/models/smolvla_libero_hf \
      --out "<repo>/apps/vla.cpp/results/prune/cka"
Writes layer_scores.csv, pair_scores.csv, drop_lists.txt, sanity.txt into --out.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


class StreamStats:
    """Running sums for linear CKA / cosine between states H_j and H_{j+lag} (lag 1 and 2)."""

    def __init__(self, n_states, dim):
        self.L, self.d, self.n = n_states, dim, 0
        z = lambda k: torch.zeros(k, dim, dim, dtype=torch.float64)
        self.s = torch.zeros(n_states, dim, dtype=torch.float64)
        self.gram = z(n_states)
        self.cross = {1: z(n_states - 1), 2: z(n_states - 2)}
        self.cos = {1: torch.zeros(n_states - 1, dtype=torch.float64),
                    2: torch.zeros(n_states - 2, dtype=torch.float64)}

    def add(self, hs):                       # hs: list of L tensors [rows, dim]
        h = [x.to("cpu", torch.float64) for x in hs]
        self.n += h[0].shape[0]
        for j in range(self.L):
            self.s[j] += h[j].sum(0)
            self.gram[j] += h[j].T @ h[j]
        hn = [x / x.norm(dim=1, keepdim=True).clamp_min(1e-12) for x in h]
        for lag in (1, 2):
            for j in range(self.L - lag):
                self.cross[lag][j] += h[j].T @ h[j + lag]
                self.cos[lag][j] += (hn[j] * hn[j + lag]).sum()

    def cka(self, lag):
        mu = self.s / self.n
        cen = lambda S, a, b: S - self.n * torch.outer(mu[a], mu[b])
        norm = [torch.linalg.norm(cen(self.gram[j], j, j)) for j in range(self.L)]
        return [float(torch.linalg.norm(cen(self.cross[lag][j], j, j + lag)) ** 2 / (norm[j] * norm[j + lag]))
                for j in range(self.L - lag)]

    def cosine(self, lag):
        return [float(c / self.n) for c in self.cos[lag]]


def greedy_pairs(score, k, first=1):
    """Pick k non-overlapping adjacent pairs (i, i+1), i >= first, with the highest score."""
    order = sorted(range(first, len(score)), key=lambda i: -score[i])
    chosen = []
    for i in order:
        if len(chosen) == k:
            break
        if all(abs(i - j) >= 2 for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def spec(pairs):
    return ",".join(f"{i},{i + 1}" for i in pairs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-samples", type=int, default=0, help="0 = all")
    ap.add_argument("--ks", default="1,2,4", help="numbers of pairs for drop_lists.txt")
    ap.add_argument("--seed", type=int, default=0, help="for the random baseline")
    a = ap.parse_args()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla import smolvlm_with_expert
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    # SmolVLMWithExpertModel builds an AutoProcessor (tokenizer + image processor); lerobot only
    # reads processor.tokenizer.{fake,global}_image_token_id from it (modeling_smolvla.py:585-586).
    # The full processor needs the `num2words` package, which the LIBERO venv lacks, so build a
    # stand-in holding just the (cached) tokenizer instead of installing packages into the venv
    # inside the vla.cpp clone.
    from types import SimpleNamespace
    from transformers import AutoTokenizer

    class _TokenizerOnlyProcessor:
        @staticmethod
        def from_pretrained(model_id, *_args, **_kwargs):
            return SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(model_id))
    smolvlm_with_expert.AutoProcessor = _TokenizerOnlyProcessor

    ckpt = a.ckpt.expanduser()
    cfg = PreTrainedConfig.from_pretrained(ckpt)
    cfg.load_vlm_weights = False          # weights come from the checkpoint; skip the 1 GB SmolVLM2 download
    cfg.device = a.device
    policy = SmolVLAPolicy.from_pretrained(ckpt, config=cfg).to(a.device, torch.float32).eval()
    model = policy.model
    vwe = model.vlm_with_expert
    txt, exp = vwe.get_vlm_model().text_model, vwe.lm_expert
    n_layers = len(txt.layers)
    assert len(exp.layers) == n_layers, "expert and VLM layer counts differ"

    st = load_file(str(ckpt / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
    at = load_file(str(ckpt / "policy_postprocessor_step_1_unnormalizer_processor.safetensors"))
    s_mean, s_std = st["observation.state.mean"].float(), st["observation.state.std"].float()
    a_mean, a_std = at["action.mean"].float(), at["action.std"].float()
    eps = 1e-8

    # ---- hooks: input of every layer (pre-hook on input_layernorm) and of the final norm
    cur = {"vlm": [None] * (n_layers + 1), "exp": [None] * (n_layers + 1), "exp_steps": []}

    def grab(which, j):
        def hook(_m, args):
            cur[which][j] = args[0].detach()
        return hook

    for j, layer in enumerate(txt.layers):
        layer.input_layernorm.register_forward_pre_hook(grab("vlm", j))
    txt.norm.register_forward_pre_hook(grab("vlm", n_layers))
    for j, layer in enumerate(exp.layers):
        layer.input_layernorm.register_forward_pre_hook(grab("exp", j))

    def exp_done(_m, args):              # end of one denoising step: store the expert states
        cur["exp"][n_layers] = args[0].detach()
        cur["exp_steps"].append(list(cur["exp"]))
    exp.norm.register_forward_pre_hook(exp_done)

    vlm_stats = StreamStats(n_layers + 1, txt.config.hidden_size)
    exp_stats = StreamStats(n_layers + 1, exp.config.hidden_size)

    files = sorted(a.calib.expanduser().glob("task_*.npz"))
    if not files:
        raise SystemExit(f"no task_*.npz in {a.calib}")
    diffs, n_done, t0 = [], 0, time.time()
    chunk, adim = model.config.chunk_size, model.config.max_action_dim
    for fpath in files:
        d = np.load(fpath)
        for r in range(len(d["n_tokens"])):
            if a.max_samples and n_done >= a.max_samples:
                break
            imgs = torch.from_numpy(d["images"][r]).float().div(255)          # [V, H, W, 3] in [0, 1]
            images = [im.permute(2, 0, 1)[None].to(a.device) * 2 - 1 for im in imgs]   # SigLIP range [-1, 1]
            img_masks = [torch.ones(1, dtype=torch.bool, device=a.device) for _ in images]
            n_tok = int(d["n_tokens"][r])
            lang = torch.from_numpy(d["tokens"][r]).long()[None].to(a.device)
            lang_mask = (torch.arange(lang.shape[1]) < n_tok)[None].to(a.device)
            state = torch.from_numpy(d["state"][r]).float()
            sd = s_mean.numel()
            state[:sd] = (state[:sd] - s_mean) / (s_std + eps)
            state = state[None].to(a.device)
            noise = torch.from_numpy(d["noise"][r]).float().view(1, chunk, adim).to(a.device)

            cur["exp_steps"].clear()
            with torch.no_grad():
                x = model.sample_actions(images, img_masks, lang, lang_mask, state, noise=noise)

            # VLM rows: [image tokens | prompt (48, padded) | state]; drop the padded prompt positions
            T = cur["vlm"][0].shape[1]
            n_img = T - lang.shape[1] - 1
            keep = torch.cat([torch.ones(n_img, dtype=torch.bool), lang_mask[0].cpu(),
                              torch.ones(1, dtype=torch.bool)])
            vlm_stats.add([h[0][keep.to(h.device)] for h in cur["vlm"]])
            for hs in cur["exp_steps"]:
                exp_stats.add([h[0][-chunk:] for h in hs])

            act = x[0].float().cpu()
            ad = a_mean.numel()
            act_w = act[:, :ad] * (a_std + eps) + a_mean
            diffs.append((act_w - torch.from_numpy(d["action"][r][:, :ad])).abs())
            n_done += 1
            if n_done % 20 == 0:
                print(f"{n_done} samples, {time.time() - t0:.0f}s, "
                      f"mean |torch - server| so far {torch.stack(diffs).mean():.2e}", flush=True)

    out = a.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    dd = torch.stack(diffs)
    mean_abs = float(np.mean([np.abs(np.load(f)["action"][..., :7]).mean() for f in files]))
    sanity = (f"samples: {n_done} from {len(files)} files\n"
              f"VLM rows: {vlm_stats.n}  expert rows: {exp_stats.n} "
              f"(denoising steps seen per sample: {len(cur['exp_steps'])})\n"
              f"|torch action - vla-server action| over 7 dims: mean {dd.mean():.3e}  "
              f"max {dd.max():.3e}  (mean |action| {mean_abs:.3e})\n")
    (out / "sanity.txt").write_text(sanity)
    print(sanity)

    cv1, ce1 = vlm_stats.cka(1), exp_stats.cka(1)
    cv2, ce2 = vlm_stats.cka(2), exp_stats.cka(2)
    sv1, se1 = vlm_stats.cosine(1), exp_stats.cosine(1)
    sv2, se2 = vlm_stats.cosine(2), exp_stats.cosine(2)
    with open(out / "layer_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "type", "cka_vlm", "cka_expert", "cos_vlm", "cos_expert"])
        for l in range(n_layers):
            w.writerow([l, "SA" if l % 2 == 0 else "CA",
                        f"{cv1[l]:.5f}", f"{ce1[l]:.5f}", f"{sv1[l]:.5f}", f"{se1[l]:.5f}"])
    pair_cka = [(cv2[i] + ce2[i]) / 2 for i in range(n_layers - 1)]
    pair_cos = [(sv2[i] + se2[i]) / 2 for i in range(n_layers - 1)]
    with open(out / "pair_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "cka_vlm", "cka_expert", "cka_mean", "cos_vlm", "cos_expert", "cos_mean"])
        for i in range(n_layers - 1):
            w.writerow([f"{i}-{i + 1}", f"{cv2[i]:.5f}", f"{ce2[i]:.5f}", f"{pair_cka[i]:.5f}",
                        f"{sv2[i]:.5f}", f"{se2[i]:.5f}", f"{pair_cos[i]:.5f}"])

    rng = random.Random(a.seed)
    lines = ["# name|drop spec|method   (feed to make_pruned_gguf.sh)"]
    for k in (int(x) for x in a.ks.split(",")):
        lines.append(f"cka_k{k}|{spec(greedy_pairs(pair_cka, k))}|cka")
        lines.append(f"cos_k{k}|{spec(greedy_pairs(pair_cos, k))}|cosine")
        lines.append(f"last_k{k}|{n_layers - 2 * k}-{n_layers - 1}|last")
        rnd = greedy_pairs([rng.random() for _ in range(n_layers - 1)], k)
        lines.append(f"rand_k{k}|{spec(rnd)}|random(seed={a.seed})")
    (out / "drop_lists.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
