  import os, re, json, yaml, torch
from pathlib import Path
from typing import Optional
import lightning as L

from learning_dynamics.data_modules import FingerReconDataModule, FingerExtrapDataModule
from learning_dynamics.models import (
    FingerPEGPVAEModel,  
    FingerRBFVAEModel,   
    FingerVAEModel,     
    Encoder, Decoder
)

def load_yaml(path: Path):
    with path.open("r") as f:
        return yaml.safe_load(f)

def make_datamodule(cfg):
    task = str(cfg["task"]).lower()
    if task == "recon":
        return FingerReconDataModule(
            train_data_path=cfg["train_data_path"],
            batch_size=cfg["batch_size"],
            val_fraction=cfg["val_fraction"],
            test_fraction=cfg["test_fraction"],
            num_workers=cfg["num_workers"],
            seed=cfg["seed"],
        )
    elif task == "extrap":
        return FingerExtrapDataModule(
            train_data_path=cfg["train_data_path"],
            batch_size=cfg["batch_size"],
            test_batch_size=cfg["batch_size"],
            val_split=cfg["extrap_val_split"],
            test_split=cfg["extrap_test_split"],
            num_workers=cfg["num_workers"],
        )
    else:
        raise ValueError(f"Unknown task: {task}")

def _maybe_get(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

def _extract_dt_from_dataset(ds):
    if ds is None: return None
    for chain in [
        ("dt",),
        ("dataset", "dt"),
        ("base_dataset", "dt"),
        ("dataset", "dataset", "dt"),
    ]:
        cur = ds
        ok = True
        for c in chain:
            cur = getattr(cur, c, None)
            if cur is None:
                ok = False
                break
        if ok and isinstance(cur, (float, int)):
            return float(cur)
    return None

def get_dims_dt(cfg, dm, task, first_batch=None):
    ds_candidates = []
    ds_candidates.append(_maybe_get(dm, "test_dataset"))
    ds_candidates.append(_maybe_get(dm, "train_dataset"))
    try:
        ds_candidates.append(dm.test_dataloader().dataset)
    except Exception:
        pass
    ds_candidates = [d for d in ds_candidates if d is not None]
    expanded = []
    for d in ds_candidates:
        expanded.append(d)
        dd = getattr(d, "dataset", None)
        if dd is not None:
            expanded.append(dd)
    ds_candidates = expanded + ds_candidates

    width = height = None
    dt = None

    for d in ds_candidates:
        w = getattr(d, "width", None)
        h = getattr(d, "height", None)
        if w is not None and h is not None:
            width, height = int(w), int(h)
            break

    for d in ds_candidates:
        dt = _extract_dt_from_dataset(d)
        if dt is not None:
            break

    if (width is None or height is None) and first_batch is not None:
        x = None
        if isinstance(first_batch, dict):
            x = first_batch.get("x", None)
        if x is None and isinstance(first_batch, (list, tuple)) and len(first_batch) > 0:
            x = first_batch[0]
        if torch.is_tensor(x):
            # expected shape: (B, T, C, H, W) or (B, C, H, W) etc.
            if x.dim() >= 4:
                height = int(x.shape[-2])
                width  = int(x.shape[-1])

    # 4) last resort for dt: config may have it (often it doesn’t; but try)
    if dt is None and "dt" in cfg:
        try:
            dt = float(cfg["dt"])
        except Exception:
            pass

    if width is None or height is None or dt is None:
        raise RuntimeError(f"Could not resolve width/height/dt (got width={width}, height={height}, dt={dt}).")

    return width, height, dt

def detect_model_kind(cfg: dict, run_dir_name: str) -> str:
    name = run_dir_name.lower()
    if any(k in name for k in ["physics", "pegp", "pegpvae"]):
        return "physics"
    if "rbf" in name:
        return "rbf"
    keys = set(map(str.lower, cfg.keys()))
    if {"freq", "damp"} & keys:
        return "physics"
    if "length_scale" in keys and not ({"freq", "damp"} & keys):
        return "rbf"
    return "vae"

def build_model(kind: str, cfg: dict, dm, dims_dt):
    w, h, dt = dims_dt
    embed_dim = int(cfg["embed_dim"])
    latent_dim = int(cfg["latent_dim"])
    enc = Encoder(w, h, embed_dim, latent_dim)
    dec = Decoder(w, h, embed_dim, latent_dim)

    if kind == "physics":
        model = FingerPEGPVAEModel(
            encoder=enc, decoder=dec, dt=float(dt),
            freq=float(cfg["freq"]), damp=float(cfg["damp"]),
            length_scale=float(cfg["length_scale"]),
            trainable_params=bool(cfg["trainable_params"]),
            trainable_ls=bool(cfg["trainable_ls"]),
            amp=bool(cfg["amp"]), scale=bool(cfg["scale"]),
        )
    elif kind == "rbf":
        model = FingerRBFVAEModel(
            encoder=enc, decoder=dec, dt=float(dt),
            length_scale=float(cfg["length_scale"]),
            trainable_ls=bool(cfg["trainable_ls"]),
            amp=bool(cfg["amp"]), scale=bool(cfg["scale"]),
        )        
    else:
        model = FingerVAEModel(encoder=enc, decoder=dec, dt=float(dt))

    return model        

def find_checkpoint(run_dir: Path) -> Optional[Path]:
    last = run_dir / "last.ckpt"
    if last.is_file():
        return last
    bests = sorted(run_dir.glob("model-*.ckpt"))
    if bests:
        def epoch_of(p: Path) -> int:
            m = re.search(r"model-(\d+)\.ckpt$", p.name)
            return int(m.group(1)) if m else -1
        bests.sort(key=epoch_of, reverse=True)
        return bests[0]
    w = run_dir / "weights.pt"
    return w if w.is_file() else None

def load_weights(model: torch.nn.Module, ckpt_path: Path) -> str:
    state = torch.load(ckpt_path, map_location="cpu")
    if ckpt_path.suffix == ".ckpt" and isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    return f"loaded {ckpt_path.name} (missing={len(missing)}, unexpected={len(unexpected)})"

def to_cpu_detach(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x, (list, tuple)): return type(x)(to_cpu_detach(xx) for xx in x)
    if isinstance(x, dict): return {k: to_cpu_detach(v) for k, v in x.items()}
    return x

def move_to_device(x, device):
    if torch.is_tensor(x): return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)): return type(x)(move_to_device(xx, device) for xx in x)
    if isinstance(x, dict): return {k: move_to_device(v, device) for k, v in x.items()}
    return x

@torch.no_grad()
def first_test_batch(dm):
    dm.setup("test")
    loader = dm.test_dataloader()
    return next(iter(loader))
@torch.no_grad()
def forward_flexible(model, batch_dev):
    model.eval()
    if hasattr(model, "test_step"):
        try:
            return model.test_step(batch_dev)
        except TypeError:
            return model.test_step(batch_dev)  # will raise if truly incompatible

    if hasattr(model, "_step"):
        return model._step("test", batch_dev)

    try:
        return model(batch_dev)
    except Exception:
        if isinstance(batch_dev, (list, tuple)) and len(batch_dev) > 0:
            return model(batch_dev[0])
        return model.forward(batch_dev)


def postprocess_outputs(model, out):
    if isinstance(out, dict):
        return out

    if isinstance(out, (list, tuple)) and len(out) == 2:
        loss, payload = out
        if isinstance(payload, (list, tuple)) and len(payload) >= 11:
            (vae_mean, vae_var, kld, recon_loss,
             gp_mean, gp_var, latent_samples, reconstructed_video,
             frequency, damping, length_scale, *rest) = payload
            return {
                "loss": loss.detach().cpu() if hasattr(loss, "detach") else loss,
                "vae_mean": vae_mean, "vae_var": vae_var,
                "kld": kld, "recon_loss": recon_loss,
                "gp_mean": gp_mean, "gp_var": gp_var,
                "latent_samples": latent_samples,
                "reconstructed_video": reconstructed_video,
                "frequency": frequency, "damping": damping, "length_scale": length_scale,
                # keep any extras under "extra"
                "extra": rest if rest else None,
            }

    if isinstance(out, (list, tuple)):
        return {"items": list(out)}

    # tensor or other
    return {"raw": out}


def main():
    import argparse
    p = argparse.ArgumentParser("Collect first test batch (and outputs) for all runs")
    p.add_argument("--ckpt_root", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_output", action="store_true")
    args = p.parse_args()

    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.float64)

    ckpt_root = Path(args.ckpt_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_files = sorted(list(ckpt_root.rglob("config.yaml")) + list(ckpt_root.rglob("config.yml")))
    if not cfg_files:
        raise SystemExit(f"No runs found under {ckpt_root} (looking for config.yaml/.yml).")

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    for cfg_path in cfg_files:
        run_dir = cfg_path.parent
        run_name = run_dir.name

        try:
            cfg = load_yaml(cfg_path)
        except Exception as e:
            print(f"[SKIP] {run_name}: bad config.yaml ({e})")
            continue

        if "task" not in cfg:
            print(f"[SKIP] {run_name}: config missing 'task'")
            continue
        cfg["task"] = str(cfg["task"]).lower()

        L.seed_everything(int(cfg.get("seed", 0)), workers=True)
        print(f"Seed set to {int(cfg.get('seed', 0))}")

        try:
            dm = make_datamodule(cfg)
            batch_cpu = first_test_batch(dm)
        except Exception as e:
            print(f"[SKIP] {run_name}: could not get first test batch ({e})")
            continue

        kind = detect_model_kind(cfg, run_name)
        try:
            dims_dt = get_dims_dt(cfg, dm, cfg["task"], first_batch=batch_cpu)
        except Exception as e:
            print(f"[SKIP] {run_name}: failed to resolve dims/dt ({e})")
            continue

        model = None
        ckpt_used = None
        outputs_cpu = None

        if not args.no_output:
            try:
                model = build_model(kind, cfg, dm, dims_dt)
                if cfg["task"] == "extrap" and not isinstance(model, FingerVAEModel):
                    print("set limit")
                    model.test_t_limit = cfg["extrap_val_split"]
            except Exception as e:
                print(f"[SKIP] {run_name}: model build error ({e})")
                continue

            ckpt = find_checkpoint(run_dir)
            if ckpt is None:
                print(f"[SKIP] {run_name}: no checkpoint found (use --no_output to save inputs only)")
                continue

            try:
                msg = load_weights(model, ckpt)
                ckpt_used = ckpt.name
                print(f"[{run_name}] {kind}: {msg}")
            except Exception as e:
                print(f"[SKIP] {run_name}: load weights failed ({e})")
                continue

            bdev = move_to_device(batch_cpu, device)
                
            raw_out = forward_flexible(model.to(device), bdev)
            
            outputs_cpu = to_cpu_detach(postprocess_outputs(model, raw_out))


        save_obj = {"inputs": to_cpu_detach(batch_cpu)}
        if outputs_cpu is not None:
            save_obj["outputs"] = outputs_cpu

        pt_path = out_root / f"{run_name}_first_batch.pt"
        torch.save(save_obj, pt_path)

        meta = {
            "model_kind": kind,
            "task": cfg["task"],
            "seed": int(cfg.get("seed", 0)),
            "run_dir": str(run_dir),
            "checkpoint_used": ckpt_used,
            "saved_keys": list(save_obj.keys()),
        }
        with open(out_root / f"{run_name}_first_batch.meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[SAVED] {run_name} -> {pt_path.name} ({', '.join(meta['saved_keys'])})")

if __name__ == "__main__":
    main()
