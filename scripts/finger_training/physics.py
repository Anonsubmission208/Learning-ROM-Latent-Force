import os, json, argparse, yaml, torch, lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from learning_dynamics.data_modules import FingerReconDataModule, FingerExtrapDataModule
from learning_dynamics.models import FingerPEGPVAEModel, Encoder, Decoder
from learning_dynamics.callbacks import FingerPEGPVAEPlotting

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_config_copy(cfg: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        
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

def main():
    p = argparse.ArgumentParser("Train FingerPEGPVAE")
    p.add_argument("--config", required=True, help="YAML with data/model/trainer fields")
    p.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    p.add_argument("--devices", default="1", help='e.g. "1" or "auto"')         
    p.add_argument("--max_epochs", type=int, default=None, help="Override epochs")
    p.add_argument("--task", choices=["recon", "extrap"], help="Override config task")

    args = p.parse_args()

    cfg = load_yaml(args.config)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.task is not None:
        cfg["task"] = args.task

    L.seed_everything(int(cfg["seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    torch.set_default_dtype(torch.float64)  

    dm = make_datamodule(cfg)
    dm.setup("fit")  

    task = str(cfg["task"]).lower()
    if task == "recon":
        width = dm.train_dataset.dataset.width
        height = dm.train_dataset.dataset.height
        dt = dm.train_dataset.dataset.dt
    else:
        width = dm.train_dataset.width
        height = dm.train_dataset.height
        dt = dm.train_dataset.dt
        
    embed_dim = int(cfg["embed_dim"])
    latent_dim = int(cfg["latent_dim"])
    
    enc = Encoder(width, height, embed_dim, latent_dim)
    dec = Decoder(width, height, embed_dim, latent_dim)

    model = FingerPEGPVAEModel(
        encoder=enc,
        decoder=dec,
        dt=float(dt),
        freq=float(cfg["freq"]),
        damp=float(cfg["damp"]),
        length_scale=float(cfg["length_scale"]),
        trainable_params=bool(cfg["trainable_params"]),
        trainable_ls=bool(cfg["trainable_ls"]),
        amp=bool(cfg["amp"]),
        scale=bool(cfg["scale"]),
    )

    model.hparams.update({
        "task": cfg["task"],
        "width": width, "height": height,
        "embed_dim": embed_dim,
        "latent_dim": latent_dim,
    })

    config_name = Path(args.config).stem
    cfg["run_name"] = config_name
    run_name = f"{cfg["run_name"]}_seed_{cfg["seed"]}"
    ckpt_root = cfg["ckpt_root"]
    ckpt_dir = os.path.join(ckpt_root, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    save_config_copy(cfg, ckpt_dir)
    
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="model-{epoch:04d}",
        monitor="validation/elbo",
        mode="max",
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    
    cbs = [ckpt_cb, LearningRateMonitor(logging_interval="epoch"), RichProgressBar()]

    if bool(cfg["make_gifs"]):
        plot_cb = FingerPEGPVAEPlotting(
            dt=dt,
            file=run_name,           
            max_gifs_per_epoch=10,    
            vids_per_batch=1,
            every_n_val_epochs=1
        )
        cbs.insert(0, plot_cb)

    logger = CSVLogger("logs", name=run_name, version=0)

    trainer = L.Trainer(
        max_epochs=args.max_epochs or int(cfg["max_epochs"]),
        devices=[1],
        precision="64-true",
        deterministic=True,
        check_val_every_n_epoch=int(cfg["val_check"]),
        log_every_n_steps=int(cfg["log_every_n_steps"]),
        callbacks=cbs,
        logger=logger,
    )
    
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        print(f"Resuming training from checkpoint: {last_ckpt}")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        print("No previous checkpoint found — starting fresh")
        trainer.fit(model, datamodule=dm)

    torch.save(model.state_dict(), os.path.join(ckpt_dir, "weights.pt"))

    trainer.test(model=None, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()