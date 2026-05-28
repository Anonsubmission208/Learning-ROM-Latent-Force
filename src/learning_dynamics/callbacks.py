import lightning as L
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import imageio
from pathlib import Path
import os
from PIL import Image, ImageDraw
import time
import csv

class PendulumPEGPVAEPlotting(L.pytorch.callbacks.Callback):
    def __init__(
        self,
        dt,
        file,
        *,
        max_gifs_per_epoch=10,
        vids_per_batch=1,
        every_n_val_epochs=1,
        out_dir=None,     
    ):
        super().__init__()
        self.dt = dt
        self.file = file

        self.max_gifs_per_epoch = max_gifs_per_epoch
        self.vids_per_batch = vids_per_batch
        self.every_n_val_epochs = every_n_val_epochs

        self.base_dir = out_dir
        self.figs_val_dir = None
        self.figs_val_gifs_dir = None
        self.figs_test_dir = None
        self.figs_test_gifs_dir = None
        self.curves_dir = None

        # running summaries for curves
        self.frequency_per_epoch = []
        self.damping_per_epoch = []
        self.length_scale_per_epoch = []
        self.current_epoch_freq = []
        self.current_epoch_damping = []
        self.current_epoch_length_scale = []

        # simple metric history (for curves; no CSV)
        self.hist = {
            "epoch": [],
            "train_recon": [], "val_recon": [],
            "train_kld":   [], "val_kld":   [],
            "train_elbo":  [], "val_elbo":  [],
        }
        self._last_train = None
        self._gif_counter_epoch = 0

    def _resolve_dirs(self, trainer):
        log_dir = getattr(getattr(trainer, "logger", None), "log_dir", None)
        if log_dir is None:
            # fallback if someone disables CSVLogger
            log_dir = os.path.join(trainer.default_root_dir, "logs", self.file, "version_manual")

        if self.base_dir is None:
            self.base_dir = log_dir

        figs = os.path.join(self.base_dir, "figs")
        self.figs_val_dir       = os.path.join(figs, "val")
        self.figs_val_gifs_dir  = os.path.join(figs, "val_gifs")
        self.figs_test_dir      = os.path.join(figs, "test")
        self.figs_test_gifs_dir = os.path.join(figs, "test_gifs")
        self.curves_dir         = os.path.join(figs, "curves")

        for d in [
            self.base_dir, self.figs_val_dir, self.figs_val_gifs_dir,
            self.figs_test_dir, self.figs_test_gifs_dir, self.curves_dir
        ]:
            os.makedirs(d, exist_ok=True)

    def on_fit_start(self, trainer, pl_module):
        self._resolve_dirs(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._gif_counter_epoch = 0

    def on_test_epoch_start(self, trainer, pl_module):
        self._gif_counter_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        def _get(name):
            v = cm.get(name, None)
            try:
                return float(v)
            except Exception:
                return None
        self._last_train = {
            "epoch": int(trainer.current_epoch),
            "recon": _get("training/recon"),
            "kld":   _get("training/kld"),
            "loss":  _get("training/loss"),
        }

    def heatmap(self, frames, time):
        video = np.array([(t+4)*v for t,v in enumerate(frames)]) 
        return np.max(video, 0)*(1/(4+time))

    def mse_scale(self, a, b):
        return (torch.sum(b*a))/torch.sum(b**2)
    
    def plot(self, true_frames, reconstructed_frames, trajectories, mean, var, latents, vid_idx):
        num_frames = true_frames.shape[1]
        time = num_frames * self.dt
        times = true_frames.shape[1]
        num_cols = 4
        rand_idxs = np.full(num_cols, vid_idx, dtype=int)
        rand_times = np.random.choice(times, size=num_cols, replace=False)

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(3, num_cols, height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.05)
        
        for i in range(num_cols):
            ax = fig.add_subplot(gs[0, i])
            # ax.imshow(true_frames[rand_idxs[i], rand_times[i], :, :]) 
            # ax.set_title(f"t = {rand_times[i]/10}", fontsize=9)
            # ax.set_xticks([]); ax.set_yticks([])
            image = self.heatmap(true_frames[i,:,:,:], num_frames)
            ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])

        for i in range(num_cols):
            ax = fig.add_subplot(gs[1, i])
            # ax.imshow(reconstructed_frames[rand_idxs[i], rand_times[i], :, :])
            # ax.set_xticks([]); ax.set_yticks([])

            image = self.heatmap(reconstructed_frames[i,:,:,:], num_frames)
            ax.imshow(image, extent=[0, image.shape[1], 0, image.shape[0]])

        scale = self.mse_scale(trajectories.cpu().numpy(), mean.cpu())

        ax1 = fig.add_subplot(gs[2, :2])
        mean_i = scale * mean[vid_idx].squeeze().cpu()
        std_i = scale * np.sqrt(var[vid_idx].squeeze().cpu())
        x = np.linspace(0, time, len(mean_i))
        ax1.plot(x, mean_i, label="gp")
        ax1.fill_between(x, mean_i + 2*std_i, mean_i - 2*std_i, alpha=0.5)
        ax1.legend()

        ax2 = fig.add_subplot(gs[2, 2:])
        time_points = np.linspace(0, time, trajectories[vid_idx].shape[0])
        trajectory = trajectories[vid_idx].detach().cpu().numpy().squeeze()
        ax2.plot(time_points, trajectory, label="ground truth")
        ax2.legend()

        fig.tight_layout()
        plt.close()
        return fig

    def on_validation_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        self._resolve_dirs(trainer)
        (loss, (trajectories, kld, recon_loss, gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping, length_scale)) = outputs

        frames, _ = batch
        frames = frames.cpu()
        bsz = frames.shape[0]
        vid_idx = np.random.randint(bsz)

        reconstructed_video = (reconstructed_video > 0.5).int()

        # print("Frequency", frequency, "\nDamping", damping, "\nLength Scale", length_scale)

        fig = self.plot(frames, reconstructed_video.cpu().numpy(), trajectories, gp_mean, gp_var, None, vid_idx)
        fig.suptitle(
            f"Epoch {trainer.current_epoch}  ELBO: {-float(loss):.4f}  "
            f"Kld: {float(kld):.4f}  Recon: {float(recon_loss):.4f}"
        )
        fig.savefig(
            os.path.join(self.figs_val_dir, f"epoch_{trainer.current_epoch:04d}_sample_{vid_idx}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        if frequency is not None:
            self.current_epoch_freq.append(float(frequency))
            self.current_epoch_damping.append(float(damping))
        if length_scale is not None:
            self.current_epoch_length_scale.append(float(length_scale))

        if (trainer.current_epoch % self.every_n_val_epochs) == 0 and self._gif_counter_epoch < self.max_gifs_per_epoch:
            K = min(self.vids_per_batch, bsz, self.max_gifs_per_epoch - self._gif_counter_epoch)
            for v in range(K):
                out = os.path.join(self.figs_val_gifs_dir, f"{self.file}_video_{trainer.current_epoch}_{v}.gif")
                self.save_recon_gif(
                    frames[v], reconstructed_video[v],
                    save_path=out, fps=None, cmap="inferno",
                    title_left="Ground Truth", title_right="Reconstruction"
                )
                self._gif_counter_epoch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        # average params for this epoch
        if self.current_epoch_freq:
            self.frequency_per_epoch.append(float(np.mean(self.current_epoch_freq)))
            self.damping_per_epoch.append(float(np.mean(self.current_epoch_damping)))
            self.length_scale_per_epoch.append(float(np.mean(self.current_epoch_length_scale)))
        self.current_epoch_freq.clear(); self.current_epoch_damping.clear(); self.current_epoch_length_scale.clear()

        # collect metrics for curves
        cm = trainer.callback_metrics
        def _get(name):
            v = cm.get(name, None)
            try:
                return float(v)
            except Exception:
                return None

        e = int(trainer.current_epoch)
        tr = self._last_train or {}
        train_recon = tr.get("recon", _get("training/recon"))
        train_kld   = tr.get("kld",   _get("training/kld"))
        train_loss  = tr.get("loss",  _get("training/loss"))
        train_elbo  = (-train_loss) if (train_loss is not None) else None

        val_recon = _get("validation/recon")
        val_kld   = _get("validation/kld")
        val_loss  = _get("validation/loss")
        val_elbo  = (-val_loss) if (val_loss is not None) else None

        self.hist["epoch"].append(e)
        self.hist["train_recon"].append(train_recon); self.hist["val_recon"].append(val_recon)
        self.hist["train_kld"].append(train_kld);     self.hist["val_kld"].append(val_kld)
        self.hist["train_elbo"].append(train_elbo);   self.hist["val_elbo"].append(val_elbo)

        # --- params curve ---
        if self.frequency_per_epoch:
            x = self.hist["epoch"]
            fig, (ax_f, ax_d, ax_l) = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
            ax_f.plot(x, self.frequency_per_epoch, marker="o", lw=1); ax_f.set_title("Frequency"); ax_f.set_xlabel("epoch"); ax_f.grid(True)
            ax_d.plot(x, self.damping_per_epoch,   marker="o", lw=1); ax_d.set_title("Damping");   ax_d.set_xlabel("epoch"); ax_d.grid(True)
            ax_l.plot(x, self.length_scale_per_epoch, marker="o", lw=1); ax_l.set_title("Length scale"); ax_l.set_xlabel("epoch"); ax_l.grid(True)
            fig.savefig(os.path.join(self.curves_dir, f"params_epoch_{e:04d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- metric curves ---
        if self.hist["epoch"]:
            x = self.hist["epoch"]
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
            ax[0].plot(x, self.hist["train_recon"], marker="o", lw=1, label="train")
            ax[0].plot(x, self.hist["val_recon"],   marker="x", lw=1, label="val")
            ax[0].set_title("Recon (sum/B)"); ax[0].set_xlabel("epoch"); ax[0].grid(True); ax[0].legend()

            ax[1].plot(x, self.hist["train_kld"], marker="o", lw=1, label="train")
            ax[1].plot(x, self.hist["val_kld"],   marker="x", lw=1, label="val")
            ax[1].set_title("KLD"); ax[1].set_xlabel("epoch"); ax[1].grid(True); ax[1].legend()

            ax[2].plot(x, self.hist["train_elbo"], marker="o", lw=1, label="train")
            ax[2].plot(x, self.hist["val_elbo"],   marker="x", lw=1, label="val")
            ax[2].set_title("ELBO (= -loss)"); ax[2].set_xlabel("epoch"); ax[2].grid(True); ax[2].legend()

            fig.savefig(os.path.join(self.curves_dir, f"metrics_epoch_{e:04d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---------- test ----------
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._resolve_dirs(trainer)
        (loss, (trajectories, kld, recon_loss, gp_mean, gp_var,
                latent_samples, reconstructed_video, frequency, damping, length_scale)) = outputs

        frames, _ = batch
        frames = frames.cpu()
        vid_idx = 0

        fig = self.plot(frames, reconstructed_video.cpu().numpy(),
                        trajectories, gp_mean, gp_var, None, vid_idx)
        fig.suptitle(
            f"[TEST] Epoch {trainer.current_epoch}  ELBO: {-float(loss):.4f}  "
            f"Kld: {float(kld):.4f}  Recon: {float(recon_loss):.4f}"
        )
        fig.savefig(
            os.path.join(self.figs_test_dir, f"test_epoch_{trainer.current_epoch:04d}_sample_{vid_idx}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        if self._gif_counter_epoch < self.max_gifs_per_epoch:
            out = os.path.join(self.figs_test_gifs_dir, f"test_video_{trainer.current_epoch}_{vid_idx}.gif")
            self.save_recon_gif(frames[vid_idx], reconstructed_video[vid_idx], save_path=out, fps=None)
            self._gif_counter_epoch += 1

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def save_recon_gif(
        self,
        true_frames,
        recon_frames,
        save_path="reconstruction.gif",
        fps=None,
        cmap="inferno",
        title_left="Ground Truth",
        title_right="Reconstruction",
        dpi=120,
        pad_inches=0.05,
    ):
        # Convert to numpy and squeeze to (T, H, W)
        def to_np(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
            except Exception:
                pass
            return np.asarray(x)

        tf = to_np(true_frames)
        rf = to_np(recon_frames)

        # If batched (B, T, H, W), take the first item
        if tf.ndim == 4:  # (B, T, H, W)
            tf = tf[0]
        if rf.ndim == 4:
            rf = rf[0]

        assert tf.ndim == 3 and rf.ndim == 3, "Inputs must be (T,H,W) or (B,T,H,W)."
        T = min(tf.shape[0], rf.shape[0])

        # Normalize color scale across both videos for consistency
        vmin = float(np.nanmin([tf[:T].min(), rf[:T].min()]))
        vmax = float(np.nanmax([tf[:T].max(), rf[:T].max()]))

        # Decide FPS
        if fps is None:
            # Use dt if available; fall back to 20 fps
            if hasattr(self, "dt") and self.dt and self.dt > 0:
                fps = max(1, int(round(1.0 / float(self.dt))))
            else:
                fps = 20

        images = []
        # Make frames
        for t in range(T):
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(8, 3), dpi=dpi, constrained_layout=True
            )

            axL, axR = axes
            imL = axL.imshow(tf[t], cmap=cmap, vmin=vmin, vmax=vmax)
            axL.set_title(title_left, fontsize=10)
            axL.set_xticks([]); axL.set_yticks([])

            imR = axR.imshow(rf[t], cmap=cmap, vmin=vmin, vmax=vmax)
            axR.set_title(title_right, fontsize=10)
            axR.set_xticks([]); axR.set_yticks([])

            if hasattr(self, "dt") and self.dt:
                time_s = t * float(self.dt)
                fig.suptitle(f"t = {time_s:.3f} s", fontsize=9, y=0.98)

            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
            images.append(frame)
            plt.close(fig)

        imageio.mimsave(save_path, images, duration=1.0 / fps, loop=0)
        return save_path

class FingerPEGPVAEPlotting(L.pytorch.callbacks.Callback):
    def __init__(
        self,
        dt,
        file,
        *,
        max_gifs_per_epoch=10,
        vids_per_batch=1,
        every_n_val_epochs=1,
        out_dir=None,     
    ):
        super().__init__()
        self.dt = dt
        self.file = file

        self.max_gifs_per_epoch = max_gifs_per_epoch
        self.vids_per_batch = vids_per_batch
        self.every_n_val_epochs = every_n_val_epochs

        self.base_dir = out_dir
        self.figs_val_dir = None
        self.figs_val_gifs_dir = None
        self.figs_test_dir = None
        self.figs_test_gifs_dir = None
        self.curves_dir = None

        # running summaries for curves
        self.frequency_per_epoch = []
        self.damping_per_epoch = []
        self.length_scale_per_epoch = []
        self.current_epoch_freq = []
        self.current_epoch_damping = []
        self.current_epoch_length_scale = []

        # simple metric history (for curves; no CSV)
        self.hist = {
            "epoch": [],
            "train_recon": [], "val_recon": [],
            "train_kld":   [], "val_kld":   [],
            "train_elbo":  [], "val_elbo":  [],
        }
        self._last_train = None
        self._gif_counter_epoch = 0

    # ---------- paths ----------
    def _resolve_dirs(self, trainer):
        log_dir = getattr(getattr(trainer, "logger", None), "log_dir", None)
        if log_dir is None:
            # fallback if someone disables CSVLogger
            log_dir = os.path.join(trainer.default_root_dir, "logs", self.file, "version_manual")

        if self.base_dir is None:
            self.base_dir = log_dir

        figs = os.path.join(self.base_dir, "figs")
        self.figs_val_dir       = os.path.join(figs, "val")
        self.figs_val_gifs_dir  = os.path.join(figs, "val_gifs")
        self.figs_test_dir      = os.path.join(figs, "test")
        self.figs_test_gifs_dir = os.path.join(figs, "test_gifs")
        self.curves_dir         = os.path.join(figs, "curves")

        for d in [
            self.base_dir, self.figs_val_dir, self.figs_val_gifs_dir,
            self.figs_test_dir, self.figs_test_gifs_dir, self.curves_dir
        ]:
            os.makedirs(d, exist_ok=True)

    # ---------- lifecycle ----------
    def on_fit_start(self, trainer, pl_module):
        self._resolve_dirs(trainer)

    def on_validation_epoch_start(self, trainer, pl_module):
        self._gif_counter_epoch = 0

    def on_test_epoch_start(self, trainer, pl_module):
        self._gif_counter_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        # stash train metrics so we can plot curves cleanly
        cm = trainer.callback_metrics
        def _get(name):
            v = cm.get(name, None)
            try:
                return float(v)
            except Exception:
                return None
        self._last_train = {
            "epoch": int(trainer.current_epoch),
            "recon": _get("training/recon"),
            "kld":   _get("training/kld"),
            "loss":  _get("training/loss"),
        }

    # ---------- panel plot ----------
    def plot(self, true_frames, reconstructed_frames, vae_mean, vae_var, mean, var, latents, vid_idx):
        num_frames = true_frames.shape[1]
        time = num_frames * self.dt
        times = true_frames.shape[1]
        num_cols = 4
        rand_idxs = np.full(num_cols, vid_idx, dtype=int)
        rand_times = np.random.choice(times, size=num_cols, replace=False)

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(3, num_cols, height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.05)

        # top: ground-truth frames
        for i in range(num_cols):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(true_frames[rand_idxs[i], rand_times[i], :, :])  # default cmap (viridis)
            ax.set_title(f"t = {rand_times[i]/10}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        # middle: reconstructed frames
        for i in range(num_cols):
            ax = fig.add_subplot(gs[1, i])
            ax.imshow(reconstructed_frames[rand_idxs[i], rand_times[i], :, :])
            ax.set_xticks([]); ax.set_yticks([])

        # bottom left: GP/latent mean ± 2σ
        ax1 = fig.add_subplot(gs[2, :2])
        mean_i = mean[vid_idx].squeeze().cpu()
        std_i = np.sqrt(var[vid_idx].squeeze().cpu())
        x = np.linspace(0, time, len(mean_i))
        ax1.plot(x, mean_i, label="gp" if vae_mean is not None else "vae")
        ax1.fill_between(x, mean_i + 2*std_i, mean_i - 2*std_i, alpha=0.5)
        ax1.legend()

        # bottom right: VAE posterior (if present)
        if vae_mean is not None:
            ax2 = fig.add_subplot(gs[2, 2:])
            mean_v = vae_mean[vid_idx].squeeze().cpu()
            std_v = np.sqrt(vae_var[vid_idx].squeeze().cpu())
            ax2.plot(x, mean_v, label="vae")
            ax2.fill_between(x, mean_v + 2*std_v, mean_v - 2*std_v, alpha=0.5)
            ax2.legend()

        fig.tight_layout()
        return fig

    # ---------- validation ----------
    def on_validation_batch_end(self, trainer: L.Trainer, pl_module, outputs, batch, batch_idx):
        self._resolve_dirs(trainer)
        (loss, (vae_mean, vae_var, kld, recon_loss, gp_mean, gp_var,
                latent_samples, reconstructed_video, frequency, damping, length_scale)) = outputs

        frames, _ = batch
        frames = frames.cpu()
        bsz = frames.shape[0]
        vid_idx = np.random.randint(bsz)

        print("Frequency", frequency, "\nDamping", damping, "\nLength Scale", length_scale)

        fig = self.plot(frames, reconstructed_video.cpu().numpy(),
                        vae_mean, vae_var, gp_mean, gp_var, None, vid_idx)
        fig.suptitle(
            f"Epoch {trainer.current_epoch}  ELBO: {-float(loss):.4f}  "
            f"Kld: {float(kld):.4f}  Recon: {float(recon_loss):.4f}"
        )
        fig.savefig(
            os.path.join(self.figs_val_dir, f"epoch_{trainer.current_epoch:04d}_sample_{vid_idx}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # accumulate per-epoch param stats for curves
        if frequency is not None:
            self.current_epoch_freq.append(float(frequency))
            self.current_epoch_damping.append(float(damping))
        if length_scale is not None:
            self.current_epoch_length_scale.append(float(length_scale))

        if (trainer.current_epoch % self.every_n_val_epochs) == 0 and self._gif_counter_epoch < self.max_gifs_per_epoch:
            K = min(self.vids_per_batch, bsz, self.max_gifs_per_epoch - self._gif_counter_epoch)
            for v in range(K):
                out = os.path.join(self.figs_val_gifs_dir, f"{self.file}_video_{trainer.current_epoch}_{v}.gif")
                self.save_recon_gif(
                    frames[v], reconstructed_video[v],
                    save_path=out, fps=None, cmap="inferno",
                    title_left="Ground Truth", title_right="Reconstruction"
                )
                self._gif_counter_epoch += 1

    def on_validation_epoch_end(self, trainer, pl_module):
        # average params for this epoch
        if self.current_epoch_freq:
            self.frequency_per_epoch.append(float(np.mean(self.current_epoch_freq)))
            self.damping_per_epoch.append(float(np.mean(self.current_epoch_damping)))
            self.length_scale_per_epoch.append(float(np.mean(self.current_epoch_length_scale)))
        self.current_epoch_freq.clear(); self.current_epoch_damping.clear(); self.current_epoch_length_scale.clear()

        # collect metrics for curves
        cm = trainer.callback_metrics
        def _get(name):
            v = cm.get(name, None)
            try:
                return float(v)
            except Exception:
                return None

        e = int(trainer.current_epoch)
        tr = self._last_train or {}
        train_recon = tr.get("recon", _get("training/recon"))
        train_kld   = tr.get("kld",   _get("training/kld"))
        train_loss  = tr.get("loss",  _get("training/loss"))
        train_elbo  = (-train_loss) if (train_loss is not None) else None

        val_recon = _get("validation/recon")
        val_kld   = _get("validation/kld")
        val_loss  = _get("validation/loss")
        val_elbo  = (-val_loss) if (val_loss is not None) else None

        self.hist["epoch"].append(e)
        self.hist["train_recon"].append(train_recon); self.hist["val_recon"].append(val_recon)
        self.hist["train_kld"].append(train_kld);     self.hist["val_kld"].append(val_kld)
        self.hist["train_elbo"].append(train_elbo);   self.hist["val_elbo"].append(val_elbo)

        # --- params curve ---
        if self.frequency_per_epoch:
            x = self.hist["epoch"]
            fig, (ax_f, ax_d, ax_l) = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
            ax_f.plot(x, self.frequency_per_epoch, marker="o", lw=1); ax_f.set_title("Frequency"); ax_f.set_xlabel("epoch"); ax_f.grid(True)
            ax_d.plot(x, self.damping_per_epoch,   marker="o", lw=1); ax_d.set_title("Damping");   ax_d.set_xlabel("epoch"); ax_d.grid(True)
            ax_l.plot(x, self.length_scale_per_epoch, marker="o", lw=1); ax_l.set_title("Length scale"); ax_l.set_xlabel("epoch"); ax_l.grid(True)
            fig.savefig(os.path.join(self.curves_dir, f"params_epoch_{e:04d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

        # --- metric curves ---
        if self.hist["epoch"]:
            x = self.hist["epoch"]
            fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)
            ax[0].plot(x, self.hist["train_recon"], marker="o", lw=1, label="train")
            ax[0].plot(x, self.hist["val_recon"],   marker="x", lw=1, label="val")
            ax[0].set_title("Recon (sum/B)"); ax[0].set_xlabel("epoch"); ax[0].grid(True); ax[0].legend()

            ax[1].plot(x, self.hist["train_kld"], marker="o", lw=1, label="train")
            ax[1].plot(x, self.hist["val_kld"],   marker="x", lw=1, label="val")
            ax[1].set_title("KLD"); ax[1].set_xlabel("epoch"); ax[1].grid(True); ax[1].legend()

            ax[2].plot(x, self.hist["train_elbo"], marker="o", lw=1, label="train")
            ax[2].plot(x, self.hist["val_elbo"],   marker="x", lw=1, label="val")
            ax[2].set_title("ELBO (= -loss)"); ax[2].set_xlabel("epoch"); ax[2].grid(True); ax[2].legend()

            fig.savefig(os.path.join(self.curves_dir, f"metrics_epoch_{e:04d}.png"), dpi=150, bbox_inches="tight")
            plt.close(fig)

    # ---------- test ----------
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._resolve_dirs(trainer)
        (loss, (vae_mean, vae_var, kld, recon_loss, gp_mean, gp_var,
                latent_samples, reconstructed_video, frequency, damping, length_scale)) = outputs

        frames, _ = batch
        frames = frames.cpu()
        vid_idx = 0

        fig = self.plot(frames, reconstructed_video.cpu().numpy(),
                        vae_mean, vae_var, gp_mean, gp_var, None, vid_idx)
        fig.suptitle(
            f"[TEST] Epoch {trainer.current_epoch}  ELBO: {-float(loss):.4f}  "
            f"Kld: {float(kld):.4f}  Recon: {float(recon_loss):.4f}"
        )
        fig.savefig(
            os.path.join(self.figs_test_dir, f"test_epoch_{trainer.current_epoch:04d}_sample_{vid_idx}.png"),
            dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        if self._gif_counter_epoch < self.max_gifs_per_epoch:
            out = os.path.join(self.figs_test_gifs_dir, f"test_video_{trainer.current_epoch}_{vid_idx}.gif")
            self.save_recon_gif(frames[vid_idx], reconstructed_video[vid_idx], save_path=out, fps=None)
            self._gif_counter_epoch += 1

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def save_recon_gif(
        self,
        true_frames,
        recon_frames,
        save_path="reconstruction.gif",
        fps=None,
        cmap="inferno",
        title_left="Ground Truth",
        title_right="Reconstruction",
        dpi=120,
        pad_inches=0.05,
    ):
        # Convert to numpy and squeeze to (T, H, W)
        def to_np(x):
            try:
                import torch
                if isinstance(x, torch.Tensor):
                    x = x.detach().cpu().numpy()
            except Exception:
                pass
            return np.asarray(x)

        tf = to_np(true_frames)
        rf = to_np(recon_frames)

        # If batched (B, T, H, W), take the first item
        if tf.ndim == 4:  # (B, T, H, W)
            tf = tf[0]
        if rf.ndim == 4:
            rf = rf[0]

        assert tf.ndim == 3 and rf.ndim == 3, "Inputs must be (T,H,W) or (B,T,H,W)."
        T = min(tf.shape[0], rf.shape[0])

        # Normalize color scale across both videos for consistency
        vmin = float(np.nanmin([tf[:T].min(), rf[:T].min()]))
        vmax = float(np.nanmax([tf[:T].max(), rf[:T].max()]))

        # Decide FPS
        if fps is None:
            # Use dt if available; fall back to 20 fps
            if hasattr(self, "dt") and self.dt and self.dt > 0:
                fps = max(1, int(round(1.0 / float(self.dt))))
            else:
                fps = 20

        images = []
        # Make frames
        for t in range(T):
            fig, axes = plt.subplots(
                nrows=1, ncols=2, figsize=(8, 3), dpi=dpi, constrained_layout=True
            )

            axL, axR = axes
            imL = axL.imshow(tf[t], cmap=cmap, vmin=vmin, vmax=vmax)
            axL.set_title(title_left, fontsize=10)
            axL.set_xticks([]); axL.set_yticks([])

            imR = axR.imshow(rf[t], cmap=cmap, vmin=vmin, vmax=vmax)
            axR.set_title(title_right, fontsize=10)
            axR.set_xticks([]); axR.set_yticks([])

            # Optional time label under the right plot
            if hasattr(self, "dt") and self.dt:
                time_s = t * float(self.dt)
                fig.suptitle(f"t = {time_s:.3f} s", fontsize=9, y=0.98)

            # Render to RGB array
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            # use buffer_rgba instead of tostring_rgb
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
            images.append(frame)
            plt.close(fig)

        imageio.mimsave(save_path, images, duration=1.0 / fps, loop=0)
        return save_path