import torch
import torch.nn as nn
import lightning as L
import torch.nn.functional as F
import math
from .gaussian_process import gaussian_process_ode, gaussian_process_rbf
from .functions import gauss_cross_entropy, truncated_normal
            
###########################################################################################################
class Encoder(torch.nn.Module):
    def __init__(self, px, py, embed_dim, latent_dim):
        super().__init__()
        self.px = px
        self.py = py
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(self.px*self.py, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, 2*self.latent_dim)
        )

    def forward(self, x):
        h = x.flatten(2)                       # (B , T , H·W)
        params = self.encoder(h)               # (B , T , 2·L)
        mean, log_var = torch.chunk(params, 2, dim=-1)
        return mean, log_var
        
###########################################################################################################
class Decoder(torch.nn.Module):
    def __init__(self, px, py, embed_dim, latent_dim):
        super().__init__()
        self.px = px
        self.py = py
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.embed_dim),
            nn.Tanh(),
            nn.Linear(self.embed_dim, self.px*self.py)
        )
                
    def forward(self, x):
        batch_size = x.shape[0]
        reconstructed = self.decoder(x)
        reconstructed = reconstructed.view(batch_size, -1, self.py, self.px)
        return reconstructed

###########################################################################################################
class FingerPEGPVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, freq, damp, length_scale, trainable_params, trainable_ls, amp, scale):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt
        self.trainable_params = trainable_params
        self.trainable_ls = trainable_ls
        self.amp = amp
        self.scale = scale

        self.test_t_limit = None

        self.hparams.update({
            "width":  encoder.px,
            "height": encoder.py,
            "embed_dim": encoder.embed_dim,
            "latent_dim": encoder.latent_dim,
        })

        if self.trainable_params:
            self.frequency = nn.Parameter((torch.tensor(freq, dtype=torch.double)))
            self.damping = nn.Parameter((torch.tensor(damp, dtype=torch.double)))
        else:
            self.frequency = torch.tensor(freq, dtype=torch.double)
            self.damping = torch.tensor(0.5, dtype=torch.double)

        if self.trainable_ls:
            self.length_scale = nn.Parameter((torch.tensor(length_scale, dtype=torch.double)))
        else:
            self.length_scale = torch.tensor(0.6, dtype=torch.double)

        if self.amp:
            self.amp_param = nn.Parameter(torch.zeros((), dtype=torch.double))
            # self.amp_param = nn.Parameter(torch.ones((), dtype=torch.double))

        if self.scale:
            self.latent_log_scale = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))
            self.latent_bias = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))

    def _step(self, mode, batch):
        if self.trainable_params:
            frequency = torch.clamp(F.softplus(self.frequency), min=1)
            damping = torch.clamp(torch.sigmoid(self.damping), min=0.1, max=0.99)
        else:
            frequency = self.frequency
            damping = self.damping
        
        if self.trainable_ls:
            length_scale = torch.clamp(F.softplus(self.length_scale), min=0.1)
        else:
            length_scale = self.length_scale

        if self.amp:
            if not self.scale:
                raw_amp = torch.clamp(self.amp_param, max=8)
            else:
                raw_amp = self.amp_param
            amp = torch.exp(raw_amp)
            # amp = self.amp_param**2
        else:
            amp = None
        
        frame, trajectory = batch
        batch_size, time, px, py = frame.shape

        device = frame.device
        
        vae_mean, vae_log_var = self.encoder(frame.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        if mode == "test" and self.test_t_limit is not None:
            test_t_limit = int(self.test_t_limit * time)
            time_points = torch.arange(test_t_limit, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, test_t_limit) for i in range(batch_size)], 0) 
        else: 
            time_points = torch.arange(time, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0) 
            
        test_time_points = torch.arange(time, device=device) * self.dt
        batches_time_test = torch.cat([test_time_points.view(1, time) for i in range(batch_size)], 0)     

        # compute posterior mean and variance with the gaussian process
        if self.scale:
            s = torch.exp(self.latent_log_scale)
            b = self.latent_bias
            vae_mean  = s * vae_mean + b
            vae_var = (s**2) * vae_var

        if mode == "test" and self.test_t_limit is not None:
            gp_mean, gp_var, gp_likelihood = gaussian_process_ode(batches_time, vae_mean[:,:test_t_limit,0], vae_var[:,:test_t_limit,0], batches_time_test, length_scale=length_scale, nat_freq=frequency, damp_ratio=damping, amp=amp)
        else:
            gp_mean, gp_var, gp_likelihood = gaussian_process_ode(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time_test, length_scale=length_scale, nat_freq=frequency, damp_ratio=damping, amp=amp)

        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)  
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_video_logits, frame, reduction='sum') / batch_size

        loss = reconstruction_loss + kld

        if self.scale and self.amp:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/amp": amp,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/frequency": frequency,
                f"{mode}/damping": damping,
                f"{mode}/length_scale": length_scale
    
            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        else:
            self.log_dict({
                    f"{mode}/recon": reconstruction_loss,
                    f"{mode}/kld":kld,
                    f"{mode}/loss": loss,
                    f"{mode}/elbo": -loss,
                    f"{mode}/frequency": frequency,
                    f"{mode}/damping": damping,
                    f"{mode}/length_scale": length_scale
                }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        
        return loss, (vae_mean, vae_var, kld, reconstruction_loss, gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping, length_scale)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)
        
    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer
        
###########################################################################################################
class FingerRBFVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, length_scale, trainable_ls, amp, scale):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt
        self.trainable_ls = trainable_ls
        self.amp = amp
        self.scale = scale

        self.test_t_limit = None

        self.hparams.update({
            "width":  encoder.px,
            "height": encoder.py,
            "embed_dim": encoder.embed_dim,
            "latent_dim": encoder.latent_dim,
        })

        if self.trainable_ls:
            self.length_scale = nn.Parameter((torch.tensor(length_scale, dtype=torch.double)))
        else:
            self.length_scale = torch.tensor(length_scale, dtype=torch.double)

        if self.amp:
            self.amp_param = nn.Parameter(torch.zeros((), dtype=torch.double))

        if self.scale:
            self.latent_log_scale = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))
            self.latent_bias = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))

    def _step(self, mode, batch):
        min_val = 0.1
        eps = 1e-3

        if self.trainable_ls:
            length_scale = F.softplus(self.length_scale) + min_val
        else:
            length_scale = self.length_scale

        if self.amp:
            amp = torch.exp(self.amp_param)
        else:
            amp = None
        
        frame, trajectory = batch
        batch_size, time, px, py = frame.shape

        device = frame.device
        vae_mean, vae_log_var = self.encoder(frame.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        if mode == "test" and self.test_t_limit is not None:
            test_t_limit = int(self.test_t_limit * time)
            time_points = torch.arange(test_t_limit, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, test_t_limit) for i in range(batch_size)], 0) 
        else: 
            time_points = torch.arange(time, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0) 
            
        test_time_points = torch.arange(time, device=device) * self.dt
        batches_time_test = torch.cat([test_time_points.view(1, time) for i in range(batch_size)], 0) 
        
        # compute posterior mean and variance with the gaussian process
        if self.scale:
            s = torch.exp(self.latent_log_scale)
            b = self.latent_bias
            vae_mean  = s * vae_mean + b
            vae_var = (s**2) * vae_var

        if mode == "test" and self.test_t_limit is not None:
            gp_mean, gp_var, gp_likelihood = gaussian_process_rbf(batches_time, vae_mean[:,:test_t_limit,0], vae_var[:,:test_t_limit,0], batches_time_test, length_scale=length_scale, amp=amp)
        else:
            gp_mean, gp_var, gp_likelihood = gaussian_process_rbf(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time_test, length_scale=length_scale, amp=amp)


        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)  
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_video_logits, frame, reduction='sum') / batch_size

        loss = reconstruction_loss + kld

        if self.scale and self.amp:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/amp": amp,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/length_scale": length_scale

            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        else:
            self.log_dict({
                    f"{mode}/recon": reconstruction_loss,
                    f"{mode}/kld":kld,
                    f"{mode}/loss": loss,
                    f"{mode}/elbo": -loss,
                    f"{mode}/length_scale": length_scale
                }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        
        return loss, (vae_mean, vae_var, kld, reconstruction_loss, gp_mean, gp_var, latent_samples, reconstructed_video, None, None, length_scale)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)
        
    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer
        
###########################################################################################################
class FingerVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt

    def _step(self, mode, batch):        
        frame, trajectory = batch
        batch_size, time, py, px = frame.shape
        device = frame.device
        vae_mean, vae_log_var = self.encoder(frame.double())
        
        std = torch.exp(0.5 * vae_log_var)
        eps = torch.randn_like(std)
        z = vae_mean + eps * std 
        
        logits = self.decoder(z)
        recon_video = torch.sigmoid(logits)
        recon_loss = nn.BCEWithLogitsLoss(reduction="sum")(logits, frame) / batch_size

        kld = -0.5 * torch.sum((1 + vae_log_var - vae_mean.pow(2) - vae_log_var.exp()), dim=(1,2)).mean()

        #compute total loss
        loss = recon_loss + kld

        # log loss
        elbo = -loss

        self.log_dict({
            f"{mode}/recon": recon_loss,
            f"{mode}/kld": kld,
            f"{mode}/loss": loss,
            f"{mode}/elbo": elbo
        }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
    
        return loss, (None, None, kld, recon_loss, vae_mean, torch.exp(vae_log_var), z, recon_video, None, None, None)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer


###########################################################################################################
class PendulumPEGPVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, freq, damp, length_scale, trainable_params, trainable_ls, amp, scale):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt
        self.trainable_params = trainable_params
        self.trainable_ls = trainable_ls
        self.amp = amp
        self.scale = scale
        
        self.test_t_limit = None

        self.hparams.update({
            "width":  encoder.px,
            "height": encoder.py,
            "embed_dim": encoder.embed_dim,
            "latent_dim": encoder.latent_dim,
        })

        if self.trainable_params:
            self.frequency = nn.Parameter((torch.tensor(freq, dtype=torch.double)))
            self.damping = nn.Parameter((torch.tensor(damp, dtype=torch.double)))
        else:
            self.frequency = torch.tensor(3.13, dtype=torch.double)
            self.damping = torch.tensor(0.07, dtype=torch.double)

        if self.trainable_ls:
            self.length_scale = nn.Parameter((torch.tensor(length_scale, dtype=torch.double)))
        else:
            self.length_scale = torch.tensor(1.0, dtype=torch.double)

        if self.amp:
            # self.amp_param = nn.Parameter(torch.zeros((), dtype=torch.double))
            self.amp_param = nn.Parameter(torch.ones((), dtype=torch.double))

        if self.scale:
            self.latent_log_scale = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))
            self.latent_bias = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))

    def _step(self, mode, batch):
        if self.trainable_params:
            frequency = torch.clamp(F.softplus(self.frequency), min=1)
            damping = torch.clamp(torch.sigmoid(self.damping), min=0.05, max=0.99)
            # damping = torch.sigmoid(self.damping)

        else:
            frequency = self.frequency
            damping = self.damping
        
        if self.trainable_ls:
            length_scale = torch.clamp(F.softplus(self.length_scale), min=0.1)
        else:
            length_scale = self.length_scale

        if self.amp:
            if not self.scale:
                raw_amp = torch.clamp(self.amp_param, max=8)
            else:
                raw_amp = self.amp_param
            # amp = torch.exp(raw_amp)
            amp = self.amp_param**2
        else:
            amp = None
        
        frame, trajectory = batch
        batch_size, time, px, py = frame.shape

        device = frame.device
        vae_mean, vae_log_var = self.encoder(frame.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        if mode == "test" and self.test_t_limit is not None:
            test_t_limit = int(self.test_t_limit * time)
            time_points = torch.arange(test_t_limit, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, test_t_limit) for i in range(batch_size)], 0) 
        else: 
            time_points = torch.arange(time, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0) 
            
        test_time_points = torch.arange(time, device=device) * self.dt
        batches_time_test = torch.cat([test_time_points.view(1, time) for i in range(batch_size)], 0) 
        
        # compute posterior mean and variance with the gaussian process
        if self.scale:
            s = torch.exp(self.latent_log_scale)
            b = self.latent_bias
            vae_mean  = s * vae_mean + b
            vae_var = (s**2) * vae_var

        if mode == "test" and self.test_t_limit is not None:
            gp_mean, gp_var, gp_likelihood = gaussian_process_ode(batches_time, vae_mean[:,:test_t_limit,0], vae_var[:,:test_t_limit,0], batches_time_test, length_scale=length_scale, nat_freq=frequency, damp_ratio=damping, amp=amp)
        else:
            gp_mean, gp_var, gp_likelihood = gaussian_process_ode(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time_test, length_scale=length_scale, nat_freq=frequency, damp_ratio=damping, amp=amp)
        
        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)  
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_video_logits, frame, reduction='sum') / batch_size

        loss = reconstruction_loss + kld

        if self.scale and self.amp:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/amp": amp,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/frequency": frequency,
                f"{mode}/damping": damping,
                f"{mode}/length_scale": length_scale
    
            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        elif self.scale:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/frequency": frequency,
                f"{mode}/damping": damping,
                f"{mode}/length_scale": length_scale

            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        else:
            self.log_dict({
                    f"{mode}/recon": reconstruction_loss,
                    f"{mode}/kld":kld,
                    f"{mode}/loss": loss,
                    f"{mode}/elbo": -loss,
                    f"{mode}/frequency": frequency,
                    f"{mode}/damping": damping,
                    f"{mode}/length_scale": length_scale
                }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        
        return loss, (trajectory, kld, reconstruction_loss, gp_mean, gp_var, latent_samples, reconstructed_video, frequency, damping, length_scale)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)
        
    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer
        
###########################################################################################################
class PendulumRBFVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt, length_scale, trainable_ls, amp, scale):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "decoder"])
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt
        self.trainable_ls = trainable_ls
        self.amp = amp
        self.scale = scale

        self.test_t_limit = None

        self.hparams.update({
            "width":  encoder.px,
            "height": encoder.py,
            "embed_dim": encoder.embed_dim,
            "latent_dim": encoder.latent_dim,
        })

        if self.trainable_ls:
            self.length_scale = nn.Parameter((torch.tensor(length_scale, dtype=torch.double)))
        else:
            self.length_scale = torch.tensor(length_scale, dtype=torch.double)

        if self.amp:
            self.amp_param = nn.Parameter(torch.zeros((), dtype=torch.double))

        if self.scale:
            self.latent_log_scale = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))
            self.latent_bias = nn.Parameter(torch.zeros(1,1,1, dtype=torch.double))

    def _step(self, mode, batch):
        min_val = 0.1
        eps = 1e-3

        if self.trainable_ls:
            length_scale = F.softplus(self.length_scale) + min_val
        else:
            length_scale = self.length_scale

        if self.amp:
            amp = torch.exp(self.amp_param)
        else:
            amp = None
        
        frame, trajectory = batch
        batch_size, time, px, py = frame.shape

        device = frame.device
        vae_mean, vae_log_var = self.encoder(frame.double())
        vae_var = torch.exp(vae_log_var)

        # create time for gp
        if mode == "test" and self.test_t_limit is not None:
            test_t_limit = int(self.test_t_limit * time)
            time_points = torch.arange(test_t_limit, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, test_t_limit) for i in range(batch_size)], 0) 
        else: 
            time_points = torch.arange(time, device=device) * self.dt
            batches_time = torch.cat([time_points.view(1, time) for i in range(batch_size)], 0) 
            
        test_time_points = torch.arange(time, device=device) * self.dt
        batches_time_test = torch.cat([test_time_points.view(1, time) for i in range(batch_size)], 0) 
        
        # compute posterior mean and variance with the gaussian process
        if self.scale:
            s = torch.exp(self.latent_log_scale)
            b = self.latent_bias
            vae_mean  = s * vae_mean + b
            vae_var = (s**2) * vae_var

        if mode == "test" and self.test_t_limit is not None:
            gp_mean, gp_var, gp_likelihood = gaussian_process_rbf(batches_time, vae_mean[:,:test_t_limit,0], vae_var[:,:test_t_limit,0], batches_time_test, length_scale=length_scale, amp=amp)
        else:
            gp_mean, gp_var, gp_likelihood = gaussian_process_rbf(batches_time, vae_mean[:,:,0], vae_var[:,:,0], batches_time_test, length_scale=length_scale, amp=amp)

        gp_mean = torch.unsqueeze(gp_mean, 2)
        gp_var = torch.unsqueeze(gp_var, 2)
        
        # compute loss function of vae
        # compute cross entropy between encoder and gp distribution predictions
        cross_entropy = torch.sum(gauss_cross_entropy(gp_mean, gp_var, vae_mean, vae_var), (1,2))
        # compute KL divergence bewteen prior distribution of gp and predicted distribution of vae
        kld = torch.mean(cross_entropy - gp_likelihood)  
        
        # compute reconstruction loss terms    
        epsilon = torch.randn((batch_size, time, 1), device=device)
        latent_samples = gp_mean + epsilon * torch.sqrt(gp_var)
        
        reconstructed_video_logits = self.decoder(latent_samples)
        reconstructed_video = torch.sigmoid(reconstructed_video_logits)
        reconstruction_loss = F.binary_cross_entropy_with_logits(reconstructed_video_logits, frame, reduction='sum') / batch_size

        loss = reconstruction_loss + kld

        if self.scale and self.amp:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/amp": amp,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/length_scale": length_scale

            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        elif self.scale:
            self.log_dict({
                f"{mode}/recon": reconstruction_loss,
                f"{mode}/kld": kld,
                f"{mode}/loss": loss,
                f"{mode}/elbo": -loss,
                f"{mode}/s": s,
                f"{mode}/b": b,
                f"{mode}/length_scale": length_scale

            }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        else:
            self.log_dict({
                    f"{mode}/recon": reconstruction_loss,
                    f"{mode}/kld":kld,
                    f"{mode}/loss": loss,
                    f"{mode}/elbo": -loss,
                    f"{mode}/length_scale": length_scale
                }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))
        
        return loss, (trajectory, kld, reconstruction_loss, gp_mean, gp_var, latent_samples, reconstructed_video, None, None, length_scale)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)
        
    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        return optimizer
        
###########################################################################################################
class PendulumVAEModel(L.LightningModule):
    def __init__(self, encoder, decoder, dt):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.dt = dt

    def _step(self, mode, batch):        
        frame, trajectory = batch
        batch_size, time, py, px = frame.shape
        device = frame.device
        vae_mean, vae_log_var = self.encoder(frame.double())
        
        std = torch.exp(0.5 * vae_log_var)
        eps = torch.randn_like(std)
        z = vae_mean + eps * std 
        
        logits = self.decoder(z)
        recon_video = torch.sigmoid(logits)
        recon_loss = nn.BCEWithLogitsLoss(reduction="sum")(logits, frame) / batch_size

        kld = -0.5 * torch.sum((1 + vae_log_var - vae_mean.pow(2) - vae_log_var.exp()), dim=(1,2)).mean()

        #compute total loss
        loss = recon_loss + kld

        # log loss
        elbo = -loss

        self.log_dict({
            f"{mode}/recon": recon_loss,
            f"{mode}/kld": kld,
            f"{mode}/loss": loss,
            f"{mode}/elbo": elbo
        }, on_epoch=True, on_step=False, prog_bar=(mode=="training"))

        return loss, (trajectory, kld, recon_loss, vae_mean, torch.exp(vae_log_var), z, recon_video, None, None, None)

    def training_step(self, batch):
        return self._step("training", batch)[0]

    def validation_step(self, batch):
        return self._step("validation", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer