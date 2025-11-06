import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from lpips import LPIPS
from torchmetrics.image.fid import FrechetInceptionDistance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lpips_metric = LPIPS(net='alex').to(device)
fid_metric = FrechetInceptionDistance(feature=64).to(device)

def evaluate_model(generator, val_loader):
    generator.eval()
    total_L1, total_SSIM, total_PSNR, total_LPIPS = [], [], [], []
    
    with torch.no_grad():
        for i, (t1, t2) in enumerate(val_loader):
            t1, t2 = t1.to(device), t2.to(device)
            fake_t2 = generator(t1)

            # Normaliza para [0,1]
            fake_t2_norm = (fake_t2 - fake_t2.min()) / (fake_t2.max() - fake_t2.min() + 1e-8)
            t2_norm = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)

            # L1 LOSS
            L1 = F.l1_loss(fake_t2_norm, t2_norm).item()
            total_L1.append(L1)

            # SSIM e PSNR
            # Calcula slice a slice (já que são 3D volumes)
            ssim_slices, psnr_slices = [], []
            fake_np = fake_t2_norm.squeeze().cpu().numpy()
            real_np = t2_norm.squeeze().cpu().numpy()
            for z in range(fake_np.shape[2]):
                ssim_slices.append(ssim(real_np[:,:,z], fake_np[:,:,z], data_range=1.0))
                psnr_slices.append(psnr(real_np[:,:,z], fake_np[:,:,z], data_range=1.0))
            total_SSIM.append(np.mean(ssim_slices))
            total_PSNR.append(np.mean(psnr_slices))

            # LPIPS
            # Para LPIPS, usamos uma fatia central representativa
            z_mid = fake_t2_norm.shape[2] // 2
            lpips_value = lpips_metric(
                fake_t2_norm[:, :, :, :, z_mid],
                t2_norm[:, :, :, :, z_mid]
            ).mean().item()
            total_LPIPS.append(lpips_value)

            # FID
            # Converte para 2D slices RGB-like (repete canal para 3)
            fake_rgb = fake_t2_norm[:, :, :, :, z_mid].repeat(1, 3, 1, 1)
            real_rgb = t2_norm[:, :, :, :, z_mid].repeat(1, 3, 1, 1)
            fid_metric.update(fake_rgb, real=False)
            fid_metric.update(real_rgb, real=True)

    # Finaliza FID
    fid_score = fid_metric.compute().item()

    # Agrega resultados
    results = {
        "L1": np.mean(total_L1),
        "SSIM": np.mean(total_SSIM),
        "PSNR": np.mean(total_PSNR),
        "LPIPS": np.mean(total_LPIPS),
        "FID": fid_score
    }

    return results