import os
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from skimage.transform import resize
from skimage.restoration import denoise_nl_means, estimate_sigma
from nilearn.masking import compute_brain_mask
from scipy.ndimage import zoom
import glob
import torch

INPUT_MODALITY = 't1c'
TARGET_MODALITY = 't2w'
NEW_SHAPE = (128, 160, 128)  # ajustar se quiser preservar resolução original
VOXEL_SIZE = (1.0, 1.0, 1.0)  # mm³ padrão BraTS
DATASET_DIR = '}datasets/brats/training_data'  # pasta onde estão as subpastas dos pacientes
SAVE_PATH = 'dataset_t1c_to_t2w.npz'

print("GPU disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nome da GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo usado:", device)

torch.cuda.empty_cache()

def load_nifti(path):

    """Carrega e reorienta uma imagem NIfTI para o espaço canônico (RAS+).
    Garante que todas as imagens tenham a mesma orientação espacial (alinhadas em relação aos mesmos eixos do cérebro)"""

    img = nib.load(path)
    img = nib.as_closest_canonical(img) #RAS = right, anterior, superior
    return img

def resample_image(img, voxel_size=VOXEL_SIZE):
    """Padroniza o tamanho dos voxels em 1mm³. Isso garante que todas as imagens fiquem no mesmo espaço e escala, 
    o que é essencial para comparações, registros e aprendizado de máquina."""
    return resample_to_output(img, voxel_sizes=voxel_size)


def nlmeans_and_normalize(data, patch_size=3, patch_distance=5, h_factor=1.0):
    """
    Aplica denoising Non-Local Means (NLM) e normaliza a imagem.
    
    Parâmetros:
        data: array numpy (3D MRI)
        patch_size: tamanho do patch para comparar similaridades (ex: 3x3x3)
        patch_distance: distância máxima para busca de patches similares
        h_factor: fator de suavização (controla o peso do filtro)
    """
    
    # Estima o nível de ruído na imagem
    sigma_est = np.mean(estimate_sigma(data, channel_axis=None))
    
    # Aplica Non-Local Means (NLM)
    denoised = denoise_nl_means(
        data,
        h=h_factor * sigma_est,       # controla o quanto suaviza
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,            # MRI não tem canais RGB
        fast_mode=True
    )
    
    # Normaliza a intensidade com z-score
    normed = (denoised - np.mean(denoised)) / (np.std(denoised) + 1e-8)
    return normed


def skull_strip(data):
    """
    Gera uma máscara do cérebro usando nilearn.
    Retorna a máscara binária (True = cérebro) e a imagem mascarada.
    """
    temp_img = nib.Nifti1Image(data, affine=np.eye(4)) # objeto Nifti1Image temporário, necessário para nilearn
    
    # Calcula e aplica máscara do cérebro
    brain_mask = compute_brain_mask(temp_img).get_fdata().astype(bool)
    masked_data = np.where(brain_mask, data, 0)

    return masked_data, brain_mask

def crop_background_with_padding(data, pad=5):
    """
    Recorta a imagem baseado na máscara do cérebro, mas adiciona padding seguro.
    Retorna o volume recortado, máscara recortada e coordenadas.
    
    pad: número de voxels extras a adicionar em cada direção
    """ 
    masked_data, brain_mask = skull_strip(data)
    coords = np.array(np.nonzero(brain_mask))
    if coords.size == 0:
        return masked_data, None, None  # máscara vazia
    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1)
    
    # Aplicar padding
    minz, miny, minx = max(minz - pad, 0), max(miny - pad, 0), max(minx - pad, 0)
    maxz, maxy, maxx = min(maxz + pad, data.shape[0]), min(maxy + pad, data.shape[1]), min(maxx + pad, data.shape[2])
    
    cropped = masked_data[minz:maxz, miny:maxy, minx:maxx]
    cropped_mask = brain_mask[minz:maxz, miny:maxy, minx:maxx]
    
    return cropped, cropped_mask, (minz, maxz, miny, maxy, minx, maxx)


def crop_background_with_mask(data):
    """
    Recorta a imagem baseado na máscara do cérebro.
    Retorna o volume recortado e as coordenadas para reconstrução.
    """
    masked_data, brain_mask = skull_strip(data)
    coords = np.array(np.nonzero(brain_mask))
    if coords.size == 0:
        return masked_data, None  # máscara vazia
    
    minz, miny, minx = coords.min(axis=1)
    maxz, maxy, maxx = coords.max(axis=1)
    
    cropped = masked_data[minz:maxz, miny:maxy, minx:maxx]
    cropped_mask = brain_mask[minz:maxz, miny:maxy, minx:maxx]
    
    return cropped, cropped_mask, (minz, maxz, miny, maxy, minx, maxx)

def resize_volume(img, new_shape=NEW_SHAPE):
    """Redimensiona o volume 3D para new_shape sem cortar,
    ajustando proporcionalmente cada eixo via interpolação linear."""
    factors = [n / s for n, s in zip(new_shape, img.shape)]
    return zoom(img, factors, order=3)

def preprocess_patient(patient_dir):
    """
    Carrega t1c (entrada) e t2w (alvo) de um paciente,
    aplica o pipeline completo e retorna volumes prontos.
    """
    pid = os.path.basename(patient_dir)
    t1c_path = os.path.join(patient_dir, f"{pid}-{INPUT_MODALITY}.nii.gz")
    t2w_path = os.path.join(patient_dir, f"{pid}-{TARGET_MODALITY}.nii.gz")

    # Load + Reorient
    img_in = load_nifti(t1c_path)
    img_out = load_nifti(t2w_path)

    # Resample para voxel uniforme
    img_in = resample_image(img_in)
    img_out = resample_image(img_out)

    data_in = img_in.get_fdata()
    data_out = img_out.get_fdata()

    # Background removing (com coordenadas iguais)
    cropped_in, _, bbox = crop_background_with_mask(data_in)
    if bbox is None:
        return None, None
    minz, maxz, miny, maxy, minx, maxx = bbox
    masked_data_out, _ = skull_strip(data_out)
    cropped_out = masked_data_out[minz:maxz, miny:maxy, minx:maxx]

    # NLM + Normalização
    cropped_in = nlmeans_and_normalize(data_in)
    cropped_out = nlmeans_and_normalize(data_out)

    # Resize para shape fixo
    cropped_in = resize_volume(cropped_in)
    cropped_out = resize_volume(cropped_out)

    return cropped_in, cropped_out

X_list, Y_list = [], []

patient_dirs = sorted(glob.glob(os.path.join("datasets/brats/training_data/BraTS-GLI-*")))
print(f"Encontrados {len(patient_dirs)} pacientes.") 

for i, p_dir in enumerate(patient_dirs):
    print(f"[{i+1}/{len(patient_dirs)}] Processando {p_dir}...")
    X, Y = preprocess_patient(p_dir)
    if X is not None:
        X_list.append(X)
        Y_list.append(Y)

X_list = [x.astype(np.float32) for x in X_list]
Y_list = [y.astype(np.float32) for y in Y_list]

X_all = np.stack(X_list, axis=0)
Y_all = np.stack(Y_list, axis=0)

print("Shape final:")
print("X:", X_all.shape, "Y:", Y_all.shape)

# Converter para arrays numpy
X_all = np.array(X_all)
Y_all = np.array(Y_all)

print("Shape após augmentation:")
print("X:", X_all.shape)
print("Y:", Y_all.shape)

# Salvar dataset aumentado
np.savez_compressed(SAVE_PATH, X=X_all, Y=Y_all)
print(f"Dataset salvo em {SAVE_PATH}")