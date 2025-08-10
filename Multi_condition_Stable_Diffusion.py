

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:20:49 2025

@author: RASULEVLAB
"""

import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from diffusers import UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from torch.cuda.amp import autocast, GradScaler
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class CTMRINiftiDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        patient_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        print ('patient_dirs' , patient_dirs)
        
        for patient_dir in patient_dirs:
            ct_path = os.path.join(patient_dir, "ct.nii.gz")
            mri_path = os.path.join(patient_dir, "mr.nii.gz")

            print("CT:", ct_path, "| MRI:", mri_path)
            
            if os.path.isfile(ct_path) and os.path.isfile(mri_path):
                print(" Found:", ct_path, "|", mri_path)
                self.samples.append((ct_path, mri_path))
            else:
                print(" Missing files in:", patient_dir)
                
            
            # if os.path.exists(ct_path) and os.path.exists(mri_path):
            #     self.samples.append((ct_path, mri_path))
        
        print(f"Total valid samples found: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ct_path, mri_path = self.samples[idx]

        ct_img = nib.load(ct_path).get_fdata()
        mri_img = nib.load(mri_path).get_fdata()

        # Use the middle slice (you can customize this logic)
        ct_slice = ct_img[:, :, ct_img.shape[2] // 2]
        mri_slice = mri_img[:, :, mri_img.shape[2] // 2]
        # Stack grayscale to RGB
        ct_slice = np.stack([ct_slice] * 3, axis=-1)
        mri_slice = np.stack([mri_slice] * 3, axis=-1)

        ct_tensor = self.transform(ct_slice.astype(np.float32))
        mri_tensor = self.transform(mri_slice.astype(np.float32))

        return ct_tensor, mri_tensor, ct_path  # return ct_path for saving
    


        
    
    

def load_prompts(prompt_file):
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts if prompts else [""]



def train(train_dataset, vae, unet, text_encoder, tokenizer, prompts, scaler, optimizer):
    unet.train()
    dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for epoch in range(5):
        for batch_idx, (ct_img, mri_img, _) in enumerate(dataloader):
            ct_img, mri_img = ct_img.cuda(), mri_img.cuda()

            with torch.no_grad():
                mri_latent = vae.encode(mri_img).latent_dist.sample() * 0.18215

            prompt = [prompts[batch_idx % len(prompts)]] * ct_img.size(0)
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.cuda())[0]

            noise = torch.randn_like(mri_latent)
            timesteps = torch.randint(0, 1000, (mri_latent.size(0),), device="cuda").long()
            noisy_latents = mri_latent + noise

            optimizer.zero_grad()
            with autocast():
                pred_noise = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeddings).sample
                loss = torch.nn.functional.mse_loss(pred_noise, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            torch.cuda.empty_cache()

            print(f"Epoch {epoch+1} Batch {batch_idx+1}: Loss = {loss.item():.4f}")



def inference(test_dataset, vae, unet, text_encoder, tokenizer, prompts, output_dir="Generative_Nii"):
    unet.eval()
    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    os.makedirs(output_dir, exist_ok=True)

    ssim_scores = []
    psnr_scores = []

    for i, (ct_img, mri_img, ct_path) in enumerate(dataloader):
        ct_img = ct_img.cuda()
        mri_img = mri_img.cuda()

        with torch.no_grad():
            # Encode CT image to latent space
            latent = vae.encode(ct_img).latent_dist.sample() * 0.18215

            # Add noise (simulate diffusion timestep)
            noise = torch.randn_like(latent)
            timestep = torch.randint(0, 1000, (1,), device="cuda").long()
            noisy_latents = latent + noise

            # Handle prompt text
            prompt = [prompts[i % len(prompts)] if prompts else ""]  # fallback empty
            text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")
            text_embeddings = text_encoder(text_inputs.input_ids.cuda())[0]

            with autocast():
                pred_noise = unet(noisy_latents, timestep, encoder_hidden_states=text_embeddings).sample
                denoised_latent = noisy_latents - pred_noise
                decoded = vae.decode(denoised_latent / 0.18215).sample
                decoded = (decoded.clamp(-1, 1) + 1) / 2

        # Convert to NumPy
        gen_img = decoded.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        real_img = mri_img.squeeze().detach().cpu().numpy().transpose(1, 2, 0)

        # Convert to grayscale if necessary
        if gen_img.shape[-1] > 1:
            gen_img = gen_img[..., 0]
            real_img = real_img[..., 0]

        # Normalize to [0, 1] for metrics
        gen_img = np.clip(gen_img, 0, 1)
        real_img = np.clip(real_img, 0, 1)

        # Calculate metrics
        ssim_val = ssim(real_img, gen_img, data_range=1.0)
        psnr_val = psnr(real_img, gen_img, data_range=1.0)
        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

        # Save to .nii.gz
        output_nii = nib.Nifti1Image((gen_img * 255).astype(np.uint8), affine=np.eye(4))
        patient_id = os.path.basename(os.path.dirname(ct_path[0]))
        nib.save(output_nii, os.path.join(output_dir, f"{patient_id}_gen_mri.nii.gz"))

        torch.cuda.empty_cache()

    print(f"\n Inference complete on {len(test_dataset)} samples.")
    print(f" Average SSIM: {np.mean(ssim_scores):.4f}")
    print(f" Average PSNR: {np.mean(psnr_scores):.2f} dB")

    return np.mean(ssim_scores), np.mean(psnr_scores)


if __name__ == "__main__":
    
    dataset = CTMRINiftiDataset(root_dir= r"/mmfs1/projects/bakhtiyor.rasulev/Rahil/Deep_Learning/Stable_Diffusion/Dataset_SynthRAD2023/Task1/brain")
    print(f"Loaded {len(dataset)} samples")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").to("cuda")
    unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet").to("cuda")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")

    for p in vae.parameters(): p.requires_grad = False
    for p in text_encoder.parameters(): p.requires_grad = False
    vae.eval()
    text_encoder.eval()

    prompts = load_prompts("prompts.txt")
    optimizer = torch.optim.Adam(unet.parameters(), lr=1e-5)
    scaler = GradScaler()
    train(train_dataset,  vae, unet, text_encoder, tokenizer, prompts, scaler, optimizer)
    inference(val_dataset, vae, unet, text_encoder, tokenizer, prompts)
