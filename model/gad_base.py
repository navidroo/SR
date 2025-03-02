from random import randrange
from re import I

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.fft

INPUT_DIM = 4
FEATURE_DIM = 64

class FFTFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        # Convert to frequency domain
        fft_features = torch.fft.rfft2(x)
        
        # Get magnitude and phase
        magnitude = torch.log(torch.abs(fft_features) + 1e-10)
        phase = torch.angle(fft_features)
        
        # Concatenate magnitude and phase
        freq_features = torch.cat([magnitude, phase], dim=1)
        
        # Process frequency features
        return self.freq_conv(freq_features)

def create_high_freq_mask(shape, threshold=0.5):
    """Create a mask that emphasizes high frequencies"""
    rows, cols = shape[-2:]
    crow, ccol = rows // 2, cols // 2
    
    # Create a coordinate grid
    y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols))
    
    # Calculate distance from center
    d = torch.sqrt((y - crow)**2 + (x - ccol)**2)
    
    # Create high-pass filter mask
    mask = (d > threshold * crow).float()
    return mask.cuda()

def apply_freq_diffusion(fft_features, sigma=0.1):
    """Apply Gaussian filtering in frequency domain"""
    magnitude = torch.abs(fft_features)
    phase = torch.angle(fft_features)
    
    # Apply Gaussian smoothing to magnitude
    smoothed_magnitude = F.gaussian_blur(magnitude, kernel_size=3, sigma=sigma)
    
    # Reconstruct complex numbers
    return smoothed_magnitude * torch.exp(1j * phase)

class GADBase(nn.Module):
    def __init__(
            self, feature_extractor='Unet',
            Npre=8000, Ntrain=1024, 
    ):
        super().__init__()

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
 
        if feature_extractor=='none': 
            # RGB verion of DADA does not need a deep feature extractor
            self.feature_extractor = None
            self.fft_branch = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            # Learned version of DADA with FFT enhancement
            self.feature_extractor = smp.Unet('resnet50', classes=FEATURE_DIM//2, in_channels=INPUT_DIM)
            self.fft_branch = FFTFeatureExtractor(in_channels=INPUT_DIM, out_channels=FEATURE_DIM//2)
            self.feature_fusion = nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 1)
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

    def forward(self, sample, train=False, deps=0.1):
        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # assert that all values are positive, otherwise shift depth map to positives
        if source.min()<=deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        # revert the shift
        if shifted:
            y_pred -= deps

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape

        # Define Downsampling operations that depend on the input size
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Enhanced feature extraction with FFT
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1)
        else:
            # Get spatial features
            spatial_feats = self.feature_extractor(
                torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1)
            )
            
            # Get frequency features
            freq_feats = self.fft_branch(
                torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1)
            )
            
            # Combine features
            guide_feats = self.feature_fusion(
                torch.cat([spatial_feats, freq_feats], 1)
            )
        
        # Convert the features to coefficients with enhanced edge detection
        cv, ch = c(guide_feats, K=K)

        # Iterations without gradient
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                for t in range(Npre):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        # Iterations with gradient
        if self.Ntrain>0: 
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)

        return img, {"cv": cv, "ch": ch}

def c(I, K: float=0.03):
    # Add frequency domain analysis for edge detection
    fft_features = torch.fft.rfft2(I)
    magnitude = torch.abs(fft_features)
    high_freq_mask = create_high_freq_mask(magnitude.shape)
    edge_response = torch.fft.irfft2(fft_features * high_freq_mask.unsqueeze(0).unsqueeze(0))
    
    # Combine with existing edge detection
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]) + edge_response[:,:,1:,:], 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]) + edge_response[:,:,:,1:], 1), 1), K)
    return cv, ch

def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))

def diffuse_step(cv, ch, I, l: float=0.24):
    # Add frequency domain diffusion
    fft_I = torch.fft.rfft2(I)
    
    # Apply diffusion in frequency domain
    freq_diffusion = apply_freq_diffusion(fft_I)
    I_freq = torch.fft.irfft2(freq_diffusion)
    
    # Combine with spatial diffusion
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    # Weight between frequency and spatial domain results
    alpha = 0.7  # adjustable parameter
    I = alpha * I + (1-alpha) * I_freq
    
    # Continue with existing diffusion
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Implementation of the adjustment step. Eq (3) in paper.

    # Iss = subsample img
    img_ss = downsample(img)

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    ratio_ss[mask_inv] = 1

    # R = NN upsample r
    ratio = upsample(ratio_ss)

    return img * ratio 
