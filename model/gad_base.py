from random import randrange
from re import I
import logging
import numpy as np
import math

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import torch.fft

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DIM = 4
FEATURE_DIM = 64

def create_gaussian_kernel(kernel_size=3, sigma=1.0, channels=1):
    """Create a Gaussian kernel for blurring"""
    # Create a 1D Gaussian kernel
    x = torch.linspace(-kernel_size//2, kernel_size//2, kernel_size)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    
    # Create 2D kernel by outer product
    kernel_2d = gauss.view(-1, 1) * gauss.view(1, -1)
    
    # Expand to match PyTorch's convolution format: (out_channels, in_channels/groups, height, width)
    kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
    
    return kernel

def gaussian_blur(x, kernel_size=3, sigma=1.0):
    """Apply Gaussian blur to a tensor"""
    if not isinstance(kernel_size, int):
        kernel_size = 3
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma, x.size(1)).to(x.device)
    
    # Apply padding to maintain size
    padding = kernel_size // 2
    
    # Apply convolution for each channel independently
    x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    return F.conv2d(x_padded, kernel, groups=x.size(1))

def validate_tensor(tensor, name, log_stats=True):
    """Validate tensor for common issues"""
    if torch.isnan(tensor).any():
        logger.error(f"{name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"{name} contains Inf values")
        return False
    if log_stats:
        logger.debug(f"{name} stats - min: {tensor.min():.4f}, max: {tensor.max():.4f}, mean: {tensor.mean():.4f}")
    return True

class FFTFeatureExtractor(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.freq_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
    
    def forward(self, x):
        # Input validation
        if not validate_tensor(x, "FFTFeatureExtractor input"):
            raise ValueError("Invalid input tensor")

        # Convert to frequency domain
        fft_features = torch.fft.fft2(x, norm='ortho')
        
        # Get magnitude and phase
        magnitude = torch.log(torch.abs(fft_features) + 1e-10)
        phase = torch.angle(fft_features)
        
        # Validate intermediate results
        validate_tensor(magnitude, "FFT magnitude")
        validate_tensor(phase, "FFT phase")
        
        # Both magnitude and phase are real tensors
        freq_features = torch.cat([magnitude, phase], dim=1)
        
        # Process and validate output
        output = self.freq_conv(freq_features)
        if not validate_tensor(output, "FFTFeatureExtractor output"):
            logger.warning("FFTFeatureExtractor output contains invalid values")
        
        return output

def create_high_freq_mask(shape, threshold=0.5):
    """Create a mask that emphasizes high frequencies"""
    b, c, h, w = shape
    
    # Create a coordinate grid
    y = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w)
    x = torch.linspace(-1, 1, w).view(1, -1).repeat(h, 1)
    
    # Calculate distance from center
    d = torch.sqrt(x*x + y*y)
    
    # Create high-pass filter mask
    mask = (d > threshold).float()
    logger.debug(f"High-freq mask coverage: {mask.mean():.2%}")
    return mask.cuda()

def apply_freq_diffusion(fft_features, sigma=0.1):
    """Apply Gaussian filtering in frequency domain"""
    # Validate input
    if not torch.is_complex(fft_features):
        logger.error("Input to freq_diffusion must be complex")
        raise ValueError("Input must be complex tensor")
    
    # Split into real and imaginary parts
    real_part = fft_features.real
    imag_part = fft_features.imag
    
    # Validate parts
    validate_tensor(real_part, "Freq diffusion real part")
    validate_tensor(imag_part, "Freq diffusion imag part")
    
    # Apply custom Gaussian smoothing to both parts
    smoothed_real = gaussian_blur(real_part, kernel_size=3, sigma=sigma)
    smoothed_imag = gaussian_blur(imag_part, kernel_size=3, sigma=sigma)
    
    # Reconstruct and validate complex tensor
    result = torch.complex(smoothed_real, smoothed_imag)
    if not validate_tensor(torch.abs(result), "Freq diffusion output magnitude"):
        logger.warning("Frequency diffusion output may be unstable")
    
    return result

class GADBase(nn.Module):
    def __init__(
            self, feature_extractor='Unet',
            Npre=8000, Ntrain=1024, 
    ):
        super().__init__()
        logger.info(f"Initializing GADBase with {feature_extractor} extractor")

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
 
        if feature_extractor=='none': 
            logger.info("Using RGB version without feature extractor")
            self.feature_extractor = None
            self.fft_branch = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            logger.info("Using UNet with FFT enhancement")
            self.feature_extractor = smp.Unet('resnet50', classes=FEATURE_DIM//2, in_channels=INPUT_DIM)
            self.fft_branch = FFTFeatureExtractor(in_channels=INPUT_DIM, out_channels=FEATURE_DIM//2)
            self.feature_fusion = nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 1)
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

    def forward(self, sample, train=False, deps=0.1):
        # Validate inputs
        for key in ['guide', 'source', 'mask_lr', 'y_bicubic']:
            if key not in sample:
                raise KeyError(f"Missing {key} in input sample")
            validate_tensor(sample[key], f"Input {key}")

        guide, source, mask_lr = sample['guide'], sample['source'], sample['mask_lr']

        # Check for negative depth values
        if source.min()<=deps:
            logger.warning("Negative depth values detected, applying shift")
            source += deps
            sample['y_bicubic'] += deps
            shifted = True
        else:
            shifted = False

        y_pred, aux = self.diffuse(sample['y_bicubic'].clone(), guide.clone(), source, mask_lr < 0.5,
                 K=torch.exp(self.logk), verbose=False, train=train)

        # Validate output
        if not validate_tensor(y_pred, "Model output"):
            logger.error("Invalid values in model output")

        if shifted:
            y_pred -= deps
            logger.debug("Reverted depth shift")

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape
        logger.debug(f"Processing image of size {h}x{w} with source {sh}x{sw}")

        # Define Downsampling operations
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Feature extraction
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1)
        else:
            # Normalize input
            normalized_input = torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1)
            
            # Get spatial features
            spatial_feats = self.feature_extractor(normalized_input)
            validate_tensor(spatial_feats, "Spatial features")
            
            # Get frequency features
            freq_feats = self.fft_branch(normalized_input)
            validate_tensor(freq_feats, "Frequency features")
            
            # Combine features
            guide_feats = self.feature_fusion(
                torch.cat([spatial_feats, freq_feats], 1)
            )
            validate_tensor(guide_feats, "Combined features")
        
        # Edge detection
        cv, ch = c(guide_feats, K=K)
        validate_tensor(cv, "Vertical coefficients")
        validate_tensor(ch, "Horizontal coefficients")

        # Diffusion iterations
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                logger.debug(f"Running {Npre} pre-iterations")
                for t in range(Npre):                     
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
                    if t % 1000 == 0:  # Log periodically
                        validate_tensor(img, f"Pre-iteration {t}")

        if self.Ntrain>0: 
            logger.debug(f"Running {self.Ntrain} training iterations")
            for t in range(self.Ntrain): 
                img = diffuse_step(cv, ch, img, l=l)
                img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
                if t % 100 == 0:  # Log periodically
                    validate_tensor(img, f"Train iteration {t}")

        return img, {"cv": cv, "ch": ch}

def c(I, K: float=0.03):
    # Validate input
    if not validate_tensor(I, "Edge detection input"):
        raise ValueError("Invalid input to edge detection")

    # Add frequency domain analysis for edge detection
    fft_features = torch.fft.fft2(I, norm='ortho')
    magnitude = torch.abs(fft_features)
    high_freq_mask = create_high_freq_mask(I.shape)
    
    # Apply mask and transform back
    masked_fft = fft_features * high_freq_mask.unsqueeze(0).unsqueeze(0)
    edge_response = torch.fft.ifft2(masked_fft, norm='ortho')
    edge_response = torch.real(edge_response)
    
    validate_tensor(edge_response, "Edge response")
    
    # Combine with existing edge detection
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]) + edge_response[:,:,1:,:], 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]) + edge_response[:,:,:,1:], 1), 1), K)
    
    return cv, ch

def g(x, K: float=0.03):
    # Validate input
    validate_tensor(x, "Perona-Malik input")
    
    # Perona-Malik edge detection
    result = 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))
    
    validate_tensor(result, "Perona-Malik output")
    return result

def diffuse_step(cv, ch, I, l: float=0.24):
    # Validate inputs
    for name, tensor in [("cv", cv), ("ch", ch), ("I", I)]:
        if not validate_tensor(tensor, f"Diffuse step {name}"):
            raise ValueError(f"Invalid {name} in diffuse step")

    # Add frequency domain diffusion
    fft_I = torch.fft.fft2(I, norm='ortho')
    
    # Apply diffusion in frequency domain
    freq_diffusion = apply_freq_diffusion(fft_I)
    I_freq = torch.real(torch.fft.ifft2(freq_diffusion, norm='ortho'))
    
    validate_tensor(I_freq, "Frequency domain diffusion result")
    
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
    
    validate_tensor(I, "Diffuse step output")
    return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Validate inputs
    validate_tensor(img, "Adjust step input image")
    validate_tensor(source, "Adjust step source")

    # Iss = subsample img
    img_ss = downsample(img)
    validate_tensor(img_ss, "Downsampled image")

    # Rss = source / Iss
    ratio_ss = source / (img_ss + eps)
    ratio_ss[mask_inv] = 1
    validate_tensor(ratio_ss, "Adjustment ratio")

    # R = NN upsample r
    ratio = upsample(ratio_ss)
    validate_tensor(ratio, "Upsampled ratio")

    result = img * ratio
    validate_tensor(result, "Adjust step output")
    return result 
