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
    
    @torch.cuda.amp.autocast(enabled=False)  # Disable AMP for FFT operations
    def forward(self, x):
        # Input validation
        if not validate_tensor(x, "FFTFeatureExtractor input"):
            raise ValueError("Invalid input tensor")

        # Convert to frequency domain
        x = x.to(torch.float32)  # Ensure float32 for FFT
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
        with torch.cuda.amp.autocast():  # Re-enable AMP for convolutions
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
    # Memory optimization: clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for complex operations
        # Process real and imaginary parts separately to save memory
        real_part = fft_features.real
        # Clear fft_features as we no longer need it
        imag_part = fft_features.imag
        del fft_features
        
        # Process real part
        smoothed_real = gaussian_blur(real_part, kernel_size=3, sigma=sigma)
        del real_part
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Process imaginary part
        smoothed_imag = gaussian_blur(imag_part, kernel_size=3, sigma=sigma)
        del imag_part
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reconstruct complex tensor
        result = torch.complex(smoothed_real, smoothed_imag)
        del smoothed_real, smoothed_imag
        
        # Validate only magnitude to save memory
        with torch.no_grad():
            magnitude = torch.abs(result)
            is_valid = not (torch.isnan(magnitude).any() or torch.isinf(magnitude).any())
            del magnitude
            
            if not is_valid:
                logger.warning("Frequency diffusion output may be unstable")
        
        return result

class GADBase(nn.Module):
    def __init__(
            self, feature_extractor='Unet',
            Npre=8000, Ntrain=1024, 
            pre_patience=5, train_patience=3,
            min_delta=1e-4
    ):
        super().__init__()
        logger.info(f"Initializing GADBase with {feature_extractor} extractor")

        self.feature_extractor_name = feature_extractor    
        self.Npre = Npre
        self.Ntrain = Ntrain
        self.pre_patience = pre_patience
        self.train_patience = train_patience
        self.min_delta = min_delta
 
        if feature_extractor=='none': 
            logger.info("Using RGB version without feature extractor")
            self.feature_extractor = None
            self.fft_branch = None
            self.Ntrain = 0
            self.logk = torch.log(torch.tensor(0.03))

        elif feature_extractor=='UNet':
            logger.info("Using UNet with FFT enhancement")
            self.feature_extractor = smp.Unet('resnet50', classes=FEATURE_DIM//2, in_channels=INPUT_DIM)
            # Enable gradient checkpointing for memory efficiency
            self.feature_extractor.encoder.requires_grad_(True)
            self.feature_extractor.encoder.train()
            for module in self.feature_extractor.encoder.modules():
                if isinstance(module, nn.Module):
                    module.requires_grad_(True)
                    if hasattr(module, 'checkpoint') and callable(module.checkpoint):
                        module.checkpoint = True
                        
            self.fft_branch = FFTFeatureExtractor(in_channels=INPUT_DIM, out_channels=FEATURE_DIM//2)
            self.feature_fusion = nn.Conv2d(FEATURE_DIM, FEATURE_DIM, 1)
            self.logk = torch.nn.Parameter(torch.log(torch.tensor(0.03)))

        else:
            raise NotImplementedError(f'Feature extractor {feature_extractor}')

    @torch.cuda.amp.autocast()
    def forward(self, sample, train=False, deps=0.1):
        # Memory optimization: clear cache at the start
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

        # Memory optimization: clear cache before returning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, img, guide, source, mask_inv,
        l=0.24, K=0.01, verbose=False, eps=1e-8, train=False):

        _,_,h,w = guide.shape
        _,_,sh,sw = source.shape
        logger.debug(f"Processing image of size {h}x{w} with source {sh}x{sw}")

        # Define Downsampling operations
        downsample = nn.AdaptiveAvgPool2d((sh, sw))
        upsample = lambda x: F.interpolate(x, (h, w), mode='nearest')

        # Helper function to compute loss
        def compute_loss(pred_img):
            with torch.no_grad():
                try:
                    # Log input tensor stats with more detail
                    logger.info("=== Starting Loss Computation ===")
                    logger.info(f"pred_img stats:")
                    logger.info(f"- shape: {pred_img.shape}")
                    logger.info(f"- dtype: {pred_img.dtype}")
                    logger.info(f"- device: {pred_img.device}")
                    logger.info(f"- min: {pred_img.min().item():.8f}")
                    logger.info(f"- max: {pred_img.max().item():.8f}")
                    logger.info(f"- mean: {pred_img.mean().item():.8f}")
                    logger.info(f"- unique values (first 5): {torch.unique(pred_img)[:5].tolist()}")
                    
                    # Log source tensor stats
                    logger.info(f"\nsource stats:")
                    logger.info(f"- shape: {source.shape}")
                    logger.info(f"- dtype: {source.dtype}")
                    logger.info(f"- device: {source.device}")
                    logger.info(f"- min: {source.min().item():.8f}")
                    logger.info(f"- max: {source.max().item():.8f}")
                    logger.info(f"- mean: {source.mean().item():.8f}")
                    logger.info(f"- unique values (first 5): {torch.unique(source)[:5].tolist()}")
                    
                    # Log mask stats
                    logger.info(f"\nmask_inv stats:")
                    logger.info(f"- shape: {mask_inv.shape}")
                    logger.info(f"- dtype: {mask_inv.dtype}")
                    logger.info(f"- device: {mask_inv.device}")
                    logger.info(f"- invalid pixel count: {mask_inv.sum().item()}")
                    logger.info(f"- total pixels: {mask_inv.numel()}")
                    logger.info(f"- invalid pixel percentage: {(mask_inv.float().mean().item()*100):.2f}%")
                    
                    # Downsample prediction and log
                    logger.info("\nDownsampling prediction...")
                    pred_lr = downsample(pred_img)
                    logger.info(f"pred_lr after downsample:")
                    logger.info(f"- shape: {pred_lr.shape}")
                    logger.info(f"- dtype: {pred_lr.dtype}")
                    logger.info(f"- device: {pred_lr.device}")
                    logger.info(f"- min: {pred_lr.min().item():.8f}")
                    logger.info(f"- max: {pred_lr.max().item():.8f}")
                    logger.info(f"- mean: {pred_lr.mean().item():.8f}")
                    logger.info(f"- unique values (first 5): {torch.unique(pred_lr)[:5].tolist()}")
                    
                    # Match device and dtype
                    pred_lr = pred_lr.to(source.device, source.dtype)
                    logger.info(f"\npred_lr after device/dtype match:")
                    logger.info(f"- dtype: {pred_lr.dtype}")
                    logger.info(f"- device: {pred_lr.device}")
                    
                    # Calculate absolute difference
                    logger.info("\nCalculating absolute difference...")
                    diff = torch.abs(pred_lr - source)
                    logger.info(f"diff stats:")
                    logger.info(f"- min: {diff.min().item():.8f}")
                    logger.info(f"- max: {diff.max().item():.8f}")
                    logger.info(f"- mean: {diff.mean().item():.8f}")
                    logger.info(f"- sum: {diff.sum().item():.8f}")
                    logger.info(f"- unique values (first 5): {torch.unique(diff)[:5].tolist()}")
                    
                    # Compute mean over valid regions
                    valid_mask = ~mask_inv
                    valid_pixels = valid_mask.sum().item()
                    logger.info(f"\nValid pixels: {valid_pixels} out of {mask_inv.numel()}")
                    
                    if valid_pixels > 0:
                        masked_diff = diff * valid_mask.float()
                        sum_diff = masked_diff.sum().item()
                        loss = sum_diff / valid_pixels
                        logger.info(f"Masked difference stats:")
                        logger.info(f"- sum: {sum_diff:.8f}")
                        logger.info(f"- valid pixels: {valid_pixels}")
                        logger.info(f"- final loss: {loss:.8f}")
                    else:
                        loss = diff.mean().item()
                        logger.info(f"No valid pixels, using full mean: {loss:.8f}")
                    
                    if loss == 0:
                        logger.error("\nWARNING: Loss is exactly zero!")
                        logger.error("Detailed investigation:")
                        logger.error(f"1. pred_lr non-zero elements: {(pred_lr != 0).sum().item()}")
                        logger.error(f"2. source non-zero elements: {(source != 0).sum().item()}")
                        logger.error(f"3. diff non-zero elements: {(diff != 0).sum().item()}")
                        logger.error(f"4. masked_diff non-zero elements: {(masked_diff != 0).sum().item() if valid_pixels > 0 else 'N/A'}")
                    
                    logger.info("=== End Loss Computation ===\n")
                    return loss
                    
                except Exception as e:
                    logger.error(f"Error in loss computation: {str(e)}")
                    logger.error("Stack trace:", exc_info=True)
                    return float('inf')

        # Feature extraction with memory optimization
        if self.feature_extractor is None: 
            guide_feats = torch.cat([guide, img], 1)
        else:
            # Normalize input
            normalized_input = torch.cat([guide, img-img.mean((1,2,3), keepdim=True)], 1)
            
            # Get spatial features with memory optimization
            with torch.cuda.amp.autocast():
                spatial_feats = self.feature_extractor(normalized_input)
            validate_tensor(spatial_feats, "Spatial features")
            
            # Get frequency features
            with torch.cuda.amp.autocast():
                freq_feats = self.fft_branch(normalized_input)
            validate_tensor(freq_feats, "Frequency features")
            
            # Clear unnecessary tensors
            del normalized_input
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Combine features
            guide_feats = self.feature_fusion(
                torch.cat([spatial_feats, freq_feats], 1)
            )
            validate_tensor(guide_feats, "Combined features")
            
            # Clear intermediate results
            del spatial_feats, freq_feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Edge detection
        cv, ch = c(guide_feats, K=K)
        validate_tensor(cv, "Vertical coefficients")
        validate_tensor(ch, "Horizontal coefficients")
        
        del guide_feats
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Initialize early stopping for diffusion iterations
        early_stopper = EarlyStopping(patience=self.pre_patience, min_delta=self.min_delta)
        prev_loss = None

        # Diffusion iterations with memory optimization
        if self.Npre>0: 
            with torch.no_grad():
                Npre = randrange(self.Npre) if train else self.Npre
                logger.debug(f"Running {Npre} pre-iterations")
                for t in range(Npre):
                    with torch.cuda.amp.autocast():
                        img = diffuse_step(cv, ch, img, l=l)
                        img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
                    
                    # Check for convergence every 100 iterations
                    if t % 100 == 0:
                        current_loss = compute_loss(img)
                        logger.info(f"Pre-iteration {t}, Loss: {current_loss:.6f}")
                        
                        if prev_loss is not None:
                            if early_stopper(current_loss):
                                logger.info(f"Early stopping at pre-iteration {t} due to convergence. Final loss: {current_loss:.6f}")
                                break
                        prev_loss = current_loss
                        validate_tensor(img, f"Pre-iteration {t}")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

        # Reset early stopping for training iterations
        early_stopper = EarlyStopping(patience=self.train_patience, min_delta=self.min_delta)
        prev_loss = None

        if self.Ntrain>0: 
            logger.debug(f"Running {self.Ntrain} training iterations")
            for t in range(self.Ntrain): 
                with torch.cuda.amp.autocast():
                    img = diffuse_step(cv, ch, img, l=l)
                    img = adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8)
                
                # Check for convergence every 50 iterations
                if t % 50 == 0:
                    current_loss = compute_loss(img)
                    logger.info(f"Training iteration {t}, Loss: {current_loss:.6f}")
                    
                    if prev_loss is not None:
                        if early_stopper(current_loss):
                            logger.info(f"Early stopping at training iteration {t} due to convergence. Final loss: {current_loss:.6f}")
                            break
                    prev_loss = current_loss
                    validate_tensor(img, f"Train iteration {t}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        return img, {"cv": cv, "ch": ch}

def c(I, K: float=0.03):
    # Validate input
    if not validate_tensor(I, "Edge detection input"):
        raise ValueError("Invalid input to edge detection")

    # Add frequency domain analysis for edge detection
    with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for FFT
        I = I.to(torch.float32)  # Ensure float32
        fft_features = torch.fft.fft2(I, norm='ortho')
        magnitude = torch.abs(fft_features)
        high_freq_mask = create_high_freq_mask(I.shape)
        
        # Apply mask and transform back
        masked_fft = fft_features * high_freq_mask.unsqueeze(0).unsqueeze(0)
        edge_response = torch.fft.ifft2(masked_fft, norm='ortho')
        edge_response = torch.real(edge_response)
    
    validate_tensor(edge_response, "Edge response")
    
    # Combine with existing edge detection
    with torch.cuda.amp.autocast():  # Re-enable AMP for the rest
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
    # Memory optimization: clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Validate inputs without storing validation tensors
    with torch.no_grad():
        for name, tensor in [("cv", cv), ("ch", ch), ("I", I)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                raise ValueError(f"Invalid {name} in diffuse step")

    # Memory efficient implementation
    with torch.cuda.amp.autocast(enabled=False):  # Disable AMP for FFT operations
        # Add frequency domain diffusion
        I = I.to(torch.float32)  # Ensure float32
        fft_I = torch.fft.fft2(I, norm='ortho')
        
        # Apply diffusion in frequency domain
        freq_diffusion = apply_freq_diffusion(fft_I)
        del fft_I
        
        I_freq = torch.real(torch.fft.ifft2(freq_diffusion, norm='ortho'))
        del freq_diffusion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Re-enable AMP for the rest of operations
    with torch.cuda.amp.autocast():
        # Compute spatial gradients
        dv = I[:,:,1:,:] - I[:,:,:-1,:]
        dh = I[:,:,:,1:] - I[:,:,:,:-1]
        
        # Weight between frequency and spatial domain results
        alpha = 0.7  # adjustable parameter
        I = alpha * I + (1-alpha) * I_freq
        del I_freq
        
        # Apply transmissions with memory optimization
        tv = l * cv * dv  # vertical transmissions
        del cv, dv
        I[:,:,1:,:] -= tv
        I[:,:,:-1,:] += tv
        del tv
        
        th = l * ch * dh  # horizontal transmissions
        del ch, dh
        I[:,:,:,1:] -= th
        I[:,:,:,:-1] += th
        del th
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Quick validation without storing additional tensors
        with torch.no_grad():
            if torch.isnan(I).any() or torch.isinf(I).any():
                logger.error("Invalid values in diffuse step output")
        
        return I

def adjust_step(img, source, mask_inv, upsample, downsample, eps=1e-8):
    # Memory optimization: clear cache at start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Quick validation without storing validation tensors
    with torch.no_grad():
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise ValueError("Invalid input image in adjust step")
        if torch.isnan(source).any() or torch.isinf(source).any():
            raise ValueError("Invalid source in adjust step")

    # Compute downsampled image
    img_ss = downsample(img)
    
    # Compute ratio with memory optimization
    ratio_ss = torch.empty_like(source)
    ratio_ss = torch.where(mask_inv, torch.ones_like(source), source / (img_ss + eps))
    del img_ss
    
    # Compute final result with memory optimization
    ratio = upsample(ratio_ss)
    del ratio_ss
    
    result = img * ratio
    del ratio
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Quick validation of output
    with torch.no_grad():
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.error("Invalid values in adjust step output")
    
    return result 

class EarlyStopping:
    """Early stopping to prevent overfitting and detect convergence"""
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.logger = logging.getLogger(__name__)

    def __call__(self, val_loss):
        if not isinstance(val_loss, (int, float)):
            self.logger.warning(f"Invalid loss value type: {type(val_loss)}")
            return False
            
        if val_loss < 0:
            self.logger.warning(f"Negative loss value: {val_loss}")
            return False

        # Scale the values to handle very small numbers better
        scaled_val_loss = val_loss * 1e6  # Scale up by 1 million
        scaled_best_loss = self.best_loss * 1e6 if self.best_loss != float('inf') else float('inf')
        scaled_min_delta = self.min_delta * 1e6

        # First call or better loss
        if self.best_loss == float('inf') or scaled_val_loss < scaled_best_loss - scaled_min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                self.logger.info(f'Loss improved to {val_loss:.8f}')
        else:
            self.counter += 1
            if self.verbose:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience} (current: {val_loss:.8f}, best: {self.best_loss:.8f})')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop 
