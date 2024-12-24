# Nando Metzger

import torch
from torch import nn

class DiffuseBase(nn.Module):
    def __init__(
            self, 
            Niter=8000,
    ):
        super().__init__()
        self.Niter = Niter
        self.logk = torch.log(torch.tensor(0.03))

    def forward(self, sample, train=False, deps=0.1):
        guide, initial = sample['guide'], sample['initial']

        # assert that all values are positive, otherwise shift depth map to positives
        if initial.min()<=deps:
            print("Warning: The forward function was called with negative depth values. Values were temporarly shifted. Consider using unnormalized depth values for stability.")
            initial += deps
            shifted = True
        else:
            shifted = False

        # Execute diffusion
        y_pred, aux = self.diffuse(initial.clone(), guide.clone(), K=torch.exp(self.logk))

        # revert the shift
        if shifted:
            y_pred -= deps

        return {**{'y_pred': y_pred}, **aux}

    def diffuse(self, depth, guide, l=0.24, K=0.01):
        _,_,h,w = guide.shape
        
        # Convert the features to coefficients with the Perona-Malik edge-detection function
        cv, ch = c(guide, K=K)

        # Iterations without gradient
        for t in range(self.Niter):                     
            depth = diffuse_step(cv, ch, depth, l=l)

        return depth, {"cv": cv, "ch": ch}

@torch.jit.script
def diffuse_step(cv, ch, I, l: float=0.24):
    # Anisotropic Diffusion implmentation, Eq. (1) in paper.

    # calculate diffusion update as increments
    dv = I[:,:,1:,:] - I[:,:,:-1,:]
    dh = I[:,:,:,1:] - I[:,:,:,:-1]
    
    tv = l * cv * dv # vertical transmissions
    I[:,:,1:,:] -= tv
    I[:,:,:-1,:] += tv 

    th = l * ch * dh # horizontal transmissions
    I[:,:,:,1:] -= th
    I[:,:,:,:-1] += th 
    
    return I
  
# @torch.jit.script
def c(I, K: float=0.03):
    # apply function to both dimensions
    cv = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,1:,:] - I[:,:,:-1,:]), 1), 1), K)
    ch = g(torch.unsqueeze(torch.mean(torch.abs(I[:,:,:,1:] - I[:,:,:,:-1]), 1), 1), K)
    return cv, ch

# @torch.jit.script
def g(x, K: float=0.03):
    # Perona-Malik edge detection
    return 1.0 / (1.0 + (torch.abs((x*x))/(K*K)))



def _test_diffuse():
    """
    Minimal test function to verify the diffusion step.
    """
    # Create a fake input (batch=1, channel=1/3, height=64, width=64)
    guide = torch.rand(1, 3, 64, 64)
    initial = torch.rand(1, 1, 64, 64) + 1.0

    sample = {
        'guide': guide,
        'initial': initial,
    }

    # Instantiate the model
    model = DiffuseBase(Niter=100)

    # Forward pass
    output = model(sample)
    print("Output keys:", output.keys())
    print("Predicted shape:", output['y_pred'].shape)

if __name__ == "__main__":
    _test_diffuse()
