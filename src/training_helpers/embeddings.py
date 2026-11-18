import torch
import torch.nn as nn
import numpy as np
from einops import repeat

def fourier_encoding(x, L):
    out = [x]  # Include the raw input as the first feature
    for j in range(L):
        out.append(torch.sin(2 ** j * torch.pi * x))
        out.append(torch.cos(2 ** j * torch.pi * x))
    return torch.cat(out, dim=1)

class BARFMasks():
    def __init__(self, start_barf_iter, end_barf_iter, L, device):
        self.start_barf_iter = start_barf_iter
        self.end_barf_iter = end_barf_iter
        self.L = L
        self.device = device

    def get_barf_mask(self, current_iter, L):
        spatial_dim = 3 # both rotation and translation have 3 dimensions
        # coarse-to-fine: smoothly mask positional encoding for BARF
        alpha = (current_iter- self.start_barf_iter)/(self.end_barf_iter-self.start_barf_iter)*self.L       
        k = torch.arange(L,dtype=torch.float32)
        mask_value = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2     # following BARF paper implementation, but I seldom see this way of writing arithmetic, 
        mask_value = repeat(mask_value, 'n -> (n d)', d=spatial_dim*2)
        mask_value = torch.cat((torch.ones(spatial_dim), mask_value))
        return mask_value.to(self.device)
    
    def get_current_spectrum(self, current_iter, L):
        alpha = (current_iter - self.start_barf_iter) / (self.end_barf_iter - self.start_barf_iter) * self.L
        spectrum = np.clip(alpha, 0, L)  # Clip between 0 and L
        return spectrum

class SurrogateFourierFeature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, L):
        """
        `Flat` means that the gradient term is not amplified by the (2**j*pi) term
        Forward pass: Compute Fourier features for input x.
        Backward pass: Compute gradient WITHOUT the (2**j*pi) term
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            L (int): Number of frequency levels for Fourier features
            
        Returns:
            torch.Tensor: Concatenated Fourier features of shape (batch_size, input_dim * (1 + 2*L))
        """
        ctx.save_for_backward(x, torch.tensor(L, device=x.device))  # Save x and L for the backward pass
        out = [x]  # Include the raw input as the first feature
        for j in range(L):
            out.append(torch.sin(2 ** j * torch.pi * x))
            out.append(torch.cos(2 ** j * torch.pi * x))
        return torch.cat(out, dim=1)

    ### NOTE: the following code is equivalent to the lower commented backward() method, slightly slower but more readable
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute surrogate gradient, excluding the (2**j * torch.pi) amplification for sine terms.
        
        Args:
            ctx: Context object containing saved tensors
            grad_output (torch.Tensor): Gradient of the loss w.r.t. the output, shape (batch_size, output_dim)
            
        Returns:
            torch.Tensor: Gradient w.r.t. x, shape (batch_size, input_dim)
            None: No gradient w.r.t. L (scalar, not learnable)
        """
        x, L = ctx.saved_tensors  # Retrieve saved x and L
        L = L.item()  # Convert L to a Python integer

        # Initialize gradient for x (same shape as input x)
        # _, input_dim = x.shape
        input_dim = 3
        grad_x = torch.zeros_like(x)
        grad_x += grad_output[:, :input_dim]  # Gradient for the raw x term

        # Process gradients for Fourier features (skip first input_dim for raw x)
        feature_idx = input_dim  # Start after the raw x term
        for j in range(L):
            # Gradient for sin(2**j * pi * x): normally (2**j * pi) * cos(2**j * pi * x)
            # Surrogate gradient: only use cos(2**j * pi * x), excluding (2**j * pi)
            sin_grad = torch.cos(2**j * torch.pi * x)
            grad_x += grad_output[:, feature_idx:feature_idx + input_dim] * sin_grad

            # Move to the next feature (cos term)
            feature_idx += input_dim

            # Gradient for cos(2**j * pi * x): normally -(2**j * pi) * sin(2**j * pi * x)
            # Surrogate gradient: only use -sin(2**j * pi * x), excluding (2**j * pi)
            cos_grad = -torch.sin(2**j * torch.pi * x)
            grad_x += grad_output[:, feature_idx:feature_idx + input_dim] * cos_grad

            # Move to the next feature
            feature_idx += input_dim
        return grad_x, None  # Return gradient for x, None for L (not learnable)


    # @staticmethod
    # def backward(ctx, grad_output):
    #     """
    #     Backward pass: Compute surrogate gradient, excluding the (2**j * torch.pi) amplification for sine terms.
        
    #     Args:
    #         ctx: Context object containing saved tensors
    #         grad_output (torch.Tensor): Gradient of the loss w.r.t. the output, shape (batch_size, output_dim)
            
    #     Returns:
    #         torch.Tensor: Gradient w.r.t. x, shape (batch_size, input_dim)
    #         None: No gradient w.r.t. L (scalar, not learnable)
    #     """
    #     # NOTE: not using einops library here to avoid gradient errors
    #     x, L = ctx.saved_tensors  # Retrieve saved x and L
    #     L = L.item()  # Convert L to a Python integer

    #     # Initialize gradient for x (same shape as input x)
    #     batch_size, input_dim = x.shape
    #     grad_x = torch.zeros_like(x)

    #     ### Original gradient for the raw input (first feature)
    #     grad_x += grad_output[:, :input_dim]  # Gradient for the raw x term

    #     ### Vectorized computation for Fourier features (skip first input_dim for raw x)
    #     # Create a tensor for all powers of 2 up to L
    #     j_values = torch.arange(L, device=x.device)  # Shape: (L,)
    #     powers = 2 ** j_values  # Shape: (L,)

    #     # Expand x and compute all sin/cos terms at once
    #     x_expanded = x.unsqueeze(-1)  # Shape: (batch_size, input_dim, 1)
    #     powers_expanded = powers.view(1, 1, L)  # Shape: (1, 1, L)
    #     args = powers_expanded * torch.pi * x_expanded  # Shape: (batch_size, input_dim, L)

    #     # Compute all sin and cos gradients at once
    #     sin_grad = torch.cos(args)  # Shape: (batch_size, input_dim, L)
    #     cos_grad = -torch.sin(args)  # Shape: (batch_size, input_dim, L)

    #     # Reshape grad_output for Fourier features (skip raw x term)
    #     fourier_grad_output = grad_output[:, input_dim:]  # Shape: (batch_size, 2 * L * input_dim)
    #     fourier_grad_output_reshaped = fourier_grad_output.view(batch_size, 2 * L, input_dim)  # Shape: (batch_size, 2*L, input_dim)

    #     # Split into sin and cos contributions
    #     sin_grad_output = fourier_grad_output_reshaped[:, ::2, :]  # Shape: (batch_size, L, input_dim)
    #     cos_grad_output = fourier_grad_output_reshaped[:, 1::2, :]  # Shape: (batch_size, L, input_dim)

    #     # Compute contributions to grad_x using einsum for efficiency
    #     sin_contribution = torch.einsum('bil,bli->bi', sin_grad, sin_grad_output)  # Shape: (batch_size, input_dim)
    #     cos_contribution = torch.einsum('bil,bli->bi', cos_grad, cos_grad_output)  # Shape: (batch_size, input_dim)

    #     # Add Fourier contributions to grad_x
    #     grad_x += sin_contribution + cos_contribution
    #     return grad_x, None  # Return gradient for x, None for L (not learnable)

