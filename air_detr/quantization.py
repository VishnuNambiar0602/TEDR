"""
Quantization engine for AIR-DETR.

Implements:
- Symmetric channel-wise INT8 & INT4 Post-Training Quantization (PTQ)
- MSE Grid Search Calibration for INT8/INT4 (minimizes reconstruction error)
- Activation-aware Weight Quantization (AWQ) channel scaling
- Generalized Post-Training Quantization (GPTQ) Hessian-aware updates
- Vectorized 4-bit packing/unpacking using PyTorch tensor operations
"""

import torch
from typing import Tuple, Dict, Any, Optional

def pack_int4(qweight: torch.Tensor) -> torch.Tensor:
    """Packs two signed 4-bit values (range [-8, 7]) into a single uint8 byte."""
    q_unsigned = (qweight + 8).to(torch.uint8)
    flat = q_unsigned.flatten()
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    flat = flat.view(-1, 2)
    packed = flat[:, 0] | (flat[:, 1] << 4)
    return packed

def unpack_int4(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """Unpacks a packed uint8 tensor back into signed 4-bit values (range [-8, 7])."""
    flat_packed = packed.flatten()
    unpacked = torch.empty(flat_packed.numel() * 2, dtype=torch.int8, device=packed.device)
    unpacked[0::2] = (flat_packed & 0x0F).to(torch.int8) - 8
    unpacked[1::2] = ((flat_packed >> 4) & 0x0F).to(torch.int8) - 8
    total_elements = original_shape.numel()
    return unpacked[:total_elements].view(original_shape)

class QuantizationManager:
    """
    Manages simple, calibrated, AWQ, and GPTQ quantization and dequantization of weights.
    """
    
    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes a weight tensor using simple symmetric channel-wise quantization.
        
        Args:
            tensor: Weight tensor of shape (out_channels, in_channels, ...).
            mode: Quantization mode ('int8', 'int4', or other formats containing them).
            
        Returns:
            Tuple of (quantized_tensor, scale_tensor).
        """
        out_channels = tensor.shape[0]
        flat_tensor = tensor.view(out_channels, -1)
        
        actual_mode = "int8" if "int8" in mode else "int4" if "int4" in mode else mode
        max_q = 127.0 if actual_mode == "int8" else 7.0
        clamp_min = -128 if actual_mode == "int8" else -8
        clamp_max = 127 if actual_mode == "int8" else 7
        target_dtype = torch.int8
        
        best_scales = torch.zeros(out_channels, dtype=tensor.dtype, device=tensor.device)
        q_rows = []
        for c in range(out_channels):
            channel_w = flat_tensor[c]
            max_abs = channel_w.abs().max()
            scale = max_abs / max_q
            scale = max(scale, 1e-8)
            best_scales[c] = scale
            
            q = torch.round(channel_w / scale).clamp(clamp_min, clamp_max).to(target_dtype)
            q_rows.append(q)
            
        qweight = torch.stack(q_rows).view(tensor.shape)
        view_shape = [out_channels] + [1] * (tensor.ndim - 1)
        scale = best_scales.view(view_shape)
        
        if actual_mode == "int4":
            qweight = pack_int4(qweight)
            
        return qweight, scale

    @staticmethod
    def quantize_mse_grid_search(tensor: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes a weight tensor using MSE grid search to find the optimal scale per channel.
        
        Args:
            tensor: Weight tensor of shape (out_channels, in_channels, ...).
            mode: Quantization mode ('int8' or 'int4').
            
        Returns:
            Tuple of (quantized_tensor, scale_tensor).
        """
        out_channels = tensor.shape[0]
        # View tensor as 2D to easily iterate over channels
        flat_tensor = tensor.view(out_channels, -1)
        
        # Max integer values
        max_q = 127.0 if mode == "int8" else 7.0
        clamp_min = -128 if mode == "int8" else -8
        clamp_max = 127 if mode == "int8" else 7
        target_dtype = torch.int8
        
        best_scales = torch.zeros(out_channels, dtype=tensor.dtype, device=tensor.device)
        q_rows = []
        
        for c in range(out_channels):
            channel_w = flat_tensor[c]
            max_abs = channel_w.abs().max()
            if max_abs < 1e-8:
                best_scales[c] = 1e-8
                q_rows.append(torch.zeros_like(channel_w, dtype=target_dtype))
                continue
                
            best_mse = float('inf')
            best_scale = max_abs / max_q
            
            # Grid search: test 10 candidates from 0.5 * max_abs to 1.0 * max_abs
            for pct in range(50, 101, 5):
                clip_threshold = max_abs * (pct / 100.0)
                scale_candidate = clip_threshold / max_q
                scale_candidate = max(scale_candidate, 1e-8)
                
                # Quantize and dequantize
                q = torch.round(channel_w / scale_candidate).clamp(clamp_min, clamp_max)
                w_rec = q * scale_candidate
                
                mse = torch.mean((channel_w - w_rec) ** 2).item()
                if mse < best_mse:
                    best_mse = mse
                    best_scale = scale_candidate
                    
            best_scales[c] = best_scale
            # Perform final quantization with best scale
            q_final = torch.round(channel_w / best_scale).clamp(clamp_min, clamp_max).to(target_dtype)
            q_rows.append(q_final)
            
        qweight = torch.stack(q_rows).view(tensor.shape)
        # Reshape scale to match original dims for broadcasting
        view_shape = [out_channels] + [1] * (tensor.ndim - 1)
        scale = best_scales.view(view_shape)
        
        if mode == "int4":
            qweight = pack_int4(qweight)
            
        return qweight, scale

    @staticmethod
    def apply_awq(
        tensor: torch.Tensor,
        act_scale: torch.Tensor,
        mode: str,
        alpha: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes weight using Activation-aware Weight Quantization (AWQ).
        
        Args:
            tensor: Weight tensor of shape (out_channels, in_channels, ...).
            act_scale: Activation scales of shape (in_channels,).
            mode: Quantization mode ('int8' or 'int4').
            alpha: AWQ scaling hyperparameter.
            
        Returns:
            Tuple of (quantized_tensor, scale_tensor of shape (out_channels, in_channels, ...)).
        """
        # act_scale: shape (in_channels,)
        in_channels = tensor.shape[1]
        
        if act_scale.numel() != in_channels:
            # Fallback to simple MSE search if activation scale shape doesn't match weight in_channels (e.g. grouped conv)
            return QuantizationManager.quantize_mse_grid_search(tensor, mode)
            
        # Clamp activation scales to prevent division by zero
        act_scale = act_scale.clamp(min=1e-8)
        
        # Calculate AWQ scaling factor per input channel
        s_act = act_scale.pow(alpha)
        # Normalize scaling factor
        s = s_act / s_act.max().clamp(min=1e-8)
        
        # Shape scale s for broadcasting: (1, in_channels, 1, 1, ...)
        s_shape = [1, in_channels] + [1] * (tensor.ndim - 2)
        s_broadcast = s.view(s_shape).to(tensor.device).to(tensor.dtype)
        
        # Scale weights up
        scaled_tensor = tensor * s_broadcast
        
        # Quantize scaled tensor using MSE grid search
        # qweight: packed int4 or raw int8, scale_c: shape (out_channels, 1, 1, ...)
        qweight_packed, scale_c = QuantizationManager.quantize_mse_grid_search(scaled_tensor, mode)
        
        # Compute the effective dequantization scale: scale_c / s_broadcast
        # Since s_broadcast is (1, in_channels, 1, ...), scale_eff will be (out_channels, in_channels, 1, ...)
        scale_eff = scale_c / s_broadcast
        
        return qweight_packed, scale_eff

    @staticmethod
    def apply_gptq(
        tensor: torch.Tensor,
        hessian: torch.Tensor,
        mode: str,
        lambda_reg: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes weight using Generalized Post-Training Quantization (GPTQ).
        
        Args:
            tensor: Weight tensor of shape (out_channels, in_channels, ...).
            hessian: Hessian matrix of shape (in_channels, in_channels).
            mode: Quantization mode ('int8' or 'int4').
            lambda_reg: Hessian regularization parameter.
            
        Returns:
            Tuple of (quantized_tensor, scale_tensor).
        """
        out_channels = tensor.shape[0]
        in_channels = tensor.shape[1]
        
        if hessian.shape[0] != in_channels:
            # Fallback to simple MSE search if Hessian shape doesn't match weight in_channels (e.g. grouped conv)
            return QuantizationManager.quantize_mse_grid_search(tensor, mode)
            
        # Reshape weight to 2D: (out_channels, in_channels * K1 * K2)
        # Note: if it's Conv2d, D = in_channels * K1 * K2
        orig_shape = tensor.shape
        flat_w = tensor.view(out_channels, -1).clone()
        D = flat_w.shape[1]
        
        # Make sure Hessian has shape (D, D)
        # If Hessian is (in_channels, in_channels) from calibration, we repeat it to cover kernel size
        if hessian.shape[0] != D:
            # Repeat Hessian to cover spatial dimensions of weight kernel
            k_size = D // in_channels
            # Simple Kronecker-style expansion
            H = hessian.repeat_interleave(k_size, dim=0).repeat_interleave(k_size, dim=1)
        else:
            H = hessian.clone()
            
        H = H.to(tensor.device).to(torch.float64)
        
        # Add regularization to diagonal to ensure invertibility
        reg = lambda_reg * torch.mean(torch.diag(H))
        reg = max(reg.item(), 1e-4)
        H += reg * torch.eye(D, device=tensor.device)
        
        # Invert Hessian
        H_inv = torch.inverse(H).to(tensor.dtype)
        
        # Grid search parameters
        max_q = 127.0 if mode == "int8" else 7.0
        clamp_min = -128 if mode == "int8" else -8
        clamp_max = 127 if mode == "int8" else 7
        target_dtype = torch.int8
        
        # We perform column-wise quantization and updates
        W_quant = flat_w.clone()
        scales = torch.zeros(out_channels, dtype=tensor.dtype, device=tensor.device)
        
        # Pre-calculate first-pass optimal channel-wise scales using MSE Grid Search
        _, initial_scales = QuantizationManager.quantize_mse_grid_search(tensor, mode)
        scales = initial_scales.view(out_channels)
        
        # Column-by-column updates
        for i in range(D):
            col = W_quant[:, i]
            # Quantize column using pre-calculated scale factors
            scale_i = scales
            q_col = torch.round(col / scale_i).clamp(clamp_min, clamp_max)
            dequant_col = q_col * scale_i
            
            # Compute quantization error
            err = col - dequant_col
            
            # Save quantized integers back
            W_quant[:, i] = q_col
            
            # Update remaining columns (columns i+1 to D-1)
            if i < D - 1:
                # Vectorized Optimal Brain Surgeon (OBS) weight update
                h_inv_ii = H_inv[i, i].clamp(min=1e-8)
                step = H_inv[i, i+1:] / h_inv_ii
                # err is (out_channels,), step is (D - 1 - i,)
                # outer product is (out_channels, D - 1 - i)
                W_quant[:, i+1:] -= torch.outer(err, step)
                
        # Reshape quantized weights back
        qweight = W_quant.view(orig_shape).to(target_dtype)
        
        # Reshape scales to match original dims for broadcasting
        view_shape = [out_channels] + [1] * (tensor.ndim - 1)
        scale = scales.view(view_shape)
        
        if mode == "int4":
            qweight = pack_int4(qweight)
            
        return qweight, scale

    @staticmethod
    def dequantize_tensor(q_tensor: torch.Tensor, scale: torch.Tensor, mode: str, original_shape: torch.Size) -> torch.Tensor:
        """Dequantizes a tensor back to float."""
        actual_mode = "int8" if "int8" in mode else "int4" if ("int4" in mode or mode in ("awq", "gptq")) else mode
        if actual_mode == 'int8':
            # Element-wise multiplication supports both channel-wise and matrix scales (AWQ)
            return q_tensor.to(scale.dtype) * scale
        elif actual_mode == 'int4':
            unpacked = unpack_int4(q_tensor, original_shape)
            return unpacked.to(scale.dtype) * scale
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")
