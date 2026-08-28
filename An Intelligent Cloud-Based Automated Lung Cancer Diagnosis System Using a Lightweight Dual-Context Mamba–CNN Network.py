

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional
import copy

# ---------------------------
# 0. Utility Functions
# ---------------------------

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute common classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = mcc_num / mcc_den if mcc_den != 0 else 0.0
    roc_auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'mcc': mcc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }

# ---------------------------
# 1. Synthetic Dataset
# ---------------------------

class SyntheticLungNoduleDataset(Dataset):
    """
    Generates synthetic 3D volumes representing:
    - nodule ROI (local path)
    - context ROI at multiple scales (context path)
    - binary label (0: benign, 1: malignant)
    """
    def __init__(self, num_samples=500, roi_size=(32,32,32), context_scales=[1,2], noise_level=0.1):
        self.num_samples = num_samples
        self.roi_size = roi_size
        self.context_scales = context_scales
        self.noise_level = noise_level

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate base nodule
        nodule = np.random.randn(*self.roi_size) * self.noise_level
        # Add a spherical nodule-like structure
        center = np.array(self.roi_size) // 2
        radius = np.random.randint(5, 12)
        for x in range(self.roi_size[0]):
            for y in range(self.roi_size[1]):
                for z in range(self.roi_size[2]):
                    if np.linalg.norm(np.array([x,y,z]) - center) < radius:
                        nodule[x,y,z] += np.random.uniform(0.5, 1.5)

        # Label: malignant if radius > 8, benign otherwise (simple rule)
        label = 1 if radius > 8 else 0

        # Generate context ROIs at different scales (surrounding tissue)
        contexts = []
        for scale in self.context_scales:
            ctx_size = tuple(s * scale for s in self.roi_size)
            ctx = np.random.randn(*ctx_size) * self.noise_level
            # Add some anatomical structure (e.g., vessels)
            if scale == 2:
                # Simulate vessel as a line
                axis = np.random.randint(0,3)
                for i in range(ctx_size[axis]):
                    idx_slice = [slice(None)]*3
                    idx_slice[axis] = i
                    if np.random.rand() < 0.3:
                        ctx[tuple(idx_slice)] += 0.8
            contexts.append(torch.tensor(ctx, dtype=torch.float32).unsqueeze(0))  # add channel dim

        nodule_tensor = torch.tensor(nodule, dtype=torch.float32).unsqueeze(0)  # (1, D, H, W)
        return nodule_tensor, contexts, torch.tensor(label, dtype=torch.long)

# ---------------------------
# 2. Model Components
# ---------------------------

class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable 3D convolution."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)

class MobileNetV4Block(nn.Module):
    """Simplified MobileNetV4-inspired block."""
    def __init__(self, in_channels, out_channels, expansion_factor=2):
        super().__init__()
        hidden_dim = in_channels * expansion_factor
        self.expand = nn.Conv3d(in_channels, hidden_dim, 1) if expansion_factor > 1 else nn.Identity()
        self.dwconv = nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.project = nn.Conv3d(hidden_dim, out_channels, 1)
        self.act = nn.ReLU6(inplace=True)
        self.use_residual = (in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.project(x)
        if self.use_residual:
            x = x + identity
        return x

class SimplifiedMambaBlock(nn.Module):
    """
    A lightweight block that mimics the selective state-space behaviour of MambaOut-Femto
    using 1D convolutions and gating. This is a simplified stand-in for the actual Mamba
    architecture to keep the script self-contained.
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1d = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.gate = nn.Conv1d(channels, channels, 1)
        self.act = nn.SiLU()

    def forward(self, x):
        # x: (B, C, D, H, W) -> reshape to (B, C, L) with L = D*H*W
        B, C, D, H, W = x.shape
        x_flat = x.view(B, C, -1)
        x_conv = self.conv1d(x_flat)
        gate = torch.sigmoid(self.gate(x_flat))
        x_ssm = x_conv * gate
        x_ssm = self.act(x_ssm)
        # add residual
        x_flat = x_flat + x_ssm
        return x_flat.view(B, C, D, H, W)

class MambaOutFemto(nn.Module):
    """Stack of simplified Mamba blocks (placeholder for MambaOut-Femto)."""
    def __init__(self, channels, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([SimplifiedMambaBlock(channels) for _ in range(num_blocks)])
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class LowRankBilinearFusion(nn.Module):
    """Rank-constrained low-rank bilinear interaction."""
    def __init__(self, local_dim, context_dim, output_dim, rank=8):
        super().__init__()
        self.proj_local = nn.Linear(local_dim, output_dim)
        self.proj_context = nn.Linear(context_dim, output_dim)
        self.U = nn.Parameter(torch.randn(output_dim, rank))
        self.V = nn.Parameter(torch.randn(output_dim, rank))
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, local_feat, context_feat):
        # local_feat: (B, local_dim), context_feat: (B, context_dim)
        local_proj = self.proj_local(local_feat)   # (B, output_dim)
        ctx_proj = self.proj_context(context_feat) # (B, output_dim)
        # Low-rank bilinear: sum_k (local_proj * U[:,k]) * (ctx_proj * V[:,k])
        interaction = torch.zeros_like(local_proj)
        for k in range(self.U.shape[1]):
            interaction = interaction + (local_proj * self.U[:, k]) * (ctx_proj * self.V[:, k])
        return interaction + self.bias

class GhostCompression(nn.Module):
    """Ghost module for lightweight feature compression."""
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        self.intrinsic_channels = out_channels // ratio
        self.primary_conv = nn.Conv3d(in_channels, self.intrinsic_channels, 1)
        self.ghost_conv = nn.Conv3d(self.intrinsic_channels, out_channels - self.intrinsic_channels,
                                    kernel_size=3, padding=1, groups=self.intrinsic_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        intrinsic = self.primary_conv(x)
        intrinsic_act = self.act(intrinsic)
        ghost = self.ghost_conv(intrinsic_act)
        ghost_act = self.act(ghost)
        return torch.cat([intrinsic_act, ghost_act], dim=1)

class CoordinateAttention3D(nn.Module):
    """Coordinate Attention adapted for 3D inputs (factorized spatial attention)."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        self.pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))  # not directly; we'll implement custom
        self.pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        # Height attention: pool over D and W
        x_h = x.mean(dim=2, keepdim=True).mean(dim=4, keepdim=True)  # (B,C,1,H,1)
        x_h = x_h.squeeze(2).squeeze(-1)  # (B,C,H)
        x_h = x_h.unsqueeze(1)  # (B,1,C,H) for conv2d
        x_h = self.conv1(x_h)
        x_h = self.act(x_h)
        x_h = self.conv2(x_h)
        x_h = torch.sigmoid(x_h)  # (B,1,C,H)
        x_h = x_h.view(B, C, 1, H, 1)
        # Width attention
        x_w = x.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)  # (B,C,1,1,W)
        x_w = x_w.squeeze(2).squeeze(2)  # (B,C,W)
        x_w = x_w.unsqueeze(1)  # (B,1,C,W)
        x_w = self.conv1(x_w)
        x_w = self.act(x_w)
        x_w = self.conv2(x_w)
        x_w = torch.sigmoid(x_w)
        x_w = x_w.view(B, C, 1, 1, W)
        # Depth attention
        x_d = x.mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)  # (B,C,D,1,1)
        x_d = x_d.squeeze(3).squeeze(-1)  # (B,C,D)
        x_d = x_d.unsqueeze(1)  # (B,1,C,D)
        x_d = self.conv1(x_d)
        x_d = self.act(x_d)
        x_d = self.conv2(x_d)
        x_d = torch.sigmoid(x_d)
        x_d = x_d.view(B, C, D, 1, 1)
        # Combine
        return x * x_h * x_w * x_d

class ECAAttention(nn.Module):
    """Efficient Channel Attention (ECA)."""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, D, H, W = x.shape
        y = self.avg_pool(x).view(B, C)
        y = y.unsqueeze(1)  # (B,1,C)
        y = self.conv(y)
        y = self.sigmoid(y).view(B, C, 1, 1, 1)
        return x * y

class DirectionalCrossAttention(nn.Module):
    """Nodule-to-context cross-attention (nodule queries, context keys/values)."""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        # query: (B, Lq, dim)  - nodule tokens
        # key_value: (B, Lkv, dim) - context tokens
        B, Lq, _ = query.shape
        Lkv = key_value.shape[1]
        Q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(key_value).view(B, Lkv, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(key_value).view(B, Lkv, self.num_heads, self.head_dim).transpose(1,2)
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)  # (B, num_heads, Lq, head_dim)
        out = out.transpose(1,2).contiguous().view(B, Lq, -1)
        return self.out_proj(out)

class DepthwiseResidualRefinement(nn.Module):
    """Depthwise convolution + residual."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.dwconv = nn.Conv3d(channels, channels, kernel_size, padding=kernel_size//2, groups=channels)
        self.norm = nn.BatchNorm3d(channels)
        self.act = nn.ReLU(inplace=True)
        self.res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.act(x)
        return identity + self.res_scale * x

# ---------------------------
# 3. NCD-MambaLite (Dual-Context Representation Learning)
# ---------------------------

class NCD_MambaLite(nn.Module):
    """
    Dual-path CNN + Mamba representation learning with low-rank bilinear fusion
    and Ghost compression.
    """
    def __init__(self, in_channels=1, base_dim=32, mamba_dim=64, fusion_dim=128, rank=8):
        super().__init__()
        # Local CNN pathway (nodule)
        self.local_dsconv = DepthwiseSeparableConv3d(in_channels, base_dim)
        self.local_mnv4 = MobileNetV4Block(base_dim, base_dim)
        self.local_pool = nn.AdaptiveAvgPool3d(1)  # global pooling -> vector

        # Context Mamba pathway (multi-scale context)
        self.context_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, mamba_dim, kernel_size=3, padding=1),
                MambaOutFemto(mamba_dim, num_blocks=2),
                nn.AdaptiveAvgPool3d(1)
            ) for _ in range(2)  # two context scales
        ])
        self.context_proj = nn.Linear(mamba_dim * 2, fusion_dim)

        # Low-rank bilinear fusion
        self.lr_fusion = LowRankBilinearFusion(local_dim=base_dim, context_dim=fusion_dim, output_dim=fusion_dim, rank=rank)

        # Ghost compression
        self.ghost = GhostCompression(fusion_dim, fusion_dim, ratio=2)

    def forward(self, nodule_roi, context_rois):
        # nodule_roi: (B,1,D,H,W)
        # context_rois: list of (B,1,Dc,Hc,Wc) tensors (multi-scale)
        # Local path
        local_feat = self.local_dsconv(nodule_roi)
        local_feat = self.local_mnv4(local_feat)
        local_vec = self.local_pool(local_feat).flatten(1)  # (B, base_dim)

        # Context path
        ctx_vecs = []
        for i, ctx_roi in enumerate(context_rois):
            ctx = self.context_encoders[i](ctx_roi)  # (B, mamba_dim,1,1,1)
            ctx_vecs.append(ctx.flatten(1))
        ctx_vec = torch.cat(ctx_vecs, dim=1)  # (B, mamba_dim*2)
        ctx_vec = self.context_proj(ctx_vec)  # (B, fusion_dim)

        # Fusion
        fused = self.lr_fusion(local_vec, ctx_vec)  # (B, fusion_dim)
        # Reshape for Ghost (add dummy spatial dims)
        fused = fused.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, fusion_dim,1,1,1)
        compressed = self.ghost(fused)  # (B, fusion_dim,1,1,1)
        compressed = compressed.flatten(1)  # (B, fusion_dim)
        return compressed

# ---------------------------
# 4. NCDAR (Adaptive Refinement Module)
# ---------------------------

class NCDAR(nn.Module):
    """
    Dual-path refinement: spatial-channel attention, directional cross-attention,
    depthwise residual, and complementary integration.
    """
    def __init__(self, local_dim, context_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        self.local_dim = local_dim
        self.context_dim = context_dim

        # Spatial-channel refinement for local and context streams
        self.local_ca = CoordinateAttention3D(local_dim)
        self.local_eca = ECAAttention(local_dim)
        self.context_ca = CoordinateAttention3D(context_dim)
        self.context_eca = ECAAttention(context_dim)

        # Project to token space for cross-attention
        self.local_to_token = nn.Linear(local_dim, hidden_dim)
        self.context_to_token = nn.Linear(context_dim, hidden_dim)

        # Directional cross-attention (nodule -> context)
        self.cross_attn = DirectionalCrossAttention(hidden_dim, num_heads=num_heads)

        # Depthwise residual refinement
        self.dw_res = DepthwiseResidualRefinement(hidden_dim)

        # Final projection for integration
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim + context_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, local_feat, context_feat, low_rank_feat):
        # local_feat: (B, local_dim, D, H, W) or flattened? We'll work with flattened vectors
        # For simplicity, assume local_feat and context_feat are already flattened vectors
        # But the attention modules expect spatial dims. We'll reshape to (B,C,1,1,1)
        local_sp = local_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if local_feat.dim()==2 else local_feat
        ctx_sp = context_feat.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) if context_feat.dim()==2 else context_feat

        # Spatial-channel refinement
        local_ref = self.local_eca(self.local_ca(local_sp)).flatten(1)
        ctx_ref = self.context_eca(self.context_ca(ctx_sp)).flatten(1)

        # Directional cross-attention: nodule (query) attends to context (key/value)
        # Create token sequences (here each token is the whole vector, so L=1)
        local_tokens = local_ref.unsqueeze(1)  # (B,1,local_dim)
        ctx_tokens = ctx_ref.unsqueeze(1)      # (B,1,context_dim)
        # Project to hidden_dim
        local_proj = self.local_to_token(local_tokens)
        ctx_proj = self.context_to_token(ctx_tokens)
        cross_out = self.cross_attn(local_proj, ctx_proj)  # (B,1,hidden_dim)

        # Depthwise residual refinement (reshape for 3D conv)
        cross_sp = cross_out.transpose(1,2).unsqueeze(-1).unsqueeze(-1)  # (B,hidden,1,1,1)
        dw_out = self.dw_res(cross_sp).flatten(1)  # (B,hidden_dim)

        # Complementary integration with low-rank fused representation
        combined = torch.cat([dw_out, low_rank_feat], dim=1)  # (B, hidden_dim + context_dim)
        final = self.final_proj(combined)  # (B, hidden_dim)
        return final

# ---------------------------
# 5. Classification Head
# ---------------------------

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# ---------------------------
# 6. ECT-DiagLite (Explainability & Trust)
# ---------------------------

class ECTDiagLite:
    """
    Explanation-Consistency and Trust-Guided Diagnostic Lightweight module.
    Computes Grad-CAM++ and simplified LIME explanations, consistency score,
    trust score, and selective routing.
    """
    def __init__(self, model, classifier, device='cpu'):
        self.model = model
        self.classifier = classifier
        self.device = device

    def grad_cam_plusplus(self, nodule_roi, context_rois):
        """Compute Grad-CAM++ on the classification output."""
        self.model.eval()
        nodule_roi = nodule_roi.to(self.device)
        context_rois = [c.to(self.device) for c in context_rois]
        nodule_roi.requires_grad = True
        # Forward
        fused = self.model(nodule_roi, context_rois)
        logits = self.classifier(fused)
        # Assume binary classification, take positive class logit
        target = logits[:, 1].sum()
        self.model.zero_grad()
        target.backward()
        # Gradients wrt input
        grads = nodule_roi.grad
        # Weighted combination of gradients and activations (simplified Grad-CAM++)
        # Use mean of gradients as importance map
        cam = grads.abs().mean(dim=1, keepdim=True)  # (B,1,D,H,W)
        cam = F.interpolate(cam, size=nodule_roi.shape[2:], mode='trilinear', align_corners=False)
        cam = cam.squeeze(1).cpu().detach().numpy()
        return cam

    def simplified_lime(self, nodule_roi, context_rois, num_superpixels=8):
        """
        Simplified LIME: divide ROI into superpixels (cubes), perturb by masking,
        fit a linear model to get importance.
        """
        self.model.eval()
        nodule_roi = nodule_roi.to(self.device)
        context_rois = [c.to(self.device) for c in context_rois]
        orig_shape = nodule_roi.shape[2:]
        # Divide into grid of superpixels
        grid = 2  # simple 2x2x2
        D, H, W = orig_shape
        step_d = D // grid
        step_h = H // grid
        step_w = W // grid
        superpixel_masks = []
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    mask = torch.zeros_like(nodule_roi)
                    mask[:, :, i*step_d:(i+1)*step_d, j*step_h:(j+1)*step_h, k*step_w:(k+1)*step_w] = 1
                    superpixel_masks.append(mask)
        # Create perturbed samples
        samples = []
        for sp_mask in superpixel_masks:
            perturbed = nodule_roi * (1 - sp_mask)  # remove superpixel
            samples.append(perturbed)
        samples = torch.cat(samples, dim=0)  # (num_sp, C, D, H, W)
        # Get predictions
        fused = self.model(samples, [c.repeat(samples.shape[0],1,1,1,1) for c in context_rois])
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=1)[:, 1]  # positive class prob
        # Original prediction
        with torch.no_grad():
            fused_orig = self.model(nodule_roi.unsqueeze(0), [c.unsqueeze(0) for c in context_rois])
            logits_orig = self.classifier(fused_orig)
            orig_prob = F.softmax(logits_orig, dim=1)[0, 1].item()
        # Importance = original_prob - perturbed_prob (positive means important)
        importance = orig_prob - probs.cpu().detach().numpy()
        # Map back to ROI
        lime_map = np.zeros(orig_shape)
        idx = 0
        for i in range(grid):
            for j in range(grid):
                for k in range(grid):
                    lime_map[i*step_d:(i+1)*step_d, j*step_h:(j+1)*step_h, k*step_w:(k+1)*step_w] = importance[idx]
                    idx += 1
        return lime_map

    def explanation_consistency(self, cam_map, lime_map):
        """Compute Jensen-Shannon divergence based consistency score."""
        # Flatten and normalize
        cam_flat = cam_map.flatten()
        lime_flat = lime_map.flatten()
        # Ensure non-negative
        cam_flat = np.abs(cam_flat)
        lime_flat = np.abs(lime_flat)
        cam_norm = cam_flat / (cam_flat.sum() + 1e-8)
        lime_norm = lime_flat / (lime_flat.sum() + 1e-8)
        # Average distribution
        avg = (cam_norm + lime_norm) / 2
        # KL divergence
        kl_cam = np.sum(cam_norm * np.log(cam_norm / (avg + 1e-8) + 1e-8))
        kl_lime = np.sum(lime_norm * np.log(lime_norm / (avg + 1e-8) + 1e-8))
        jsd = 0.5 * (kl_cam + kl_lime)
        consistency = 1 - jsd / np.log(2)  # normalized to [0,1]
        return consistency

    def compute_trust(self, prob, uncertainty, consistency, weights=(0.4, 0.3, 0.3)):
        """
        prob: calibrated probability (confidence)
        uncertainty: predictive uncertainty (std dev from MC dropout)
        consistency: explanation consistency score
        """
        # Convert uncertainty to reliability (1 - normalized uncertainty)
        # Assume uncertainty already normalized to [0,1]
        uncertainty_rel = 1 - uncertainty
        trust = weights[0] * prob + weights[1] * uncertainty_rel + weights[2] * consistency
        return trust

# ---------------------------
# 7. Full Model Wrapper
# ---------------------------

class FullModel(nn.Module):
    def __init__(self, in_channels=1, base_dim=32, mamba_dim=64, fusion_dim=128, hidden_dim=128):
        super().__init__()
        self.ncd_mambalite = NCD_MambaLite(in_channels, base_dim, mamba_dim, fusion_dim)
        self.ncdar = NCDAR(local_dim=base_dim, context_dim=fusion_dim, hidden_dim=hidden_dim)
        self.classifier = ClassificationHead(hidden_dim, num_classes=2)

    def forward(self, nodule_roi, context_rois):
        fused_low_rank = self.ncd_mambalite(nodule_roi, context_rois)
        # Need local and context features separately for NCDAR (we'll re-extract from NCD_MambaLite)
        # Simpler: reuse the low-rank fused as both? Actually NCDAR expects local and context
        # To keep code simple, we'll modify NCD_MambaLite to also return local and context vectors.
        # Here we just call the modified version.
        local_vec, ctx_vec, fused_low_rank = self.ncd_mambalite.forward_with_features(nodule_roi, context_rois)
        refined = self.ncdar(local_vec, ctx_vec, fused_low_rank)
        logits = self.classifier(refined)
        return logits

# Modify NCD_MambaLite to also return features (for NCDAR)
class NCD_MambaLite_Modified(nn.Module):
    def __init__(self, in_channels=1, base_dim=32, mamba_dim=64, fusion_dim=128, rank=8):
        super().__init__()
        # same as before but add method forward_with_features
        # (reuse code from NCD_MambaLite, but we'll rewrite compactly)
        self.base_dim = base_dim
        self.mamba_dim = mamba_dim
        self.fusion_dim = fusion_dim
        self.local_dsconv = DepthwiseSeparableConv3d(in_channels, base_dim)
        self.local_mnv4 = MobileNetV4Block(base_dim, base_dim)
        self.local_pool = nn.AdaptiveAvgPool3d(1)
        self.context_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels, mamba_dim, kernel_size=3, padding=1),
                MambaOutFemto(mamba_dim, num_blocks=2),
                nn.AdaptiveAvgPool3d(1)
            ) for _ in range(2)
        ])
        self.context_proj = nn.Linear(mamba_dim * 2, fusion_dim)
        self.lr_fusion = LowRankBilinearFusion(base_dim, fusion_dim, fusion_dim, rank)
        self.ghost = GhostCompression(fusion_dim, fusion_dim, ratio=2)

    def forward_with_features(self, nodule_roi, context_rois):
        # Local path
        local_feat = self.local_dsconv(nodule_roi)
        local_feat = self.local_mnv4(local_feat)
        local_vec = self.local_pool(local_feat).flatten(1)
        # Context path
        ctx_vecs = []
        for i, ctx_roi in enumerate(context_rois):
            ctx = self.context_encoders[i](ctx_roi)
            ctx_vecs.append(ctx.flatten(1))
        ctx_vec = torch.cat(ctx_vecs, dim=1)
        ctx_vec = self.context_proj(ctx_vec)
        # Fusion
        fused = self.lr_fusion(local_vec, ctx_vec)
        # Reshape and Ghost
        fused_sp = fused.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        compressed = self.ghost(fused_sp).flatten(1)
        return local_vec, ctx_vec, compressed

    def forward(self, nodule_roi, context_rois):
        _, _, out = self.forward_with_features(nodule_roi, context_rois)
        return out

# Update FullModel to use modified NCD_MambaLite
class FullModel(nn.Module):
    def __init__(self, in_channels=1, base_dim=32, mamba_dim=64, fusion_dim=128, hidden_dim=128):
        super().__init__()
        self.ncd_mambalite = NCD_MambaLite_Modified(in_channels, base_dim, mamba_dim, fusion_dim)
        self.ncdar = NCDAR(local_dim=base_dim, context_dim=fusion_dim, hidden_dim=hidden_dim)
        self.classifier = ClassificationHead(hidden_dim, 2)

    def forward(self, nodule_roi, context_rois):
        local_vec, ctx_vec, fused = self.ncd_mambalite.forward_with_features(nodule_roi, context_rois)
        refined = self.ncdar(local_vec, ctx_vec, fused)
        logits = self.classifier(refined)
        return logits

# ---------------------------
# 8. Training and Evaluation
# ---------------------------

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for nodule, contexts, labels in dataloader:
        nodule = nodule.to(device)
        contexts = [c.to(device) for c in contexts]
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(nodule, contexts)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    with torch.no_grad():
        for nodule, contexts, labels in dataloader:
            nodule = nodule.to(device)
            contexts = [c.to(device) for c in contexts]
            logits = model(nodule, contexts)
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

# ---------------------------
# 9. Main Script
# ---------------------------

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 4
    epochs = 3  # small for demonstration
    lr = 1e-3
    base_dim = 32
    mamba_dim = 64
    fusion_dim = 128
    hidden_dim = 128

    # Create synthetic dataset
    print("Generating synthetic dataset...")
    dataset = SyntheticLungNoduleDataset(num_samples=200, roi_size=(32,32,32), context_scales=[1,2])
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = FullModel(in_channels=1, base_dim=base_dim, mamba_dim=mamba_dim,
                      fusion_dim=fusion_dim, hidden_dim=hidden_dim).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    print("\nStarting training...")
    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

    # Evaluation
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Demonstrate ECT-DiagLite on a sample
    print("\n--- ECT-DiagLite Demonstration ---")
    ect = ECTDiagLite(model.ncd_mambalite, model.classifier, device)
    # Take one test sample
    sample_nodule, sample_contexts, sample_label = test_dataset[0]
    sample_nodule = sample_nodule.unsqueeze(0).to(device)  # add batch dim
    sample_contexts = [c.unsqueeze(0).to(device) for c in sample_contexts]

    # Get prediction
    model.eval()
    with torch.no_grad():
        logits = model(sample_nodule, sample_contexts)
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        prob_malignant = probs[0,1].item()
    print(f"Sample true label: {sample_label.item()}, predicted: {pred}, malignant prob: {prob_malignant:.3f}")

    # Generate explanations
    cam = ect.grad_cam_plusplus(sample_nodule, sample_contexts)[0]  # remove batch
    lime = ect.simplified_lime(sample_nodule, sample_contexts)
    consistency = ect.explanation_consistency(cam, lime)
    print(f"Explanation consistency: {consistency:.3f}")

    # Simulate uncertainty (e.g., from MC dropout, we just use random for demonstration)
    uncertainty = np.random.uniform(0, 0.3)  # placeholder
    trust = ect.compute_trust(prob_malignant, uncertainty, consistency)
    print(f"Trust score: {trust:.3f}")
    # Selective routing
    threshold = 0.5  # example
    if trust >= threshold:
        print("Decision: Automated reporting")
    else:
        print("Decision: Refer for clinical review")

    print("\nScript completed successfully.")

if __name__ == "__main__":
    main()