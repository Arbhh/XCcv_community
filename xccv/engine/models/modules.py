"""工业场景优化模块库（纯 torch/torchvision 实现，不依赖 mmcv/timm）。

对应 code.md 的改进方案：
- CBAM / ECA 注意力
- DCNv2 可变形卷积（基于 torchvision.ops.DeformConv2d）
- Focal Loss / Varifocal 损失 / Label Smoothing（torchvision.sigmoid_focal_loss）
- Dice + BCE 混合掩码损失
- Focaler-WIoU v3（简化实现）边界框损失（YOLO）
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d, sigmoid_focal_loss

log = logging.getLogger(__name__)


# ============================================================
# 注意力模块
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        w = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.max(dim=1, keepdim=True)[0]
        w = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * w


class CBAM(nn.Module):
    """CBAM 注意力（通道 + 空间），用于 ResNet 层级输出后与 R-CNN stage 后。

    残差门控恒等初始化：gate 零初始化使初始输出严格等于输入——
    注意力模块随机初始化的直接乘性问题（sigmoid≈0.5 的两级相乘把
    layer3/4 特征幅值压到约 1/4）会破坏 FPN 期望的预训练特征分布，
    小数据集短轮次下难以恢复，是"工业优化反而更差"的主要来源之一。
    训练中 gate 逐渐放大，注意力按需生效。
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gate * self.spatial_attn(self.channel_attn(x))


class ECA(nn.Module):
    """ECA 通道注意力（一维卷积，无降维），替代 MobileNetV3 的 SE。"""

    def __init__(self, channels: int, k_size: int | None = None):
        super().__init__()
        if k_size is None:
            t = int(abs((torch.log2(torch.tensor(channels, dtype=torch.float32)) + 1) / 2))
            k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)

    def forward(self, x):
        y = F.adaptive_avg_pool2d(x, 1)                     # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(1, 2)).transpose(1, 2).unsqueeze(-1)
        return x * torch.sigmoid(y)


# ============================================================
# DCNv2 可变形卷积（torchvision.ops 实现）
# ============================================================
_DCN_EXPORT_MODE = {"on": False}


def set_dcn_export_mode(on: bool) -> None:
    """导出开关：ONNX/TRT 无 deform_conv 算子，导出期间用 GridSample 等价路径。"""
    _DCN_EXPORT_MODE["on"] = bool(on)


class DCNv2(nn.Module):
    """用 torchvision.ops.DeformConv2d 实现的 DCNv2。

    offset 卷积零初始化，初始等价于普通卷积；可直接继承原 conv 的权重。
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        kh, kw = conv.kernel_size
        assert kh == kw == 3, "DCNv2 仅替换 3×3 卷积"
        self.with_bias = conv.bias is not None
        # offset 卷积与主卷积同 stride/padding，保证 offset 图与输出特征图同尺寸
        self.conv_offset = nn.Conv2d(conv.in_channels, 2 * kh * kw, 3,
                                     stride=conv.stride, padding=conv.padding)
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)
        self.deform = DeformConv2d(
            conv.in_channels, conv.out_channels, (kh, kw),
            stride=conv.stride, padding=conv.padding,
            dilation=conv.dilation, groups=conv.groups,
            bias=self.with_bias,
        )
        self.deform.weight.data = conv.weight.data.clone()
        if self.with_bias:
            self.deform.bias.data = conv.bias.data.clone()

    def forward(self, x):
        offset = self.conv_offset(x)
        if _DCN_EXPORT_MODE["on"]:
            return self._forward_gridsample(x, offset)
        if torch.is_autocast_enabled(x.device.type):
            # torchvision 的 deform_conv2d 没有 autocast 封装（roi_align 有），
            # fp16 下偏移采样网格精度丢失会导致特征错乱、训练不收敛
            # （表现为 mAP 近零、检出乱框、掩码碎片化）——强制 fp32 计算
            with torch.autocast(x.device.type, enabled=False):
                return self.deform(x.float(), offset.float())
        return self.deform(x, offset)

    def _forward_gridsample(self, x, offset):
        """DCN 的 GridSample 等价实现（仅导出期使用）。

        DeformConv2d 在输出位置 p 的结果 = Σ_k W_k · x(p + base_k + offset_k)，
        其中 W_k 是卷积核第 k 个位置的 (Cout, C) 切片——等价于"按偏移采样出
        平移特征图后做 1×1 卷积再逐 k 累加"。GridSample/Conv 均为 TRT 原生算子。
        """
        import torch.nn.functional as F

        if self.deform.groups != 1:
            raise RuntimeError("DCN GridSample 导出路径仅支持 groups=1 的卷积")
        B, C, H, W = x.shape
        Cout, _, kh, kw = self.deform.weight.shape
        stride = self.deform.stride[0]
        pad = self.deform.padding[0]
        dil = self.deform.dilation[0]
        Ho = (H + 2 * pad - dil * (kh - 1) - 1) // stride + 1
        Wo = (W + 2 * pad - dil * (kw - 1) - 1) // stride + 1
        dev = x.device
        # 输出像素映射回输入图的基准采样坐标（像素，核网格行主序 k = i*kw + j）
        oy = torch.arange(Ho, device=dev, dtype=x.dtype) * stride - pad
        ox = torch.arange(Wo, device=dev, dtype=x.dtype) * stride - pad
        out = None
        for i in range(kh):
            for j in range(kw):
                k = i * kw + j
                base_y = (oy + i * dil).view(1, Ho, 1) + offset[:, 2 * k]
                base_x = (ox + j * dil).view(1, 1, Wo) + offset[:, 2 * k + 1]
                grid = torch.stack([
                    base_x / max(W - 1, 1) * 2 - 1,
                    base_y / max(H - 1, 1) * 2 - 1,
                ], dim=-1)  # (B, Ho, Wo, 2)，align_corners=True 对应像素中心
                sk = F.grid_sample(x, grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
                w_k = self.deform.weight[:, :, i, j].reshape(Cout, C, 1, 1)
                part = F.conv2d(sk, w_k)
                out = part if out is None else out + part
        if self.with_bias:
            out = out + self.deform.bias.view(1, -1, 1, 1)
        return out


def replace_convs_with_dcn(module: nn.Module, skip_classes: tuple = ()) -> int:
    """递归把 module 内 3×3 的 nn.Conv2d 替换为 DCNv2，返回替换数量。"""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, skip_classes):
            continue
        if isinstance(child, nn.Conv2d) and child.kernel_size == (3, 3):
            setattr(module, name, DCNv2(child))
            count += 1
        else:
            count += replace_convs_with_dcn(child, skip_classes)
    return count


def load_state_dict_compat(net: nn.Module, state_dict: dict) -> list[str]:
    """兼容加载权重：旧版本产出的权重文件缺少少量键时按安全默认值补齐。

    允许缺失的键（均可安全补零，不改变推理语义）：
    - CBAM gate 门控（后新增参数，恒等初始化与旧行为最接近）；
    - BN 的 num_batches_tracked 计数器（剪枝/恢复链路的历史产物，纯记账）。
    其余缺失/多余键仍按严格模式抛错，避免静默错位加载。返回补齐的键。
    """
    model_state = net.state_dict()
    missing = model_state.keys() - state_dict.keys()
    fillable = {k for k in missing
                if k.rsplit(".", 1)[-1] in ("gate", "num_batches_tracked")}
    unexpected = state_dict.keys() - model_state.keys()
    if unexpected:
        raise RuntimeError(f"权重包含未知参数: {sorted(unexpected)[:8]}")
    if missing - fillable:
        raise RuntimeError(f"权重缺少参数: {sorted(missing - fillable)[:8]}")
    if fillable:
        state_dict = dict(state_dict)
        for k in fillable:
            state_dict[k] = torch.zeros(model_state[k].shape)
        log.warning("权重缺少少量可安全补齐的参数（已按默认值补零）: %s",
                    sorted(fillable))
    net.load_state_dict(state_dict)
    return sorted(fillable)


# ============================================================
# 损失：分类（Focal / Varifocal）
# ============================================================
class FocalLoss(nn.Module):
    """多分类 Focal Loss（γ=2；α 可为类别权重张量，应对类别不均衡）。"""

    def __init__(self, gamma: float = 2.0, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, targets, weight=self.class_weights, reduction="none")
        p_t = torch.exp(-ce)
        return ((1 - p_t) ** self.gamma * ce).mean()


class VarifocalBCE(nn.Module):
    """与 ultralytics BCEWithLogitsLoss 同签名的 Varifocal 逐元素损失。

    正样本以 gt_score（IoU 感知质量）加权，负样本以 α·p^γ 降权。
    用于替换 v8DetectionLoss.bce，外层 .sum()/target_scores_sum 逻辑不变。
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_scores, target_scores):
        target = target_scores.to(pred_scores.dtype)
        label = (target > 0).to(pred_scores.dtype)
        weight = label * target + (1 - label) * self.alpha * pred_scores.detach().sigmoid().pow(self.gamma)
        return self.bce(pred_scores, target) * weight


# ============================================================
# 损失：Focaler-WIoU v3（简化实现）边界框损失（YOLO 用）
# ============================================================
class FocalerWIoUBboxLoss(nn.Module):
    """复刻 ultralytics BboxLoss 接口；IoU 项替换为 Focaler-WIoU v3（简化实现）。

    L = (1 − IoU) · R_WIoU · r_v3 · weight
      R_WIoU = exp(中心距离项)（W/H 与梯度解耦）
      r_v3   = β/(δ·α^(β−δ)) 单调聚焦系数（α=1.9, δ=3.0）
    DFL 项与原生实现一致。
    """

    def __init__(self, reg_max: int = 16, alpha: float = 1.9, delta: float = 3.0):
        super().__init__()
        from ultralytics.utils.loss import DFLoss
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.alpha = alpha
        self.delta = delta
        self._bbox_iou = bbox_iou
        self._bbox2dist = bbox2dist

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb = pred_bboxes[fg_mask]    # (K, 4) 前景预测框
        tb = target_bboxes[fg_mask]  # (K, 4) 前景目标框
        iou = self._bbox_iou(pb, tb, xywh=False, CIoU=False)

        gt_w = (tb[:, 2] - tb[:, 0]).clamp(min=1)
        gt_h = (tb[:, 3] - tb[:, 1]).clamp(min=1)
        cx_p = (pb[:, 0] + pb[:, 2]) / 2
        cy_p = (pb[:, 1] + pb[:, 3]) / 2
        cx_g = (tb[:, 0] + tb[:, 2]) / 2
        cy_g = (tb[:, 1] + tb[:, 3]) / 2
        wdist = ((cx_p - cx_g) ** 2 + (cy_p - cy_g) ** 2) / ((gt_w ** 2 + gt_h ** 2) * 2 + 1e-7)
        r_wiou = torch.exp(wdist.detach())                       # 距离项与梯度解耦

        iou_loss = (1 - iou).detach()
        beta = (iou_loss / iou_loss.mean().clamp(min=1e-7)).clamp(max=5)
        r_v3 = (beta / (self.delta * self.alpha ** (beta - self.delta))).clamp(0.5, 2.8)

        loss_iou = ((1 - iou) * r_wiou * r_v3 * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = self._bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                                      target_ltrb[fg_mask]) * weight).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_bboxes.device)
        return loss_iou, loss_dfl


class NWDCIoUBboxLoss(nn.Module):
    """YOLO（anchor-free）专用工业框损失：CIoU ⊕ NWD 混合。

    微小缺陷（数像素）下 IoU 对"中心偏几十像素"几乎无梯度（IoU 梯度
    消失），NWD（Normalized Wasserstein Distance，Wang 2021）把框建模
    为二维高斯、以常数 C 归一——尺度无关、恒有界 (0,1]，微小目标在
    IoU 无感区间仍提供平滑位置梯度。混合：combined = λ·CIoU +
    (1-λ)·NWD（λ=0.5），loss = (1-combined)·weight / target_scores_sum。
    DFL 与原生 BboxLoss 完全一致。输入与 ultralytics≥8.4 的 BboxLoss
    相同（target_bboxes 为特征尺度 xyxy，stride 为逐锚步长张量）。
    """

    def __init__(self, reg_max: int = 16, nwd_c: float = 12.8,
                 blend: float = 0.5):
        super().__init__()
        from ultralytics.utils.loss import DFLoss
        from ultralytics.utils.metrics import bbox_iou
        from ultralytics.utils.tal import bbox2dist

        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self._bbox_iou = bbox_iou
        self._bbox2dist = bbox2dist
        self.nwd_c = float(nwd_c)
        self.blend = float(blend)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pb = pred_bboxes[fg_mask]    # (K, 4) xyxy，特征尺度
        tb = target_bboxes[fg_mask]
        iou = self._bbox_iou(pb, tb, xywh=False, CIoU=True)

        # NWD（像素尺度）：高斯 W2² = 中心差² + (σ差)²，σ=w/2、h/2；
        # 特征尺度差 × 逐锚 stride 还原像素后再除常数 C
        pw = (pb[:, 2] - pb[:, 0]).clamp(min=0)
        ph = (pb[:, 3] - pb[:, 1]).clamp(min=0)
        tw = (tb[:, 2] - tb[:, 0]).clamp(min=0)
        th = (tb[:, 3] - tb[:, 1]).clamp(min=0)
        pcx, pcy = (pb[:, 2] + pb[:, 0]) / 2, (pb[:, 3] + pb[:, 1]) / 2
        tcx, tcy = (tb[:, 2] + tb[:, 0]) / 2, (tb[:, 3] + tb[:, 1]) / 2
        s = stride.squeeze(-1).to(pb.dtype)                 # (N,) 逐锚步长
        s = s.unsqueeze(0).expand(fg_mask.shape[0], -1)[fg_mask]
        w2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2             + ((pw - tw) ** 2 + (ph - th) ** 2) / 4
        nwd = torch.exp(-torch.sqrt(w2 + 1e-12) * s / self.nwd_c)

        combined = self.blend * iou + (1.0 - self.blend) * nwd
        loss_iou = ((1.0 - combined) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = self._bbox2dist(anchor_points, target_bboxes,
                                          self.dfl_loss.reg_max - 1)
            loss_dfl = (self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]) * weight).sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_bboxes.device)
        return loss_iou, loss_dfl


# ============================================================
# 损失：Mask 的 Dice + BCE 混合（R-CNN 族用）
# ============================================================
def dice_bce_mask_loss(mask_logits, mask_targets) -> torch.Tensor:
    """mask_logits/targets: (N, H, W) sigmoid 概率与 0/1 目标；Dice 与 BCE 各半。"""
    if mask_targets.numel() == 0:
        return mask_logits.sum() * 0
    probs = torch.sigmoid(mask_logits)
    bce = F.binary_cross_entropy(probs, mask_targets, reduction="mean")
    dims = tuple(range(1, mask_targets.dim()))
    inter = (probs * mask_targets).sum(dim=dims)
    card = probs.sum(dim=dims) + mask_targets.sum(dim=dims)
    dice = 1 - ((2 * inter + 1) / (card + 1)).mean()
    return 0.5 * bce + 0.5 * dice
