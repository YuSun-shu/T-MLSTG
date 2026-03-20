
import torch
import torch.nn as nn


class WMGD(nn.Module):
    """
    时空MMD，支持有偏和无偏估计
    """

    def __init__(self, window_size=3, sigma=1.0, biased=True):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.biased = biased  # True:有偏估计, False:无偏估计

    def rbf_kernel(self, x, y):
        """计算RBF核矩阵"""
        if x.dim() == 3:
            x = x.view(x.size(0), -1)
        if y.dim() == 3:
            y = y.view(y.size(0), -1)

        d = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-d / (2 * self.sigma + 1e-8))

    def extract_window_features(self, x):
        """提取滑动窗口特征"""
        B, T, F = x.shape

        if T < self.window_size:
            # 如果时间维度小于窗口大小，使用整个序列
            return x.mean(dim=1).unsqueeze(0)  # (1, B, F)

        features = []
        for t in range(T - self.window_size + 1):
            window_feat = x[:, t:t + self.window_size, :].mean(dim=1)  # (B, F)
            features.append(window_feat)

        if not features:
            return x.mean(dim=1).unsqueeze(0)  # (1, B, F)

        return torch.stack(features, dim=0)  # (L, B, F)，其中L是窗口数

    def extract_gradient_features(self, x):
        """提取梯度特征"""
        if x.size(1) < 2:
            return x.mean(dim=1).unsqueeze(0)  # (1, B, F)

        gradients = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, F)
        gradient_mean = gradients.mean(dim=1)  # (B, F)
        return gradient_mean.unsqueeze(0)  # (1, B, F)

    def compute_composite_kernel(self, Fs, Ft):
        """
        计算复合核：k_w ⊕ k_g
        返回：K_ww, K_gg, K_wg
        """
        B_s = Fs.shape[0]
        B_t = Ft.shape[0]

        # 提取窗口特征：形状 (L, B, F)，其中L是窗口数
        Fs_windows = self.extract_window_features(Fs)  # (L_s, B_s, F)
        Ft_windows = self.extract_window_features(Ft)  # (L_t, B_t, F)

        L_s = Fs_windows.shape[0]
        L_t = Ft_windows.shape[0]

        # 提取梯度特征：形状 (1, B, F)
        Fs_grad = self.extract_gradient_features(Fs)  # (1, B_s, F)
        Ft_grad = self.extract_gradient_features(Ft)  # (1, B_t, F)

        # 初始化复合核矩阵
        K_ww = torch.zeros((B_s, B_s), device=Fs.device)  # 窗口-窗口复合核
        K_gg = torch.zeros((B_t, B_t), device=Fs.device)  # 梯度-梯度复合核
        K_wg = torch.zeros((B_s, B_t), device=Fs.device)  # 窗口-梯度复合核

        # 计算窗口复合核 ⟨k_w(x_i^{sour},) ⊕ k_g(x_j^{sour},)⟩
        for i in range(L_s):
            for j in range(L_s):
                # 窗口核部分
                K_w_ij = self.rbf_kernel(Fs_windows[i], Fs_windows[j])  # (B_s, B_s)
                # 梯度核部分
                K_g_ij = self.rbf_kernel(Fs_grad[0], Fs_grad[0])  # (B_s, B_s)
                # 复合核 = 窗口核 + 梯度核
                K_ww += K_w_ij + K_g_ij

        # 计算目标域复合核 ⟨k_w(x_i^{tar},) ⊕ k_g(x_j^{tar},)⟩
        for i in range(L_t):
            for j in range(L_t):
                # 窗口核部分
                K_w_ij = self.rbf_kernel(Ft_windows[i], Ft_windows[j])  # (B_t, B_t)
                # 梯度核部分
                K_g_ij = self.rbf_kernel(Ft_grad[0], Ft_grad[0])  # (B_t, B_t)
                # 复合核 = 窗口核 + 梯度核
                K_gg += K_w_ij + K_g_ij

        # 计算交叉复合核 ⟨k_w(x_i^{sour},) ⊕ k_g(x_j^{tar},)⟩
        for i in range(L_s):
            for j in range(L_t):
                # 窗口核部分
                K_w_ij = self.rbf_kernel(Fs_windows[i], Ft_windows[j])  # (B_s, B_t)
                # 梯度核部分
                K_g_ij = self.rbf_kernel(Fs_grad[0], Ft_grad[0])  # (B_s, B_t)
                # 复合核 = 窗口核 + 梯度核
                K_wg += K_w_ij + K_g_ij

        # 归一化：除以窗口数量的平方
        K_ww = K_ww / (L_s * L_s + 1e-8)
        K_gg = K_gg / (L_t * L_t + 1e-8)
        K_wg = K_wg / (L_s * L_t + 1e-8)

        return K_ww, K_gg, K_wg

    def compute_WMGD(self, K_ww, K_gg, K_wg):
        """
        计算MMD，根据biased参数选择有偏或无偏估计
        """
        B_s = K_ww.shape[0]  # 源域样本数
        B_t = K_gg.shape[0]  # 目标域样本数

        if self.biased:
            # 有偏估计
            mmd = (torch.sum(K_ww) / (B_s * B_s + 1e-8) +
                   torch.sum(K_gg) / (B_t * B_t + 1e-8) -
                   2 * torch.sum(K_wg) / (B_s * B_t + 1e-8))
        else:
            # 无偏估计
            # 对于源域和目标域内部核，减去对角线元素（排除i=j的情况）
            trace_ww = torch.trace(K_ww)
            trace_gg = torch.trace(K_gg)

            # 检查样本数是否足够（需要至少2个样本才能进行无偏估计）
            if B_s <= 1 or B_t <= 1:
                # 样本数不足，退化为有偏估计
                mmd = (torch.sum(K_ww) / (B_s * B_s + 1e-8) +
                       torch.sum(K_gg) / (B_t * B_t + 1e-8) -
                       2 * torch.sum(K_wg) / (B_s * B_t + 1e-8))
            else:
                # 无偏估计公式
                mmd = ((torch.sum(K_ww) - trace_ww) / (B_s * (B_s - 1) + 1e-8) +
                       (torch.sum(K_gg) - trace_gg) / (B_t * (B_t - 1) + 1e-8) -
                       2 * torch.sum(K_wg) / (B_s * B_t + 1e-8))

        return torch.relu(mmd) + 1e-8

    def forward(self, Fs, Ft):
        """前向传播：计算复合核MMD"""
        # 确保输入是三维的
        if Fs.dim() == 2:
            Fs = Fs.unsqueeze(1)
        if Ft.dim() == 2:
            Ft = Ft.unsqueeze(1)

        # 计算复合核矩阵
        K_ww, K_gg, K_wg = self.compute_composite_kernel(Fs, Ft)

        # 计算WMGD
        return self.compute_WMGD(K_ww, K_gg, K_wg)