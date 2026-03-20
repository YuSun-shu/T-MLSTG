
import torch
import torch.nn as nn


class WMGD(nn.Module):
    def __init__(self, window_size=2, sigma=1):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma

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
        features = []
        for t in range(T - self.window_size + 1):
            window_feat = x[:, t:t + self.window_size, :].mean(dim=1)  # (B, F)
            features.append(window_feat)
        if not features:
            return x.mean(dim=1).unsqueeze(0)  # (1, B, F)

        return torch.stack(features, dim=0)  # (L, B, F)，其中L是窗口数

    def extract_gradient_features(self, x):
        if x.size(1) < 2:
            return x.mean(dim=1).unsqueeze(0)
        gradients = x[:, 1:, :] - x[:, :-1, :]  # (B, T-1, F)
        gradients = gradients.permute(1, 0, 2)  # (T-1, B, F)

        return gradients  # (T-1, B, F)

    def compute_composite_kernel_matrix(self, Fs, Ft):
        B_s = Fs.shape[0]
        B_t = Ft.shape[0]

        K_ss = self._compute_composite_kernel_block(Fs, Fs)  # (B_s, B_s)
        K_tt = self._compute_composite_kernel_block(Ft, Ft)  # (B_t, B_t)
        K_st = self._compute_composite_kernel_block(Fs, Ft)  # (B_s, B_t)

        K = torch.zeros((B_s + B_t, B_s + B_t), device=Fs.device)
        K[:B_s, :B_s] = K_ss
        K[B_s:, B_s:] = K_tt
        K[:B_s, B_s:] = K_st
        K[B_s:, :B_s] = K_st.T

        return K, (B_s, B_t)

    def _compute_composite_kernel_block(self, X, Y):
        B_x = X.shape[0]
        B_y = Y.shape[0]

        # 提取特征
        X_windows = self.extract_window_features(X)  # (L_x, B_x, F)
        Y_windows = self.extract_window_features(Y)  # (L_y, B_y, F)

        # 使用改进的梯度特征提取
        X_grad = self.extract_gradient_features(X)  # (L_grad_x, B_x, F)
        Y_grad = self.extract_gradient_features(Y)  # (L_grad_y, B_y, F)

        L_x = X_windows.shape[0]
        L_y = Y_windows.shape[0]
        L_grad_x = X_grad.shape[0]
        L_grad_y = Y_grad.shape[0]

        # 计算复合核矩阵
        K_composite = torch.zeros((B_x, B_y), device=X.device)

        # 1. 窗口特征贡献
        if L_x > 0 and L_y > 0:
            for i in range(L_x):
                for j in range(L_y):
                    K_window = self.rbf_kernel(X_windows[i], Y_windows[j])
                    K_composite += K_window / (L_x * L_y)

        # 2. 梯度特征贡献
        if L_grad_x > 0 and L_grad_y > 0:
            for i in range(L_grad_x):
                for j in range(L_grad_y):
                    K_grad = self.rbf_kernel(X_grad[i], Y_grad[j])
                    K_composite += K_grad / (L_grad_x * L_grad_y)

        return K_composite

    def compute_wmgd(self, Fs, Ft):
        K, (B_s, B_t) = self.compute_composite_kernel_matrix(Fs, Ft)

        K_ss = K[:B_s, :B_s]
        K_tt = K[B_s:, B_s:]
        K_st = K[:B_s, B_s:]

        wmgd_sq = (K_ss.mean() + K_tt.mean() - 2 * K_st.mean())
        wmgd = torch.sqrt(torch.relu(wmgd_sq) + 1e-8)

        return wmgd

    def forward(self, Fs, Ft):
        wmgd = self.compute_wmgd_squared(Fs, Ft)
        return wmgd
