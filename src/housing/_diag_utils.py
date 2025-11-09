import numpy as np

def make_full_rank(X, tol=1e-8):
    """
    传入 numpy 2D 矩阵，返回去掉零方差列、并用QR挑选出的线性无关列。
    输出：X_reduced
    """
    Xc = X.copy()
    # 1) 去零方差列
    keep = [i for i in range(Xc.shape[1]) if np.nanmax(Xc[:, i]) - np.nanmin(Xc[:, i]) > 0]
    Xc = Xc[:, keep] if keep else Xc[:, :1]  # 至少保留常数项

    # 2) 去NaN行（仅用于诊断，不影响已估计模型）
    mask = ~np.isnan(Xc).any(1)
    Xc = Xc[mask, :]

    # 3) 若仍不满秩，用QR选取独立列
    if Xc.size == 0:
        return Xc  # 空矩阵，交给上层处理
    Q, R, piv = np.linalg.qr(Xc, mode='economic', pivoting=True)
    r = int((np.abs(np.diag(R)) > tol).sum())
    piv = piv[:r] if r>0 else piv[:1]
    Xr = Xc[:, piv]
    return Xr
