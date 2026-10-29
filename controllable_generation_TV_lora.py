import functools

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from sampling import NoneCorrector, NonePredictor, shared_corrector_update_fn, shared_predictor_update_fn
from utils import clear, batchfy


class lambda_schedule:
  def __init__(self, total=2000):
    self.total = total

  def get_current_lambda(self, i):
    pass

class lambda_schedule_linear(lambda_schedule):
  def __init__(self, start_lamb=1.0, end_lamb=0.0):
    super().__init__()
    self.start_lamb = start_lamb
    self.end_lamb = end_lamb

  def get_current_lambda(self, i):
    return self.start_lamb + (self.end_lamb - self.start_lamb) * (i / self.total)


class lambda_schedule_const(lambda_schedule):
  def __init__(self, lamb=1.0):
    super().__init__()
    self.lamb = lamb

  def get_current_lambda(self, i):
    return self.lamb


def _Dz(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]
    return y - x


def _DzT(x): # Batch direction
    y = torch.zeros_like(x)
    y[:-1] = x[1:]
    y[-1] = x[0]

    tempt = -(y-x)
    difft = tempt[:-1]
    y[1:] = difft
    y[0] = x[-1] - x[0]

    return y

def _Dx(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    return y - x


def _DxT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :-1, :] = x[:, :, 1:, :]
    y[:, :, -1, :] = x[:, :, 0, :]
    tempt = -(y - x)
    difft = tempt[:, :, :-1, :]
    y[:, :, 1:, :] = difft
    y[:, :, 0, :] = x[:, :, -1, :] - x[:, :, 0, :]
    return y


def _Dy(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    return y - x


def _DyT(x):  # Batch direction
    y = torch.zeros_like(x)
    y[:, :, :, :-1] = x[:, :, :, 1:]
    y[:, :, :, -1] = x[:, :, :, 0]
    tempt = -(y - x)
    difft = tempt[:, :, :, :-1]
    y[:, :, :, 1:] = difft
    y[:, :, :, 0] = x[:, :, :, -1] - x[:, :, :, 0]
    return y

def shrink(weight_src, lamb):
    return torch.sign(weight_src) * torch.max(torch.abs(weight_src) - lamb, torch.zeros_like(weight_src))


def SVT(X, tau):
    # 奇异值阈值处理函数，适用于批量数据
    device = X.device
    batch_size, slices, h, w = X.shape
    X_mat = X.view(batch_size, slices, h * w)
    U, S, Vh = torch.linalg.svd(X_mat, full_matrices=False)
    S_thresh = torch.clamp(S - tau, min=0)
    # 重建矩阵
    reconstructed = torch.matmul(U * S_thresh.unsqueeze(1), Vh)
    reconstructed = reconstructed.view(batch_size, slices, h, w)
    return reconstructed.to(device)


def get_pc_radon_ADMM_TV_lora_vol(sde, predictor, corrector, inverse_scaler, snr,
                             n_steps=1, probability_flow=False, continuous=False,
                             denoise=True, eps=1e-5, radon=None, save_progress=True, save_root=None,
                             final_consistency=False, img_shape=None, lamb_1=5, lamb_2=5, rho_1=10, rho_2=10,
                             lamb_lowrank=1.0):
    """ Sparse application of measurement consistency with Anisotropic TV and Low-Rank constraints """
    # 初始化ADMM变量
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)
    del_x = torch.zeros(img_shape)
    del_y = torch.zeros(img_shape)
    del_z = torch.zeros(img_shape)
    udel_x = torch.zeros(img_shape)
    udel_y = torch.zeros(img_shape)
    udel_z = torch.zeros(img_shape)
    L = torch.zeros(img_shape)  # 低秩变量
    u_lowrank = torch.zeros(img_shape)  # 低秩对偶变量
    eps = 1e-10

    def _A(x):
        return radon.A(x)

    def _AT(sinogram):
        return radon.AT(sinogram)

    def kaczmarz(x, x_mean, measurement=None, lamb=1.0, i=None, norm_const=None):
        x = x + lamb * _AT(measurement - _A(x)) / norm_const
        x_mean = x
        return x, x_mean

    def A_cg(x):
        return _AT(_A(x)) + rho_1*(_DxT(_Dx(x)) + _DyT(_Dy(x)) + _DzT(_Dz(x))) + rho_2 * x

    def CG(A_fn, b_cg, x, n_inner=10):
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)

        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)
            x += a * p
            r -= a * Ap
            rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
            if torch.sqrt(rs_new) < eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x

    def CS_routine(x, ATy, niter=20):
        nonlocal del_x, del_y, del_z, udel_x, udel_y, udel_z, L, u_lowrank
        # 设备一致性检查
        if del_z.device != x.device:
            del_x = del_x.to(x.device)
            del_y = del_y.to(x.device)
            del_z = del_z.to(x.device)
            udel_x = udel_x.to(x.device)
            udel_y = udel_y.to(x.device)
            udel_z = udel_z.to(x.device)
            L = L.to(x.device)
            u_lowrank = u_lowrank.to(x.device)

        for i in range(niter):
            # b_cg = ATy + rho_1 * (_DxT(del_x - udel_x) + rho_1*(_DyT(del_y - udel_y)) + rho_1*(_DzT(del_z - udel_z)) + rho_2*(L - u_lowrank)

            b_cg = ATy + rho_1 * ((_DxT(del_x) - _DxT(udel_x)) + (_DyT(del_y) - _DyT(udel_y)) +  (_DzT(del_z) - _DzT(udel_z))) + rho_2*(L - u_lowrank)

            x = CG(A_cg, b_cg, x, n_inner=1)
            # 更新各向异性TV变量
            del_x = shrink(_Dx(x) + udel_x, lamb_1 / rho_1)
            del_y = shrink(_Dy(x) + udel_y, lamb_1 / rho_1)
            del_z = shrink(_Dz(x) + udel_z, lamb_1 / rho_1)

            # 更新TV对偶变量
            udel_x += _Dx(x) - del_x
            udel_y += _Dy(x) - del_y
            udel_z += _Dz(x) - del_z

            # 更新低秩变量
            L = SVT(x + u_lowrank, lamb_lowrank / rho_2)

            # 更新低秩对偶变量
            u_lowrank += x - L

        x_mean = x
        return x, x_mean

    # 其余部分与原代码保持一致


    def get_update_fn(update_fn):
        def radon_update_fn(model, data, x, t):
            with torch.no_grad():
                vec_t = torch.ones(x.shape[0], device=x.device) * t
                x, x_mean = update_fn(x, vec_t, model=model)
                return x, x_mean
        return radon_update_fn

    def get_ADMM_TV_fn():
        def ADMM_TV_fn(x, measurement=None):
            with torch.no_grad():
                ATy = _AT(measurement)
                x, x_mean = CS_routine(x, ATy, niter=1)
                return x, x_mean
        return ADMM_TV_fn

    predictor_denoise_update_fn = get_update_fn(predictor_update_fn)
    corrector_denoise_update_fn = get_update_fn(corrector_update_fn)
    mc_update_fn = get_ADMM_TV_fn()

    def pc_radon(model, data, measurement=None, batch_size=12):
        with torch.no_grad():
            x = sde.prior_sampling(data.shape).to(data.device)

            ones = torch.ones_like(x).to(data.device)
            norm_const = _AT(_A(ones))
            timesteps = torch.linspace(sde.T, eps, sde.N)
            for i in tqdm(range(sde.N)):
                t = timesteps[i]
                # 1. batchify into sizes that fit into the GPU
                x_batch = batchfy(x, batch_size)
                # 2. Run PC step for each batch
                x_agg = list()
                for idx, x_batch_sing in enumerate(x_batch):
                    x_batch_sing, _ = predictor_denoise_update_fn(model, data, x_batch_sing, t)
                    x_batch_sing, _ = corrector_denoise_update_fn(model, data, x_batch_sing, t)
                    x_agg.append(x_batch_sing)
                # 3. Aggregate to run ADMM TV
                x = torch.cat(x_agg, dim=0)
                # 4. Run ADMM TV
                x, x_mean = mc_update_fn(x, measurement=measurement)

                if save_progress:
                    if (i % 50) == 0:
                        print(f'iter: {i}/{sde.N}')
                        plt.imsave(save_root / 'recon' / 'progress' / f'progress{i}.png', clear(x_mean[0:1]),
                                   cmap='gray')
            # Apply one final normalized back-projection correction.
            if final_consistency:
                x, x_mean = kaczmarz(x, x, measurement, lamb=1.0, norm_const=norm_const)

            return inverse_scaler(x_mean if denoise else x)

    return pc_radon