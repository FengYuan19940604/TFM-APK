# ============================================================
#  TFM 成像核心算法（纯 numpy 实现，无 scipy / matplotlib 依赖）
#  用于 Android 打包（Kivy + Buildozer）
#
#  数据规格：5~6通道（CH0~CH5均为传感器）/ 每传感器敲1次 = n_ch炮/文件
#  多文件拼接：相邻文件重叠2传感器
#  参考：刘昊等. 混凝土结构缺陷低频超声阵列成像方法研究.
#  铁道科学与工程学报, 2025, 22(10): 4712-4725.
# ============================================================

import numpy as np
import os
import json


# ============================================================
#  常量 / 默认参数
# ============================================================
OVERLAP = 2            # 相邻文件重叠传感器数（固定=2）
DEF_SPACING = 0.3      # 默认道间距 (m)
DEF_VEL = 4363.0       # 默认弹性波速 (m/s)
DEF_PPD = 0.0          # 默认脉冲峰值延迟 (s)
DEF_PRE = 100          # 默认预触发点数


# ============================================================
#  信号处理（numpy 替代 scipy）
# ============================================================
def hilbert(x, axis=-1):
    """解析信号的虚部（Hilbert 变换），numpy FFT 实现。

    等价于 scipy.signal.hilbert(x, axis=axis)。
    """
    N = x.shape[axis]
    Xf = np.fft.fft(x, axis=axis)
    # 频域滤波系数
    h = np.zeros(N, dtype=Xf.dtype)
    if N % 2 == 0:
        h[0] = 1.0
        h[N // 2] = 1.0
        h[1:N // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(N + 1) // 2] = 2.0
    # 广播到 x 的形状
    shape = [1] * x.ndim
    shape[axis] = N
    h = h.reshape(shape)
    return np.fft.ifft(Xf * h, axis=axis)


def hilbert_envelope(x, axis=-1):
    """Hilbert 包络（幅值）。"""
    return np.abs(hilbert(x, axis=axis))


def bandpass_filter(x, fs, f_low=500.0, f_high=15000.0, axis=-1):
    """FFT 频域带通滤波（numpy 实现，替代 scipy.signal.butter+sosfilt）。"""
    f_low = max(f_low, 0.0)
    nyq = fs / 2.0
    f_high = min(f_high, nyq - 1.0)
    N = x.shape[axis]
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mask = (freqs >= f_low) & (freqs <= f_high)
    Xf = np.fft.rfft(x, axis=axis)
    shape = [1] * x.ndim
    shape[axis] = mask.shape[0]
    Xf = Xf * mask.reshape(shape)
    return np.fft.irfft(Xf, n=N, axis=axis)


# ============================================================
#  数据读取
# ============================================================
def read_oac3(file_path):
    """读取 .oac3 配置。"""
    oac_path = os.path.splitext(file_path)[0] + '.oac3'
    cfg = {
        'dr_itol': 2048, 'fSmpClk': 2.0, 'iChlNu': 6,
        'iDelayNum': DEF_PRE, 'ffRcvDistence': [0.1] * 6
    }
    if os.path.exists(oac_path):
        try:
            with open(oac_path, 'r', encoding='utf-8') as f:
                jd = json.load(f)
            cdc = jd.get('CDC', {})
            cfg['dr_itol'] = cdc.get('dr_itol', cdc.get('iScan', 2048))
            cfg['fSmpClk'] = cdc.get('fSmpClk', 2.0)
            cfg['iChlNu'] = cdc.get('iChlNu', 6)
            cfg['iDelayNum'] = cdc.get('iDelayNum', DEF_PRE)
            cfg['ffRcvDistence'] = cdc.get('comp_Dvc', {}).get(
                'ffRcvDistence', [0.1] * 6)
        except Exception:
            pass
    return cfg


def read_scatter_data(file_path, cfg, pre=None):
    """读取散射波 .bin 数据（样品交织）。自动探测通道数（6→5）。

    返回: data[n_shots, n_ch, n_smp-pre], n_ch, pre, fs
    """
    n_smp = cfg.get('dr_itol', cfg.get('iScan', 2048))
    pre = pre if pre is not None else cfg.get('iDelayNum', DEF_PRE)
    fs = 1e6 / cfg['fSmpClk']

    raw = np.fromfile(file_path, dtype=np.float32)

    n_ch = None
    for nc in [6, 5]:
        if len(raw) % (nc * n_smp) == 0:
            n_ch = nc
            break
    if n_ch is None:
        raise ValueError('数据长度无法匹配5或6通道规格')

    per_shot = n_ch * n_smp
    total_shots = len(raw) // per_shot
    n_use = min(total_shots, n_ch)
    data = raw[:n_use * per_shot].reshape(n_use, n_smp, n_ch).transpose(0, 2, 1)
    data = data[:, :, pre:]
    return data, n_ch, pre, fs


def build_sensor_positions(n_sensors, base_offset=0.0, spacing=DEF_SPACING):
    """CH0→CHn 升序排列。返回 sensor_x, shot_x。"""
    sensor_x = np.arange(n_sensors, dtype=float) * spacing + base_offset
    shot_x = sensor_x.copy()
    return sensor_x, shot_x


# ============================================================
#  TFM 成像
# ============================================================
def tfm_image(data, shot_x, rcvr_x, x_grid, z_grid, vel, dt, ppd=0.0,
              use_hilbert=True, use_scf=False, use_envelope=2,
              scf_power=1.0, wave_mode=0, gate_width=500e-6):
    """全聚焦成像（单文件）。

    wave_mode: 0=全波, 1=P波(vel设Vp), 2=S波(vel改设Vs)
    gate_width: 走时窗半宽(s)，压制非目标波型及面波
    """
    n_shots, n_ch, n_samples = data.shape
    nx = len(x_grid)
    nz = len(z_grid)

    if use_hilbert:
        proc_data = hilbert_envelope(data, axis=-1)
    else:
        proc_data = data.copy()
    m_file = np.max(proc_data)
    if m_file > 0:
        proc_data /= m_file

    # 波型时间窗
    if wave_mode in (1, 2):
        t_axis = np.arange(n_samples) * dt
        for i_s in range(n_shots):
            for i_r in range(n_ch):
                d = abs(shot_x[i_s] - rcvr_x[i_r])
                if d < 1e-6:
                    continue
                t0 = d / vel
                dt_rel = (t_axis - t0) / gate_width
                win = np.where(dt_rel <= 0,
                               np.exp(-dt_rel ** 2),
                               np.exp(-dt_rel))
                proc_data[i_s, i_r] *= win

    # 旅行时间
    all_tau = np.empty((n_shots * n_ch, nz, nx), dtype=np.float64)
    pair_idx = 0
    for i_src in range(n_shots):
        for i_rcv in range(n_ch):
            d_s = np.sqrt((x_grid - shot_x[i_src])**2 + z_grid[:, None]**2)
            d_r = np.sqrt((x_grid - rcvr_x[i_rcv])**2 + z_grid[:, None]**2)
            all_tau[pair_idx] = (d_s + d_r) / vel + ppd
            pair_idx += 1

    # 线性插值
    idx_f = all_tau / dt
    idx_lo = np.floor(idx_f).astype(np.int32)
    frac = idx_f - idx_lo
    valid = (idx_lo >= 0) & (idx_lo < n_samples - 1)
    np.clip(idx_lo, 0, n_samples - 2, out=idx_lo)
    idx_hi = idx_lo + 1

    image = np.zeros((nz, nx), dtype=np.float64)
    weight = np.zeros((nz, nx), dtype=np.float64)
    pair_idx = 0
    for i_src in range(n_shots):
        for i_rcv in range(n_ch):
            amp = proc_data[i_src, i_rcv, idx_lo[pair_idx]] * (1 - frac[pair_idx]) \
                + proc_data[i_src, i_rcv, idx_hi[pair_idx]] * frac[pair_idx]
            image += amp
            weight += valid[pair_idx].astype(np.float64)
            pair_idx += 1

    np.divide(image, weight, out=image, where=(weight > 0))

    # SCF
    if use_scf:
        idx_sign = np.round(all_tau / dt).astype(np.int32)
        np.clip(idx_sign, 0, n_samples - 1, out=idx_sign)
        b_total = np.zeros((nz, nx), dtype=np.float64)
        pair_idx = 0
        for i_src in range(n_shots):
            for i_rcv in range(n_ch):
                s = np.where(data[i_src, i_rcv, idx_sign[pair_idx]] >= 0, 1.0, -1.0)
                b_total += s
                pair_idx += 1
        b_mean = b_total / (n_shots * n_ch)
        scf = b_mean ** (2 * scf_power)
        image = image * scf

    if use_envelope == 1:
        image = np.abs(image)
    elif use_envelope == 2:
        img_abs = np.abs(image)
        m = img_abs.max()
        if m > 0:
            image = img_abs / m
    return image


# ============================================================
#  多文件拼接成像（封装完整流程）
# ============================================================
def run_imaging(file_list, params):
    """多文件拼接成像。

    file_list: [(data, shot_x, sensor_x), ...]
    params: dict，包含 vel, ppd, x_start, x_end, z_start, z_end,
            nx, nz, use_hilbert, use_scf, scf_power, use_envelope,
            norm_each, wave_mode, gate_width

    返回: image[nz, nx], x_grid, z_grid
    """
    x_grid = np.linspace(params['x_start'], params['x_end'], params['nx'])
    z_grid = np.linspace(params['z_start'], params['z_end'], params['nz'])
    vel = params['vel']
    dt = params['dt']
    ppd = params['ppd']

    use_env = params['use_envelope']
    norm_each = params['norm_each']

    if norm_each:
        per_env = 2
    elif use_env == 0:
        per_env = 0
    else:
        per_env = 1

    file_images = []
    for (data_i, shot_x_i, sensor_x_i) in file_list:
        d = data_i.astype(np.float64)
        if params.get('filter_on', True):
            d = bandpass_filter(d, params['fs'],
                                params.get('f_low', 500.0),
                                params.get('f_high', 15000.0), axis=-1)
        img_i = tfm_image(
            d, shot_x_i, sensor_x_i,
            x_grid, z_grid, vel, dt, ppd=ppd,
            use_hilbert=params['use_hilbert'],
            use_scf=params['use_scf'],
            use_envelope=per_env,
            scf_power=params['scf_power'],
            wave_mode=params['wave_mode'],
            gate_width=params['gate_width'],
        )
        file_images.append(img_i)

    stacked = np.stack(file_images, axis=0)
    mask = (np.abs(stacked) > 1e-12).astype(np.float64)
    count = np.sum(mask, axis=0)
    image = np.sum(stacked, axis=0)
    np.divide(image, count, out=image, where=(count > 0))

    if not norm_each:
        if use_env == 1:
            image = np.abs(image)
        elif use_env == 2:
            img_abs = np.abs(image)
            m = img_abs.max()
            if m > 0:
                image = img_abs / m

    return image, x_grid, z_grid
