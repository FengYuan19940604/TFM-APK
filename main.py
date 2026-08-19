# ============================================================
#  TFM 成像 - Android 版（Kivy）
#  纯 numpy 算法 + Kivy UI，可打包 APK
#
#  运行: python main.py
#  打包: buildozer android debug
# ============================================================

import os
import threading
import numpy as np

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.spinner import Spinner
from kivy.uix.checkbox import CheckBox
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp

import tfm_core


# ============================================================
#  Colormap（内置 jet / hot，替代 matplotlib）
# ============================================================
def make_colormap(cmap='jet', n=256):
    if cmap == 'jet':
        anchors = np.array([
            [0, 0, 0.5], [0, 0, 1], [0, 0.5, 1], [0, 1, 1],
            [0.5, 1, 0.5], [1, 1, 0], [1, 0.5, 0], [1, 0, 0], [0.5, 0, 0],
        ])
    elif cmap == 'hot':
        anchors = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
        ])
    else:
        anchors = np.array([[0, 0, 0], [1, 1, 1]])
    xa = np.linspace(0, 1, len(anchors))
    xi = np.linspace(0, 1, n)
    lut = np.empty((n, 3))
    for i in range(3):
        lut[:, i] = np.interp(xi, xa, anchors[:, i])
    return lut


def img_to_rgba(img, lut, vmin=None, vmax=None):
    """numpy 图像 → RGBA uint8 数组 [h, w, 4]。"""
    img = np.asarray(img, dtype=np.float64)
    if vmin is None:
        vmin = float(np.min(img))
    if vmax is None:
        vmax = float(np.max(img))
    if vmax > vmin:
        norm = (img - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(img)
    norm = np.clip(norm, 0.0, 1.0)
    idx = (norm * (lut.shape[0] - 1)).astype(np.int32)
    rgb = lut[idx] * 255.0
    alpha = np.full(img.shape, 255.0)
    rgba = np.dstack([rgb, alpha]).astype(np.uint8)
    return rgba


def array_to_texture(rgba):
    h, w, _ = rgba.shape
    tex = Texture.create(size=(w, h), colorfmt='rgba')
    tex.blit_buffer(rgba.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
    tex.flip_vertical()
    return tex


# ============================================================
#  文件扫描
# ============================================================
def scan_bin_files(directory):
    """扫描目录下的 .bin 文件，按文件名排序。"""
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if f.lower().endswith('.bin')]
    files.sort()
    return [os.path.join(directory, f) for f in files]


DEFAULT_DIRS = [
    '/sdcard/Download',
    '/sdcard/DCIM',
    '/sdcard/Documents',
    os.getcwd(),
]


# ============================================================
#  主界面
# ============================================================
class TFMApp(App):
    title = '弹性波散射 TFM 成像'

    def build(self):
        self.file_list = []      # [(data, shot_x, sensor_x), ...]
        self.dt = 1.0 / 125000.0
        self.fs = 125000.0
        self.lut = make_colormap('jet')
        self.cmap = 'jet'

        root = BoxLayout(orientation='vertical', padding=dp(6), spacing=dp(6))

        # ---- 顶部：文件 ----
        top = BoxLayout(orientation='horizontal', size_hint_y=None, height=dp(44))
        self.btn_dir = Button(text='扫描文件', size_hint_x=0.35)
        self.btn_dir.bind(on_press=self._on_scan)
        self.lbl_file = Label(text='未加载数据', size_hint_x=1, font_size=dp(13))
        top.add_widget(self.btn_dir)
        top.add_widget(self.lbl_file)
        root.add_widget(top)

        # ---- 中部：成像图 ----
        self.img_view = Image(size_hint_y=1, allow_stretch=True, keep_ratio=False)
        root.add_widget(self.img_view)

        # 图例信息条
        self.lbl_legend = Label(text='', size_hint_y=None, height=dp(20),
                                font_size=dp(12), halign='center')
        root.add_widget(self.lbl_legend)

        # ---- 底部：参数（可滚动） ----
        sv = ScrollView(size_hint_y=0.62)
        self.param_box = GridLayout(cols=2, size_hint_y=None, spacing=dp(4),
                                    padding=dp(4))
        self.param_box.bind(minimum_height=self.param_box.setter('height'))
        self._build_params()
        sv.add_widget(self.param_box)
        root.add_widget(sv)

        # ---- 执行按钮 ----
        self.btn_run = Button(text='执行成像', size_hint_y=None, height=dp(52),
                              background_color=(0.2, 0.6, 1, 1))
        self.btn_run.bind(on_press=self._on_run)
        root.add_widget(self.btn_run)

        return root

    def _build_params(self):
        """构建参数输入项。"""
        def add(label_text, widget, hint=''):
            lbl = Label(text=label_text, size_hint_y=None, height=dp(36),
                        font_size=dp(14), halign='left', valign='middle')
            lbl.bind(size=lambda *a: setattr(lbl, 'text_size', lbl.size))
            self.param_box.add_widget(lbl)
            self.param_box.add_widget(widget)

        def num(text, val):
            t = TextInput(text=str(val), multiline=False,
                          input_filter='float', size_hint_y=None, height=dp(36),
                          font_size=dp(14))
            return t

        self.in_vel = num('波速(m/s)', tfm_core.DEF_VEL)
        self.in_spacing = num('道间距(m)', tfm_core.DEF_SPACING)
        self.in_ppd = num('PPD(μs)', 0.0)
        self.in_x0 = num('X起点(m)', -0.2)
        self.in_x1 = num('X终点(m)', 3.0)
        self.in_z0 = num('Z起点(m)', 0.0)
        self.in_z1 = num('Z终点(m)', 1.5)
        self.in_nx = num('像素X', 200)
        self.in_nz = num('像素Z', 260)
        self.in_f_low = num('低截(Hz)', 500.0)
        self.in_f_high = num('高截(Hz)', 15000.0)
        self.in_scf_power = num('SCF强度', 1.0)
        self.in_gate = num('窗宽(μs)', 500.0)

        add('波速 (m/s):', self.in_vel)
        add('道间距 (m):', self.in_spacing)
        add('PPD (μs):', self.in_ppd)
        add('X起点 (m):', self.in_x0)
        add('X终点 (m):', self.in_x1)
        add('Z起点 (m):', self.in_z0)
        add('Z终点 (m):', self.in_z1)
        add('像素 X×Z:', self.in_nx)
        add('像素 Z:', self.in_nz)
        add('低截 (Hz):', self.in_f_low)
        add('高截 (Hz):', self.in_f_high)
        add('SCF强度:', self.in_scf_power)
        add('窗宽 (μs):', self.in_gate)

        # 波型选择
        self.sp_wave = Spinner(
            text='全波', values=('全波', 'P波', 'S波'),
            size_hint_y=None, height=dp(36), font_size=dp(14))
        add('波型:', self.sp_wave)

        # 输出类型
        self.sp_env = Spinner(
            text='归一化', values=('原始', '绝对值', '归一化'),
            size_hint_y=None, height=dp(36), font_size=dp(14))
        add('输出:', self.sp_env)

        # 色标
        self.sp_cmap = Spinner(
            text='jet', values=('jet', 'hot', 'gray'),
            size_hint_y=None, height=dp(36), font_size=dp(14))
        add('色标:', self.sp_cmap)

        # 开关
        add('Hilbert包络:', self._check(True, 'chk_hilbert'))
        add('SCF加权:', self._check(True, 'chk_scf'))
        add('带通滤波:', self._check(True, 'chk_filter'))
        add('各段归一化拼接:', self._check(False, 'chk_norm_each'))

    def _check(self, val, name):
        cb = CheckBox(active=val, size_hint_y=None, height=dp(36))
        setattr(self, name, cb)
        return cb

    # ---- 文件扫描与加载 ----
    def _on_scan(self, *a):
        found = None
        for d in DEFAULT_DIRS:
            fs = scan_bin_files(d)
            if fs:
                found = (d, fs)
                break
        if not found:
            self._popup('未找到 .bin 文件\n请将数据放入 Download 目录')
            return
        d, fs = found
        self._load_all(d, fs)

    def _load_all(self, d, fs):
        """加载目录下所有 .bin 文件并拼接（按文件名排序，重叠2通道）。"""
        try:
            spacing = self._f(self.in_spacing)
            file_list = []
            base = 0.0
            for p in fs:
                cfg = tfm_core.read_oac3(p)
                data, n_ch, pre, fs_hz = tfm_core.read_scatter_data(p, cfg)
                sensor_x, shot_x = tfm_core.build_sensor_positions(
                    n_ch, base_offset=base, spacing=spacing)
                file_list.append((data, shot_x, sensor_x))
                base += (n_ch - tfm_core.OVERLAP) * spacing
                self.fs = fs_hz
                self.dt = 1.0 / fs_hz
            self.file_list = file_list
            n_ch = file_list[0][0].shape[1]
            total_len = file_list[-1][1][-1]
            self.lbl_file.text = ('%d 个文件, %d通道, 覆盖 %.2f m' % (
                len(file_list), n_ch, total_len))
            # 自动扩展 X 范围
            self.in_x0.text = '%.1f' % (-0.1)
            self.in_x1.text = '%.1f' % (total_len + 0.1)
        except Exception as e:
            self.lbl_file.text = '加载失败: %s' % str(e)

    def _popup(self, msg):
        popup = Popup(title='提示', content=Label(text=msg),
                      size_hint=(0.85, 0.4))
        popup.open()

    # ---- 成像执行 ----
    def _on_run(self, *a):
        if not self.file_list:
            self._toast('请先扫描并加载文件')
            return
        self.btn_run.text = '计算中...'
        self.btn_run.disabled = True
        threading.Thread(target=self._do_imaging, daemon=True).start()

    def _do_imaging(self):
        try:
            params = self._collect_params()
            image, x_grid, z_grid = tfm_core.run_imaging(self.file_list, params)
            self.image = image
            self.x_grid = x_grid
            self.z_grid = z_grid
            Clock.schedule_once(self._show_image, 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self._toast('成像失败: %s' % str(e)), 0)
        finally:
            Clock.schedule_once(lambda dt: self._reset_run_btn(), 0)

    def _collect_params(self):
        wave_map = {'全波': 0, 'P波': 1, 'S波': 2}
        env_map = {'原始': 0, '绝对值': 1, '归一化': 2}
        return {
            'vel': self._f(self.in_vel),
            'ppd': self._f(self.in_ppd) * 1e-6,
            'x_start': self._f(self.in_x0),
            'x_end': self._f(self.in_x1),
            'z_start': self._f(self.in_z0),
            'z_end': self._f(self.in_z1),
            'nx': self._i(self.in_nx),
            'nz': self._i(self.in_nz),
            'dt': self.dt,
            'fs': self.fs,
            'use_hilbert': self.chk_hilbert.active,
            'use_scf': self.chk_scf.active,
            'scf_power': self._f(self.in_scf_power),
            'use_envelope': env_map[self.sp_env.text],
            'norm_each': self.chk_norm_each.active,
            'wave_mode': wave_map[self.sp_wave.text],
            'gate_width': self._f(self.in_gate) * 1e-6,
            'f_low': self._f(self.in_f_low),
            'f_high': self._f(self.in_f_high),
            'filter_on': self.chk_filter.active,
        }

    def _show_image(self, dt):
        cmap = self.sp_cmap.text
        if cmap != self.cmap:
            self.cmap = cmap
            self.lut = make_colormap(cmap)
        rgba = img_to_rgba(self.image, self.lut)
        tex = array_to_texture(rgba)
        self.img_view.texture = tex

        x0, x1 = self.x_grid[0], self.x_grid[-1]
        z0, z1 = self.z_grid[0], self.z_grid[-1]
        vmax = float(np.max(np.abs(self.image)))
        self.lbl_legend.text = ('X: %.2f ~ %.2f m   Z: %.2f ~ %.2f m   '
                                'max=%.3f' % (x0, x1, z0, z1, vmax))

    def _reset_run_btn(self):
        self.btn_run.text = '执行成像'
        self.btn_run.disabled = False

    def _toast(self, msg):
        self.lbl_legend.text = msg

    # ---- 工具 ----
    def _f(self, t):
        try:
            return float(t.text.strip())
        except Exception:
            return 0.0

    def _i(self, t):
        try:
            return int(float(t.text.strip()))
        except Exception:
            return 100


if __name__ == '__main__':
    TFMApp().run()
