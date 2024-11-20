from math import pi
from typing import Callable, List
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict
from qutip import mesolve, expect

class Schedule:
    def __init__(self, sampling_rate=0.2) -> None:
        self.time = 0.
        self.sampling_rate = sampling_rate
        self._sequence = []

    def __add__(self, sched):

        if self.sampling_rate != sched.sampling_rate:
            raise Exception("The sampling rates of two schedules are not the same.")
        else:
            sched_added = Schedule(sampling_rate=self.sampling_rate)
            sched_added.append_sched(self)
            sched_added.append_sched(sched)

        return sched_added

    @property
    def sequence(self):
        return np.array(self._sequence)
    
    @property
    def time_sequence(self):
        return np.linspace(0., self.time, int(np.round(self.time / self.sampling_rate)))
        
    def append(self, wave_form, t, args):
        _times = np.linspace(0, t, int(np.round(t / self.sampling_rate)))
        for _t in _times:
            _amp = wave_form(_t, args)
            self._sequence.append(_amp)
        self.time = self.time + t
    
    def append_list(self, amps_list: list):
        self._sequence.extend(amps_list)
        self.time = self.time + self.sampling_rate * len(amps_list)

    def append_sched(self, sched):
        if self.sampling_rate != sched.sampling_rate:
            raise Exception("The sampling rates of two schedules are not the same.")
        else:
            self.time = self.time + sched.time
            self._sequence.extend(sched._sequence)

    def plot(self, time=None, freq=None, label=None, fill=True):
        if time is None:
            plot_y = np.array(self.sequence)
            plot_x = np.linspace(0, self.time, len(self.sequence))
            print(self.time)
            print(len(self.sequence))
            plt.plot(plot_x, plot_y, '-o')
            label_plot_x = 0.
        else:
            idx_time = np.arange(np.round(time[0] / self.sampling_rate), np.round(time[1] / self.sampling_rate), 1, dtype=int)
            plot_y = np.array(self.sequence[idx_time])
            whole_time = np.linspace(0., self.time, len(self.sequence))
            plot_x = whole_time[idx_time]
            print(plot_x[0], plot_x[-1])
            plt.plot(plot_x, plot_y)
            label_plot_x = plot_x[0]

        if freq is not None:
            for _freq, _label in zip(freq, label):
                plt.axhline(y=_freq, color='r', linestyle='-')
                plt.text(x=label_plot_x, y=_freq + 0.001, s=_label)
                plt.text(x=plot_x[-1], y=_freq + 0.001, s=f'{_freq}')

        else:
            pass
        if fill is True:
            plt.fill_between(plot_x, plot_y, alpha=0.1)
        else:
            pass
        plt.xlabel('ns')
        plt.ylabel('GHz')
        plt.show()

def square(t, args: Dict):
    return args['a']

def ramp(t, args: Dict):
    if args['t'] != 0.:
        if t <= args['t']:
            _k = (args['a1'] - args['a0']) / args['t']
            _amp = args['a0'] + _k * t
        else:
            _amp = 0.
    else:
        _amp = 0.
    return _amp


def ramp_array(t, args: Dict):
    _k = (args['a1'] - args['a0']) / args['t']
    _amp = args['a0'] + _k * t
    return _amp
 

def right_half_gaussian(t, args: Dict):
    if t <= 4 * args['sig']:
        y = (args['a0'] - args['a1']) * np.exp(-t ** 2 / (2 * args['sig'] ** 2)) + args['a1']
    else: 
        y = 0.
    return y

def left_half_gaussian(t, args: Dict):
    if t > 4 * args['sig']:
        y = 0.
    else:
        y = (args['a1'] - args['a0']) * np.exp(-(t - 4 * args['sig']) ** 2 / (2 * args['sig'] ** 2)) + args['a0']
    return y

def opt_fun(t, args: Dict):
    # args['t'], args['y0'], args['y1'], args['k'], args['dt']
    end_idx = np.ceil(args['t'] / args['dt'])
    now_idx = np.ceil(t / args['dt'])
    if now_idx > end_idx:
        y = 0.
    else:
        _y = args['y0']
        _y1 = args['y1']
        _k = args['k']
        _dt = args['dt']
        for _ in range(now_idx):
            _y = _y + _k * abs(_y - _y1) * _dt
        y = _y
    return y

def plot_bar(x_range, y_range, Z, label_bar='', label_x='', label_y='', title='', figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x_range.max(), x_range.min(), y_range.min(), y_range.max()), origin='lower', aspect='auto', vmin=0., vmax=1.)
    plt.colorbar(label=label_bar)  # Add colorbar with label
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    # x_ticks = np.arange(0, 10, 2)  # 设置刻度位置，每隔2个单位显示一次
    x_labels = ['0', '2', '4', '6', '8']  # 自定义刻度标签
    plt.xticks(labels=x_labels)
    plt.title(title)  
    # Show plot
    plt.tight_layout()
    plt.show()

def plot_bar2(x_range, y_range, Z, label_bar='', label_x='', label_y='', title='', figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', aspect='auto', vmin=0., vmax=1.)
    plt.colorbar(label=label_bar)  # Add colorbar with label
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    plt.title(title)  
    # Show plot
    plt.tight_layout()
    plt.show()

def find_eigen(barestate, eigenstates, eigenenergies):
    overlap_list = []
    for state in eigenstates:
        overlap_list.append(abs(barestate.overlap(state)))
    max_overlap_idx = np.argmax(overlap_list)
    return eigenstates[max_overlap_idx], eigenenergies[max_overlap_idx], max_overlap_idx

def run(H0, Ht, times, psi_init, e_ops, eigen_idx):
    states = mesolve([H0, Ht], psi_init, times).states
    e_vals = [np.zeros(shape=len(states)) for _ in range(len(e_ops))]
    if isinstance(eigen_idx, int):
        over_lap = np.zeros(shape=len(states))
    else:
        over_lap = np.zeros(shape=[len(eigen_idx), len(states)])
    
    for idx in range(len(states)):
        for op, e_val in zip(e_ops, e_vals):
            e_val[idx] = expect(op, state=states[idx])
        H_eigen = H0 + Ht[0] * Ht[1][idx]
        if isinstance(eigen_idx, int):
            eigenstate = H_eigen.eigenstates()[1][eigen_idx]
            over_lap[idx] = expect(oper=eigenstate * eigenstate.dag(), state=states[idx])
        else:
            eigenstates = H_eigen.eigenstates()[1]
            for iter_idx, e_idx in np.ndenumerate(eigen_idx):
                eigenstate = eigenstates[e_idx]
                over_lap[iter_idx][idx] = expect(oper=eigenstate * eigenstate.dag(), state=states[idx])
    return e_vals, over_lap

def local_analytic(t, a, g, c):
    numerator =  - 8 * g * (a * g * t + c)
    denominator = np.sqrt(abs(1 - 16 * (a * g * t) ** 2 - 32 * a * g * t * c - 16 * c ** 2))
    y = numerator / denominator
    return y

def generate_local_adiabatic_pulse(g, T, y0, yt, dt = 0.1):
    c = - y0 / np.sqrt(64 * g ** 2 + 16 * y0 ** 2)
    a = (- 4 * c * (4 * g ** 2 + yt ** 2) - yt * np.sqrt(4 * g ** 2 + yt ** 2)) / (4 * g * T * (4 * g ** 2 + yt ** 2))
    times = np.linspace(0, T, int(np.ceil(T / dt)))
    seq = local_analytic(times, a, g, c)

    return seq

def generate_local_adiabatic_fc(g, duration, f0, ft, fq, dt = 1.):

    y0 = fq - f0
    yt = fq - ft
    seq_fq_fc_detuning = generate_local_adiabatic_pulse(g, duration, y0, yt, dt=dt)
    seq_fc_f0_detuning = - seq_fq_fc_detuning + fq - f0

    return seq_fc_f0_detuning

def num_generate_local_adiabatic_pulse(amp_init, amp_final, dy_dt, args, dt):
    seq = [amp_init]
    diff = seq[-1] - amp_final
    _y = amp_init
    while diff < 0:
        _y = seq[-1] + dy_dt(_y, args) * dt
        diff = _y - amp_final
        seq.append(_y)
    if abs(seq[-1] - amp_final) >= 1e-2:
        seq[-1] = amp_final
    else:
        pass
    seq = np.array(seq, dtype=float)
    return seq

def mod_adiabatic_pulse(y, args):
    g = args['g']
    y0 = args['y0']
    yt = args['yt']
    scale = args['scale']
    init_slope = args['s0']
    y_mid = (y0 + yt) / 2
    dy_dt = scale * (init_slope + (1 - init_slope) * (1 - abs((y - y_mid) / (yt - y_mid))) ** 2) * (1 / g) * (y ** 2 + 4 * g ** 2) ** 1.5
    return dy_dt

def plot_spectrum(H0, Ht, dt):
    ht_op = Ht[0]
    amps = Ht[1]
    spectrum_array = np.zeros([H0.shape[0], len(amps)])
    for i, amp in np.ndenumerate(amps):
        h = H0 + ht_op * amp
        energies = h.eigenenergies()
        spectrum_array[:, i] = energies.reshape([len(energies), 1])
    times = np.linspace(0, len(amps) * dt, len(amps))
    return times, spectrum_array

def plot_bar_adv(x_range, y_range, Z, label_bar='', label_x='', label_y='', title='', x_line=None, vmin=0., vmax=1., figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x_range.max(), x_range.min(), y_range.min(), y_range.max()), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label=label_bar)  # Add colorbar with label
    size = 12
    plt.xlabel(label_x, size=size)
    plt.ylabel(label_y, size=size)
    if x_line is not None:
        plt.axvline(x=x_line, color='red', linestyle='--', linewidth=2)

        # 在 x 轴下方标出该值
        plt.text(x_line, y_range.min() - (y_range.max() - y_range.min()) * 0.05,  # x轴下方稍微偏移
                    f'{x_line:.2f}', color='red', ha='center')
    
    x_ticks = np.linspace(x_range.min(), x_range.max(), num=5)
    plt.xticks(ticks=x_ticks, labels=[f'{tick:.1f}' for tick in x_ticks])  # 可根据需要调整格式化
    plt.title(title, size=size)  
    # Show plot
    plt.tight_layout()
    plt.show()

def plot_bar2_adv(x_range, y_range, Z, label_bar='', label_x='', label_y='', title='', x_ticks=False, x_line=None, vmin=0., vmax=1., figsize=(5, 4)):
    plt.figure(figsize=figsize)
    plt.imshow(Z, extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), origin='lower', aspect='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label=label_bar)  # Add colorbar with label
    size = 12
    plt.xlabel(label_x, size=size)
    plt.ylabel(label_y, size=size)
    if x_line is not None:
        plt.axvline(x=x_line, color='red', linestyle='--', linewidth=2)

        # 在 x 轴下方标出该值
        plt.text(x_line, y_range.min() - (y_range.max() - y_range.min()) * 0.05,  # x轴下方稍微偏移
                    f'{x_line:.0f}', color='red', ha='center')
    if x_ticks is True:
        x_ticks = np.linspace(x_range.min(), x_range.max(), num=5)
        plt.xticks(ticks=x_ticks, labels=[f'{tick:.1f}' for tick in x_ticks])  # 可根据需要调整格式化
    plt.title(title, size=size)  
    # Show plot
    plt.tight_layout()
    plt.show()

def return_z_line(x_sweep, y_sweep, Z, x_idx):
    # 将数据存入 pandas DataFrame
    X, Y = np.meshgrid(x_sweep, y_sweep)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = Z.ravel()
    df = pd.DataFrame({'X': X_flat, 'Y': Y_flat, 'Z': Z_flat})
    filtered_data = df[df['X'] == x_sweep[x_idx]]
    y_values = filtered_data['Y'].values
    z_values = filtered_data['Z'].values
    return y_values, z_values


