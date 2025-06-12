import logging
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal

Number = int | float
F64_A = npt.NDArray[np.float64]

lim_max = np.array([1.7016, 1.047, 3.0541, 2.618, 3.059, 2.094, 3.059] + [2] * 4 + [4] * 3 + [50] * 4 + [15] * 3)
lim_min = np.array(
    [-1.7016, -2.147, -3.0541, -0.05, -3.059, -1.5707, -3.059] + [-2] * 4 + [-4] * 3 + [-50] * 4 + [-15] * 3
)
train_min = lim_min
train_ptp = lim_max - lim_min


def get_path_descriptor(array_angs: Iterable[F64_A]) -> tuple[F64_A, ...]:
    from data_generator import analytic_model as rbd

    descriptors: list[F64_A] = []
    limb_baxter = rbd.limb_left
    for array_ang in array_angs:
        xyz = []
        for ang in array_ang:
            limb_baxter.reset_joints(ang)  # type: ignore[arg-type]
            xyz_i = limb_baxter.get_ee_state()
            xyz.append(xyz_i)

        traj = np.array(xyz)
        descriptors.append(np.stack([traj.min(axis=0), traj.max(axis=0)], dtype=np.float64))

    return tuple(descriptors)


@dataclass
class Trajectory:
    name: str
    descriptor: F64_A
    path: F64_A

    def __len__(self) -> int:
        return len(self.path)

    def descriptor_dict(self) -> dict[str, float]:
        descriptor_names = [
            ["a_0", "a_1", "a_2"],
            ["b_0", "b_1", "b_2"],
        ]
        return {
            name: coord
            for names, point in zip(descriptor_names, self.descriptor.tolist(), strict=False)
            for name, coord in zip(names, point, strict=False)
        }


def sort_samples_inv(df: Iterable[Trajectory], max_len: int, step: int) -> tuple[F64_A, F64_A, list[int]]:
    x_q_qd = []
    y_tau = []
    long_traj = []

    for traj in df:
        train_df = norm_max_min(traj.path)
        x, y = samples_sort(train_df, max_len, step)
        x_q_qd.append(x)
        y_tau.append(y)
        long_traj.append(x.shape[0])

    return np.concatenate(x_q_qd, axis=0), np.concatenate(y_tau, axis=0), long_traj


def norm_max_min_pos_vel(data_orig: F64_A) -> F64_A:
    return (data_orig - train_min[:14]) / train_ptp[:14]


def norm_max_min(data_orig: F64_A) -> F64_A:
    return (data_orig - train_min) / train_ptp


def norm_max_min_torque(data_orig: F64_A) -> F64_A:
    return (data_orig - np.concatenate((train_min, train_min[14:21]))) / np.concatenate((train_ptp, train_ptp[14:21]))


def norm_max_min_sim_torque(data_orig: F64_A) -> F64_A:
    return (data_orig - train_min[14:21]) / train_ptp[14:21]


def denorm_max_min(data_norm: F64_A) -> F64_A:
    return data_norm * train_ptp[14:21] + train_min[14:21]


def denorm_max_min_pos_vel(data_norm: F64_A) -> F64_A:
    return data_norm * train_ptp[:14] + train_min[:14]


def samples_sort(
    samples: F64_A, size_wind: int, step: int = 1, output: bool = True, offset: int = 25
) -> tuple[F64_A, F64_A]:
    x_samples = []
    y_label = []
    ini_wind = offset - int(size_wind / 2)

    for i in range(0, len(samples) - 2 * offset, step):
        x_samples.append(samples[i + ini_wind : i + ini_wind + size_wind, 0:14])
        if output:
            y_label.append(samples[i + offset, 14:21])
        else:
            y_label.append(np.zeros((1, 7), dtype=np.float64))
    logging.info("nb sequences: %s", len(x_samples))

    return np.array(x_samples), np.array(y_label)


def samples_sort_torque(
    samples: F64_A, size_wind: int, step: int = 1, output: bool = True, offset: int = 25
) -> tuple[F64_A, F64_A]:
    x_samples = []
    y_label = []
    ini_wind = offset - int(size_wind / 2)

    for i in range(0, len(samples) - 2 * offset, step):
        x_samples.append(samples[i + ini_wind : i + ini_wind + size_wind, 0:21])
        if output:
            y_label.append(samples[i + offset, 21:28])
        else:
            y_label.append(np.zeros((1, 7), dtype=np.float64))
    logging.info("nb sequences: %s", len(x_samples))

    x = np.array(x_samples)
    y = np.array(y_label)

    return x, y


def sort_samples_forward(
    samples: F64_A, size_wind: int, step: int = 1, output: bool = True, offset: int = 25
) -> tuple[F64_A, F64_A]:
    x_samples = []
    y_label = []
    init_wind = offset
    final_wind = len(samples) - size_wind - offset

    for i in range(init_wind, final_wind, step):
        x_samples.append(samples[i : i + size_wind, :])
        if output:
            y_label.append(samples[i + 1 : i + size_wind + 1, :14])
        else:
            y_label.append(np.zeros((1, 14), dtype=np.float64))
    logging.info("nb sequences: %s", len(x_samples))

    x = np.array(x_samples)
    y = np.array(y_label)

    return x, y


def sort_samples(df: F64_A, max_len: int, step: int) -> tuple[F64_A, F64_A, list[int]]:
    x_pos_vel_torque = []
    y_pos_vel_real = []
    long_trajectories = []

    for tray in df:
        x, y = sort_samples_forward(tray, max_len, step=step, offset=0)  # type: ignore[arg-type]
        x_pos_vel_torque.append(x)
        y_pos_vel_real.append(y)
        long_trajectories.append(x.shape[0])

    return np.concatenate(x_pos_vel_torque, axis=0), np.concatenate(y_pos_vel_real, axis=0), long_trajectories


def idx_split_traj(
    size_data2split: Sequence[int], ratio_val: float = 0.20, num_split: int = 3
) -> npt.NDArray[np.bool_]:
    mask_idx_all = []

    for long_traj in size_data2split:
        mask_idx = np.zeros((num_split, long_traj))
        idx_cut = np.random.randint(0, long_traj, size=num_split)
        long_div = int(long_traj * ratio_val)
        idx_end = idx_cut + long_div

        for n_split, (id_c, id_e) in enumerate(zip(idx_cut, idx_end, strict=False)):
            logging.info("%s %s %s", n_split, id_c, id_e)
            if id_e >= long_traj:
                mask_idx[n_split, id_c:] = True
                mask_idx[n_split, : id_e - long_traj] = True
            else:
                mask_idx[n_split, id_c:id_e] = True

        mask_idx_all.append(np.bool_(mask_idx))

    return np.concatenate(mask_idx_all, axis=-1)


def get_train_val(x: F64_A, y: F64_A, mask_idx: npt.NDArray[np.bool_]) -> tuple[F64_A, F64_A, F64_A, F64_A]:
    x_val = x[mask_idx]
    y_val = y[mask_idx]
    x_train = x[np.logical_not(mask_idx)]
    y_train = y[np.logical_not(mask_idx)]

    return x_train, y_train, x_val, y_val


def normalization(data: F64_A, mean: float, std: float) -> F64_A:
    return (data - mean) / std


def d_normalization(data: F64_A, mean: float, std: float) -> F64_A:
    return (data * std) + mean


def load_dataset(path_data: str, filenames: Sequence[str], delta: int) -> list[F64_A]:
    df = [np.genfromtxt(path_data + filename, skip_header=1, delimiter=",")[:, 1:22] for filename in filenames]

    x = []

    for traj in df:
        pos_tmp, vel_tmp, acl_tmp = preprocess_traj(traj[:, :7], delta)
        data_tmp = np.concatenate([pos_tmp, vel_tmp, acl_tmp, traj[:, 14:21]], axis=-1)
        x.append(data_tmp)

    return x


def get_train_val_vector(
    data_x: F64_A, data_y: F64_A, long_traj: Sequence[int], val: float = 0.2
) -> tuple[F64_A, F64_A, F64_A, F64_A]:
    mask_idx = idx_split_traj(long_traj, num_split=1, ratio_val=val)

    return get_train_val(data_x, data_y, mask_idx[0])


def preprocess_traj(traj: F64_A, delta_t: float) -> tuple[F64_A, F64_A, F64_A]:
    sos = signal.butter(3, 5, fs=1.0 / delta_t, output="sos")

    pos_filter = signal.sosfiltfilt(sos, traj[:, :7], axis=0)
    vel_est = calculate_drv(pos_filter, delta_t)
    acl_est = calculate_drv(vel_est, delta_t)

    return pos_filter, vel_est, acl_est


def calculate_drv(q_vel: F64_A, delta_t: float) -> F64_A:
    q_acl = np.zeros(q_vel.shape)

    q_acl[0] = (q_vel[1] - q_vel[0]) / delta_t

    q_acl[1:-1] = (q_vel[2:] - q_vel[:-2]) / (2 * delta_t)

    q_acl[-1] = (q_vel[-1] - q_vel[-2]) / delta_t

    return q_acl


def sort_samples_inv_v2(
    samples: F64_A, size_wind: int, step: int = 1, output: bool = True, offset: int = 25
) -> tuple[F64_A, F64_A]:
    x_samples = []
    y_label = []
    ini_wind = offset - size_wind // 2

    for idx_ini in range(ini_wind, len(samples) - 2 * offset, step):
        x_samples.append(samples[idx_ini : idx_ini + size_wind, :21])
        if output:
            y_label.append(samples[idx_ini : idx_ini + size_wind, 21:])
        else:
            y_label.append(np.zeros((1, 7), dtype=np.float64))
    logging.info("nb sequences: %s", len(x_samples))

    return np.array(x_samples), np.array(y_label)


def sort_samples_v2(df: list[F64_A], max_len: int, step: int) -> tuple[F64_A, F64_A, list[int]]:
    x_pos_vel_torque = []
    y_pos_vel_real = []
    long_trajectories = []

    for tray in df:
        x, y = sort_samples_inv_v2(tray, max_len, step=step)
        x_pos_vel_torque.append(x)
        y_pos_vel_real.append(y)
        long_trajectories.append(x.shape[0])

    return np.concatenate(x_pos_vel_torque, axis=0), np.concatenate(y_pos_vel_real, axis=0), long_trajectories
