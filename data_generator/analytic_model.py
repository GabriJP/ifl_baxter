from functools import cache
from typing import Literal
from typing import overload

import numpy as np
import pybullet as p

from utils import F64_A
from utils import Number

model_left = "data_generator/baxter_common/baxter_description/urdf/baxter_left_no_gripper.urdf"
model_right = "data_generator/baxter_common/baxter_description/urdf/baxter_right_no_gripper.urdf"

max_joint_velocity = [2.0] * 4 + [4.0] * 3


class Limb:
    def __init__(self, name: Literal["left", "right"]):
        self.name = name

        if self.name not in ["left", "right"]:
            msg = "Arg 'name' must be 'right' or 'left'."
            raise ValueError(msg)

        #
        self.joint_indices = np.arange(1, 8)
        self.joint_idx_dof = np.arange(7)
        # The robot's "endpoint" is defined as the <side>_gripper tf frame.
        self.ee = self.ee_index = 8  # 42: w2, 48: gripper base, 49: end_point

        if self.name == "left":
            self._joint_names = ["left_s0", "left_s1", "left_e0", "left_e1", "left_w0", "left_w1", "left_w2"]
        else:
            self._joint_names = ["right_s0", "right_s1", "right_e0", "right_e1", "right_w0", "right_w1", "right_w2"]

        # Real2Sim conversions
        self._name_2_idx_dof = dict(zip(self._joint_names, self.joint_idx_dof, strict=False))
        self._idx_2_name = dict(zip(self.joint_idx_dof, self._joint_names, strict=False))

        for i, j in enumerate(self.joint_indices):
            p.changeDynamics(self.baxter_id, j, maxJointVelocity=max_joint_velocity[i])

    @property
    @cache  # noqa: B019
    def baxter_id(self) -> int:
        if self.name == "left":
            path_model = model_left
        elif self.name == "right":
            path_model = model_right
        else:
            raise ValueError

        baxter_id = p.loadURDF(path_model, useFixedBase=True)

        p.resetBasePositionAndOrientation(baxter_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # endpoint (49) gripper
        p.setGravity(0.0, 0.0, -9.8)
        # p.setGravity(0., 0., 0.)

        for i in range(p.getNumJoints(baxter_id)):
            p.changeDynamics(baxter_id, i, linearDamping=0, angularDamping=0, jointDamping=0)

        return baxter_id

    def inv_dyn(self, ang: F64_A, vel: F64_A, acl: F64_A) -> F64_A:
        exp_ang = [0] * 7
        exp_vel = [0] * 7
        exp_acl = [0] * 7

        for idx, idx_dof in enumerate(self.joint_idx_dof):
            exp_ang[idx_dof] = ang[idx]
            exp_vel[idx_dof] = vel[idx]
            exp_acl[idx_dof] = acl[idx]

        torque = np.array(p.calculateInverseDynamics(self.baxter_id, exp_ang, exp_vel, exp_acl))

        return torque[self.joint_idx_dof]

    def get_torque_traj(self, pos: F64_A, vel: F64_A, acl: F64_A) -> F64_A:
        return np.array([self.inv_dyn(*pva) for pva in zip(pos, vel, acl, strict=False)])

    def reset_joints(self, ang_joint: F64_A, vel_joint: F64_A | None = None) -> F64_A:
        if vel_joint is None:
            vel_joint = np.zeros(7, dtype=np.float64)
        max_force = 0.0
        mode = p.VELOCITY_CONTROL

        for j in range(7):
            p.resetJointState(self.baxter_id, self.joint_indices[j], ang_joint[j], targetVelocity=vel_joint[j])
            p.setJointMotorControl2(self.baxter_id, self.joint_indices[j], controlMode=mode, force=max_force)
        return np.concatenate((ang_joint, vel_joint))

    def set_torque_joints(self, torque: F64_A, spring: bool = True) -> F64_A:
        torque_w_spring = np.copy(torque)

        if spring:
            th_s1 = p.getJointState(self.baxter_id, self.joint_indices[1])[0]
            torque_w_spring[1] += self.clc_torque_spring(th_s1)

        for j in range(7):
            p.setJointMotorControl2(
                bodyIndex=self.baxter_id,
                jointIndex=self.joint_indices[j],
                controlMode=p.TORQUE_CONTROL,
                force=torque_w_spring[j],
            )
        p.stepSimulation()
        new_state = p.getJointStates(self.baxter_id, self.joint_indices)
        pos_vel = np.empty((7, 2))
        for joint_i, joint_i_state in zip(pos_vel, new_state, strict=False):
            joint_i[:] = [joint_i_state[0], joint_i_state[1]]

        # Read Sensor States:
        return pos_vel

    @overload
    def get_ee_state(
        self, clc_velocity: Literal[True]
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]: ...

    @overload
    def get_ee_state(self, clc_velocity: Literal[False]) -> tuple[float, float, float]: ...

    @overload
    def get_ee_state(self) -> tuple[float, float, float]: ...

    def get_ee_state(
        self, clc_velocity: bool = False
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]] | tuple[float, float, float]:
        if clc_velocity:
            link_state = p.getLinkState(self.baxter_id, self.ee, computeLinkVelocity=clc_velocity)
            return link_state[4], link_state[5]
        return p.getLinkState(self.baxter_id, self.ee)[4]

    def clc_gravity_compensation(self, pos: F64_A, vel: F64_A) -> F64_A:
        return self.inv_dyn(pos, vel, np.zeros(7, dtype=np.float64))

    @staticmethod
    def clc_torque_spring(th: Number, rad: bool = True) -> float:
        l_ = 165.0  # 151 165
        s1 = 69
        ls = 30  # 38 30
        s0 = 270
        k = 9.6

        th_r = th if rad else th * np.pi / 180.0

        l1 = s0 - ls * np.sin(th_r)
        l2 = s1 + ls * np.cos(th_r)

        ht = (l1**2 + l2**2) ** 0.5

        x = ht - l_

        bt = np.arctan(l1 / l2)
        alph = bt + th_r

        fx = k * x

        ts = ls * fx * np.sin(alph) / 1000.0

        return -ts

    def clc_compensations(self, pos: F64_A, vel: F64_A) -> F64_A:
        torque_comp = self.clc_gravity_compensation(pos, vel)
        torque_comp[1] -= self.clc_torque_spring(pos[1])
        return torque_comp


limb_left = Limb("left")
limb_right = Limb("right")
