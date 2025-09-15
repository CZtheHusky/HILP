from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
import math
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi
from legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.h1.h1_config import H1Cfg
from legged_gym.legged_utils.observation_buffer import ObservationBuffer
from legged_gym.envs.base.curriculum import RewardThresholdCurriculum
from copy import deepcopy

def _polynomial_planer(t0, t1, x0, x1, v0=0, v1=0, a0=0, a1=0):

    T = t1 - t0
    h = x1 - x0
    k0 = x0
    k1 = v0
    k2 = 0.5 * a0
    k3 = (20 * h - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * (T ** 2)) / (2 * (T ** 3))
    k4 = (-30 * h + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * (T ** 2)) / (2 * (T ** 4))
    k5 = (12 * h - 6 * (v1 + v0) * T + (a1 - a0) * (T ** 2)) / (2* (T ** 5))
    coef = [k0, k1, k2, k3, k4, k5]

    return coef

class H1Robot(BaseTask):
    def __init__(self, cfg: H1Cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """

        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self._init_command_distribution(torch.arange(self.num_envs, device=self.device))

        self.action_scale = torch.tensor(self.cfg.control.action_scale, dtype=torch.float, device=self.device)

        if self.cfg.env.stack_history_obs:
            self.obs_buf_history = ObservationBuffer(
                self.num_envs, self.num_partial_obs,
                self.cfg.env.include_history_steps, self.device, zero_pad=False)

        if self.cfg.domain_rand.randomize_control_latency:
            include_obs_steps = int(self.cfg.domain_rand.control_latency_range[1]/self.dt) + 1
            self.sensor_delayed_dim = 6 + self.num_dof * 2  # IMU + joint encoder
            self.delayed_obs_buf = ObservationBuffer(self.num_envs, self.sensor_delayed_dim, include_obs_steps, self.device, zero_pad=False)
            
        self.init_done = True
    
    def reset(self):
        env_idxs = torch.arange(self.num_envs, device=self.device)
        self.reset_idx(env_idxs)

        if self.cfg.env.stack_history_obs:
            self.obs_buf_history.reset(env_idxs, self.obs_buf[:, :self.num_partial_obs])

        if self.cfg.domain_rand.randomize_control_latency:
            self.delayed_obs_buf.reset(env_idxs, self.obs_buf[:, :self.sensor_delayed_dim])

        obs, privileged_obs, _, _, _  = self.step(torch.zeros(
            self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def calculate_action(self, actions):
        self.actions = actions.clone()
        clip_actions = self.cfg.normalization.clip_actions
        return torch.clip(actions.clone(), -clip_actions, clip_actions).to(self.device)

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        actions_after_push = self.calculate_action(actions)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(actions_after_push).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time-elapsed_time>0:
                    time.sleep(sim_time-elapsed_time)
            
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        return self.partial_obs_buf, self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        foot_contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.bool_foot_contact = torch.logical_or(foot_contact, self.last_contacts)
        self.collision_states = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > self.cfg.rewards.max_collision_force
        self.last_contacts = foot_contact

        self.foot_pos_world = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.foot_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.foot_velocity_world = self.rigid_body_states[:, self.feet_indices, 7:10]
        foot_base_distance = self.foot_pos_world - (self.root_states[:, :3]).unsqueeze(1)
        self.foot_pos_b_h = torch.zeros_like(self.foot_pos_world)
        quat_yaw = self.base_quat.clone().view(-1, 4)
        quat_yaw[:, :2] = 0.
        quat_yaw = normalize(quat_yaw)
        for i in range(self.feet_indices.shape[0]):
            self.foot_pos_b_h[:,i,:] = quat_rotate_inverse(quat_yaw, foot_base_distance[:,i,:])

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations(env_ids) # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

    def check_termination(self):   # TODO ADD HEIGHT TERMINATION
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.large_ori_buf = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.large_ori_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # update level and type
        if self.cfg.terrain.curriculum:
            curriculum_mask = self.terrain_curriculum_mode[env_ids]
            self._update_terrain_curriculum(env_ids[curriculum_mask])
            
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        self.apply_randomizations(env_ids)

        # reset buffers
        self.last_last_actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.trap_static_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.tgt_commands[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = torch.max(self.commands[:, 0])
            self.extras["episode"]["max_command_yaw"] = torch.max(self.commands[:, 2])
        if self.cfg.terrain.curriculum:
            self.extras["terrain_levels"] = self.terrain_levels
            if self.cfg.terrain.print_all_levels:
                for i, terrain_ids in enumerate(self.terrain_type_ids):
                    if len(terrain_ids)>0:
                        self.extras["episode"]['terrain_'+ self.terrain.terrain_names[i]] = torch.mean(self.max_reached_level[terrain_ids].float())
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        if self.cfg.rewards.penalize_curriculum:
            self.extras["episode"]["curriculum_scales"] = self.curriculum_scale

        self.gait_indices[env_ids] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        if not self.cfg.rewards.penalize_curriculum:
            self.curriculum_scale = 1
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            if name in self.curriculum_reward_list:
                rew *= self.curriculum_scale 
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if name in self.command_sums.keys():
                if name in ['tracking_contacts_shaped_force', 'tracking_contacts_shaped_vel']:
                    self.command_sums[name] += self.reward_scales[name] + rew
                else:
                    self.command_sums[name] += rew

        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        real_distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        des_distance = torch.norm(self.commands[env_ids, :2], dim=1) * self.max_episode_length_s
        
        # update level
        is_success_level = True
        for key, value in self.curriculum_thresholds['terrains_level'].items():
            task_reward = self.episode_sums[key][env_ids] / self.max_episode_length
            success_threshold = value * self.reward_scales[key] * torch.ones_like(task_reward)
            if key in self.curriculum_reward_list:
                success_threshold *= self.curriculum_scale
            is_success_level = is_success_level * (task_reward > success_threshold)
        
        self.terrain_levels[env_ids] -= 1 * (real_distance < des_distance * 0.5)
        self.terrain_levels[env_ids] += 1 * (real_distance > self.terrain.env_length / 2) * ~self.large_ori_buf[env_ids] * is_success_level
        
        self.max_reached_level[env_ids] = torch.where(self.terrain_levels[env_ids] > self.max_reached_level[env_ids],
                                                      self.terrain_levels[env_ids],
                                                      self.max_reached_level[env_ids])
        
        leave_max_level_envs = env_ids[self.terrain_levels[env_ids]>= self.max_terrain_level]
        
        self.terrain_levels[leave_max_level_envs] = torch.randint_like(leave_max_level_envs, 0, self.max_terrain_level)
        self.terrain_levels.clip_(min=0)
        
        if self.cfg.commands.curriculum:
            high_track_leave_envs = leave_max_level_envs[self.high_track_mode[leave_max_level_envs]]
            self.terrain_curriculum_mode[high_track_leave_envs] = False

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],self.terrain_types[env_ids]]

    def _preprocess_obs(self):
        
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            self.commands * self.commands_scale,
            self.clock_inputs, 
        ), dim=-1)

        # privileged states
        if self.cfg.env.has_privileged_info:
            friction_range = self.cfg.domain_rand.friction_range

            foot_clearnace = torch.clip(self.foot_pos_world[:, :, 2] - self.measured_foot_scan.mean(-1), -1, 1)
            base_h_error = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.heights_below_base
                                     -self.cfg.rewards.base_height_target, -0.5, 0.5)
            if self.cfg.env.observe_body_height:
                jump_h_error = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.heights_below_base
                                     -self.cfg.rewards.base_height_target - self.commands[:, 7].unsqueeze(1), -0.5, 0.5)
                self.obs_buf = torch.cat((self.obs_buf,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        jump_h_error,
                                        foot_clearnace,
                                        unscale_np(self.friction_coeffs, friction_range[0], friction_range[1]) * 0.5,
                                        self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1) * self.obs_scales.contact_force,
                                        1 * self.collision_states,
                                        ), dim=-1)
            else:
                self.obs_buf = torch.cat((self.obs_buf,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        base_h_error,
                                        foot_clearnace,
                                        unscale_np(self.friction_coeffs, friction_range[0], friction_range[1]) * 0.5,
                                        self.contact_forces[:, self.feet_indices, :].reshape(self.num_envs, -1) * self.obs_scales.contact_force,
                                        1 * self.collision_states,
                                        ), dim=-1)

        # add other privilege()
        self.add_other_privilege()

        # terrain info
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - self.measured_heights
                                 - self.cfg.rewards.base_height_target, -1, 1.) * self.obs_scales.height_measurements
            # heights_shift = torch.clip(self.heights_shift, -1, 1.)
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
            
    def add_other_privilege(self):
        pass

    def compute_observations(self, reset_env_ids):
        """ Computes observations
        """
        self._preprocess_obs()

        # sensor obs delayed
        if self.cfg.domain_rand.randomize_control_latency:
            self.delayed_obs_buf.reset(reset_env_ids, self.obs_buf[reset_env_ids, :self.sensor_delayed_dim])
            self.delayed_obs_buf.insert(self.obs_buf[:, :self.sensor_delayed_dim])
            # delay obs
            self.obs_buf[:, :self.sensor_delayed_dim] = self.GetDelayedBodyObservation(self.control_latency)

        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
        
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)

        partial_obs = self.obs_buf[..., :self.num_partial_obs]

        if self.cfg.env.stack_history_obs:
            self.obs_buf_history.reset(reset_env_ids, partial_obs[reset_env_ids])
            self.obs_buf_history.insert(partial_obs)

        if self.cfg.env.stack_history_obs:
            self.partial_obs_buf, self.atten_mask = self.obs_buf_history.get_obs_tensor_3D()
        else:
            self.partial_obs_buf = partial_obs
        
    def GetDelayedBodyObservation(self, latency):
        """Get observation that is delayed by the amount specified in latency.

        Args:
        latency: The latency (in seconds) of the delayed observation.

        Returns:
        observation: The observation which was actually latency seconds ago.
        """
        if latency <= 0 or self.delayed_obs_buf.include_history_steps == 1:
            sensor_obs = self.delayed_obs_buf.obs_buf[:, -1, :self.sensor_delayed_dim]
        else:
            n_steps_ago = int(latency / self.dt)
            if n_steps_ago + 1 >= self.delayed_obs_buf.include_history_steps:
                return self.delayed_obs_buf.obs_buf[:, 0, :self.sensor_delayed_dim]

            remaining_latency = latency - n_steps_ago * self.dt
            blend_alpha = remaining_latency / self.dt
            sensor_obs = (
                (1.0 - blend_alpha) *
                self.delayed_obs_buf.obs_buf[:, -(1+n_steps_ago), :self.sensor_delayed_dim] +
                blend_alpha * self.delayed_obs_buf.obs_buf[:, -(1+n_steps_ago+1), :self.sensor_delayed_dim])
        return sensor_obs

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type == 'plane':
            self._create_ground_plane()
        elif mesh_type == 'trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError(
                "Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat, env_id=0):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, self.envs[env_id], cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id]
            # props[s].restitution = self.restitutions[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass
                # print(f"Mass of body {i}:{p.mass} (before randomization)")
            print(f"Total mass {sum} (before randomization)")
            if self.cfg.domain_rand.randomize_link_props:
                print(
                    f"randomize_link_inertia with ratio: {self.cfg.domain_rand.inertia_ratio}")
                print(
                    f"randomize_link_mass with ratio: {self.cfg.domain_rand.mass_ratio}")
                print(
                    f"randomize_link_com with ofsset: {self.cfg.domain_rand.link_com_offset}")

        if self.cfg.domain_rand.randomize_link_props:
            for p in props:
                inertia_noise = np.random.uniform(self.cfg.domain_rand.inertia_ratio[0],
                                                  self.cfg.domain_rand.inertia_ratio[1], 3)
                p.inertia.x *= inertia_noise[0]
                p.inertia.y *= inertia_noise[1]
                p.inertia.z *= inertia_noise[2]

                p.mass *= np.random.uniform(self.cfg.domain_rand.mass_ratio[0],
                                            self.cfg.domain_rand.mass_ratio[1])  

                link_com_noise = np.random.uniform(-self.cfg.domain_rand.link_com_offset,
                                                   self.cfg.domain_rand.link_com_offset, 3)
                p.com += gymapi.Vec3(link_com_noise[0],
                                     link_com_noise[1], link_com_noise[2])

        if self.cfg.domain_rand.randomize_base_mass:
            mass_range = self.cfg.domain_rand.added_mass_range
            com_x_range = self.cfg.domain_rand.base_com_x_offset_range
            com_y_range = self.cfg.domain_rand.base_com_y_offset_range
            com_x_offset = np.random.uniform(com_x_range[0], com_x_range[1])
            com_y_offset = np.random.uniform(com_y_range[0], com_y_range[1])
            props[0].mass += np.random.uniform(mass_range[0], mass_range[1])
            props[0].com += gymapi.Vec3(com_x_offset, com_y_offset, 0)

        return props
    
    def apply_randomizations(self, env_ids):
        if self.cfg.domain_rand.randomize_gains:  
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]
            self.randomized_motor_strength[env_ids] = new_randomized_gains[2]

        if self.cfg.domain_rand.randomize_control_latency:
            self.control_latency = np.random.uniform(self.cfg.domain_rand.control_latency_range[0],
                                                     self.cfg.domain_rand.control_latency_range[1])

        if self.cfg.domain_rand.randomize_friction:
            min_friction, max_friction = self.cfg.domain_rand.friction_range
            self.friction_coeffs[env_ids] = torch.rand(len(env_ids), 1, dtype=torch.float, device=self.device,
                                                          requires_grad=False) * (
                                                       max_friction - min_friction) + min_friction

        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.motor_offset_range
            self.motor_offsets[env_ids, :] = torch.rand(len(env_ids), self.num_dof, dtype=torch.float,
                                                        device=self.device, requires_grad=False) * (
                                                        max_offset - min_offset) + min_offset

        self.refresh_actor_rigid_shape_props(env_ids)

    def refresh_actor_rigid_shape_props(self, env_ids):
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        heading_mask = self.terrain_curriculum_mode[env_ids]
        self._resample_commands(env_ids[~heading_mask])

        if self.cfg.terrain.curriculum:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_error = wrap_to_pi(self.heading_cmd[self.terrain_curriculum_mode] - heading[self.terrain_curriculum_mode])
            counter_clock = heading_error > 0.1
            clock = heading_error < -0.1
            direction = torch.zeros_like(heading_error, device=self.device)
            direction[counter_clock] = 1
            direction[clock] = -1
            self.commands[self.terrain_curriculum_mode, 2] = direction * torch.abs(self.commands[self.terrain_curriculum_mode, 2])

        if self.cfg.env.observe_gait_commands:
            self._step_contact_targets()

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            if self.cfg.terrain.measure_foot_scan:
                self.measured_foot_scan = self._get_foot_scan_points_heights()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.push_interval == 0):
            self._push_robots()
        
        if self.custom_origins:
            all_env_ids = torch.arange(self.num_envs, device=self.device)
            self.teleport_robots(all_env_ids)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # update high speed envs
        heading_mask = self.terrain_curriculum_mode[env_ids]
        self.heading_cmd[env_ids[heading_mask]] = torch_rand_float(
            self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids[heading_mask]), 1), device=self.device).squeeze(1)
        
        # to tightly
        high_track_mask = self.high_track_mode[env_ids[~heading_mask]]
        high_track_envs = env_ids[~heading_mask][high_track_mask]
        if len(high_track_envs) > 0 and self.cfg.commands.curriculum:
            self.update_command_curriculum_grid(high_track_envs)
        
        # sample envs as standing
        standing_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_standing = 1. / 10
        standing_env_ids = env_ids[torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)] 
        none_standing_env_ids = env_ids[~torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)]
        self.standing_envs_mask[standing_env_ids] = True 
        self.standing_envs_mask[none_standing_env_ids] = False 
        self.commands[standing_env_ids, :3] = 0 

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)

        self.velocity_level[env_ids] = torch.clip(1.0*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)

        # update gait commands
        self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["gait_frequency"][0], self.command_ranges["gait_frequency"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # Frequency
        phases = torch.tensor([0, 0.5], device=self.device)
        random_indices = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
        self.commands[env_ids, 4] = phases[random_indices] # phases
        self.commands[env_ids, 5] = 0.5  # durations
        self.commands[env_ids, 6] = torch_rand_float(self.command_ranges["foot_swing_height"][0], self.command_ranges["foot_swing_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # swing_heights

        # clip commands for high speed envs
        high_speed_env_mask = self.velocity_level[env_ids] > 1.8
        self.commands[env_ids[high_speed_env_mask], 3] = self.commands[env_ids[high_speed_env_mask], 3].clip(min=2.0)  # Frequency

        # clip swing height for high frequency
        high_frequency_env_mask = self.commands[env_ids, 3] > 2.5
        self.commands[env_ids[high_frequency_env_mask], 6] = self.commands[env_ids[high_frequency_env_mask], 6].clip(max=0.20)

        hopping_mask = self.commands[env_ids, 4] == 0
        walking_mask = self.commands[env_ids, 4] == 0.5
        hopping_env_ids = env_ids[hopping_mask]
        walking_env_ids = env_ids[walking_mask]
        if self.cfg.env.observe_body_height:
            self.commands[env_ids, 7] = torch_rand_float(self.command_ranges["body_height"][0], self.command_ranges['body_height'][1], (len(env_ids), 1), device=self.device).squeeze(1)

            # clip swing height for low body height
            low_height_env_mask = self.commands[env_ids, 7] < -0.15
            self.commands[env_ids[low_height_env_mask], 6] = self.commands[env_ids[low_height_env_mask], 6].clip(max=0.20)
        
        if self.cfg.env.observe_body_pitch:
            self.commands[env_ids, 8] = torch_rand_float(self.command_ranges["body_pitch"][0], self.command_ranges['body_pitch'][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            # clip body_pitch for low body height
            low_height_env_mask = self.commands[env_ids, 7] < -0.2
            self.commands[env_ids[low_height_env_mask], 8] = self.commands[env_ids[low_height_env_mask], 8].clip(max=0.3)
            self.commands[env_ids[high_speed_env_mask], 8] = self.commands[env_ids[high_speed_env_mask], 8].clip(max=0.3)      
            
        if self.cfg.env.observe_waist_roll:
            self.commands[env_ids, 9] = torch_rand_float(self.command_ranges["waist_roll"][0], self.command_ranges['waist_roll'][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.commands[env_ids[high_speed_env_mask], 9] = self.commands[env_ids[high_speed_env_mask], 9].clip(min=-0.15, max=0.15) 

        # clip commands for hopping envs
        self.commands[hopping_env_ids, 6] = self.commands[hopping_env_ids, 6].clip(max=0.2)
        if self.cfg.env.observe_body_pitch:
            self.commands[hopping_env_ids, 8] = self.commands[hopping_env_ids, 8].clip(max=0.3)
        
        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
    
    # update for multipe gait
    def _init_command_distribution(self, env_ids):
        self.category_names = ['walking', 'hopping']
        self.curricula = []
        for category in self.category_names:
            self.curricula += [RewardThresholdCurriculum(seed=self.cfg.seed,
                                                        x_vel=(self.cfg.commands.ranges.limit_vel_x[0],
                                                               self.cfg.commands.ranges.limit_vel_x[1],
                                                               self.cfg.commands.num_bins_vel_x),
                                                        yaw_vel=(self.cfg.commands.ranges.limit_vel_yaw[0],
                                                               self.cfg.commands.ranges.limit_vel_yaw[1],
                                                               self.cfg.commands.num_bins_vel_yaw)
                                                        )]

        self.env_command_bins = np.zeros(len(env_ids), dtype=np.int16)
        self.env_command_categories = np.zeros(len(env_ids), dtype=np.int16)
        low = np.array([self.cfg.commands.ranges.lin_vel_x[0], self.cfg.commands.ranges.ang_vel_yaw[0], ])
        high = np.array([self.cfg.commands.ranges.lin_vel_x[1], self.cfg.commands.ranges.ang_vel_yaw[1], ])
        for curriculum in self.curricula:
            curriculum.set_to(low=low, high=high)
    
    def update_command_curriculum_grid(self, env_ids):
        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # update curricula based on terminated environment bins and categories
        for i, (category, curriculum) in enumerate(zip(self.category_names, self.curricula)):
            env_ids_in_category = self.env_command_categories[env_ids.cpu()] == i
            if isinstance(env_ids_in_category, np.bool_) or len(env_ids_in_category) == 1:
                env_ids_in_category = torch.tensor([env_ids_in_category], dtype=torch.bool)
            elif len(env_ids_in_category) == 0:
                continue

            env_ids_in_category = env_ids[env_ids_in_category]

            task_rewards, success_thresholds = [], []
            for key ,value in self.curriculum_thresholds['commands'].items():
                task_rewards.append(self.command_sums[key][env_ids_in_category] / ep_len)
                success_threshold = value * self.reward_scales[key]
                if key in self.curriculum_reward_list:
                    success_threshold *= self.curriculum_scale
                success_thresholds.append(success_threshold)

            old_bins = self.env_command_bins[env_ids_in_category.cpu().numpy()]
            if len(success_thresholds) > 0:
                curriculum.update(old_bins, task_rewards, success_thresholds,
                                  local_range=np.array(
                                      [0.55, 0.55]))

        # assign resampled environments to new categories
        hopping_mask = self.commands[env_ids, 4] == 0
        walking_mask = self.commands[env_ids, 4] == 0.5
        category_env_ids = [env_ids[walking_mask], env_ids[hopping_mask]]

        # sample from new category curricula
        for i, (category, env_ids_in_category, curriculum) in enumerate(
                zip(self.category_names, category_env_ids, self.curricula)):

            batch_size = len(env_ids_in_category)
            if batch_size == 0: continue

            new_commands, new_bin_inds = curriculum.sample(batch_size=batch_size)

            self.env_command_bins[env_ids_in_category.cpu().numpy()] = new_bin_inds
            self.env_command_categories[env_ids_in_category.cpu().numpy()] = i

            self.commands[env_ids_in_category, 0] = torch.Tensor(new_commands[:, 0]).to(self.device)
            self.commands[env_ids_in_category, 1] = torch_rand_float(-0.5, 0.5, (len(env_ids_in_category), 1), device=self.device).squeeze(1)
            self.commands[env_ids_in_category, 2] = torch.Tensor(new_commands[:, 1]).to(self.device)
        
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0
    
    def _step_contact_targets(self):
        if self.cfg.env.observe_gait_commands:
            frequencies = self.commands[:, 3]
            phases = self.commands[:, 4]
            durations = self.commands[:, 5]
            self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

            foot_indices = [self.gait_indices + phases,
                            self.gait_indices]
            
            # set standing envs foot_indices to zero
            for idxs in foot_indices:
                idxs[self.standing_envs_mask] = 0.0
            
            self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(2)], dim=1), 1.0)

            for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

            self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
            self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        
            # von mises distribution
            kappa = self.cfg.rewards.kappa_gait_probs
            smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                                    kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2

            smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
            smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
                    1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                                       smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                               1 - smoothing_cdf_start(
                                           torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
            
            self.desired_contact_states[:, 0] = smoothing_multiplier_FL
            self.desired_contact_states[:, 1] = smoothing_multiplier_FR
        
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
            motor_strength = self.randomized_motor_strength
        else:
            p_gains = self.p_gains.repeat(self.num_envs, 1)
            d_gains = self.d_gains.repeat(self.num_envs, 1)
            motor_strength = self.motor_strength
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) - d_gains*self.dof_vel
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques *= motor_strength
        torque_limits = self.custom_torque_limits
        return torch.clip(torques, -torque_limits, torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = torch_rand_float(-1, 1, (len(env_ids), self.num_dof), device=self.device)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """

        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        
        # base yaws
        init_yaws = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)
        quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        self.root_states[env_ids, 3:7] = quat

        # init_yaws = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)
        # quat = quat_from_angle_axis(init_yaws, torch.Tensor([0, 0, 1]).to(self.device))[:, 0, :]
        # self.root_states[env_ids, 3:7] = quat

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy * self.curriculum_scale
        max_ang_v = self.cfg.domain_rand.max_push_ang_vel_xy * self.curriculum_scale
        push_vel_xyz = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3), device=self.device)
        push_ang_xyz = torch_rand_float(-max_ang_v, max_ang_v, (self.num_envs, 3), device=self.device)
        self.root_states[:, 7:10] += push_vel_xyz
        self.root_states[:, 10:13] += push_ang_xyz
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:6+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[6+self.num_actions:6+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[6+2*self.num_actions:6+3*self.num_actions] = 0. # previous actions
        noise_vec[6+3*self.num_actions:6+3*self.num_actions+3] = 0  # commands

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_states).view(self.num_envs, -1, 13)
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        self.push_force_counter = 0

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        
        self.custom_torque_limits = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength = torch.ones(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.randomized_motor_strength = torch.ones(
            self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros_like(self.actions)
        
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.heading_cmd = torch.zeros(self.num_envs,  dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        
        if self.cfg.env.observe_waist_roll:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,
                                                self.obs_scales.body_height_cmd, self.obs_scales.body_pitch_cmd,
                                                self.obs_scales.waist_roll_cmd], device=self.device, requires_grad=False,)
        elif self.cfg.env.observe_body_pitch:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,
                                                self.obs_scales.body_height_cmd, self.obs_scales.body_pitch_cmd], device=self.device, requires_grad=False,)
        elif self.cfg.env.observe_body_height:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,
                                                self.obs_scales.body_height_cmd], device=self.device, requires_grad=False,)

        elif self.cfg.env.observe_gait_commands:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel,
                                                self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                                                self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,
                                                ], device=self.device, requires_grad=False,)
        else:
            self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.tgt_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.trap_static_time = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
            if self.cfg.terrain.measure_foot_scan:
                self.foot_scan_grids = self._init_foot_scan_height_points()
        self.measured_heights = 0

        self.velocity_level = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.heights_below_base = torch.zeros(
            self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.bool_foot_contact = torch.zeros(
            self.num_envs, self.feet_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.collision_states = torch.zeros(
            self.num_envs, self.penalised_contact_indices.shape[0], dtype=torch.bool, device=self.device, requires_grad=False)
        self.rigid_body_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.push_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.force_pos = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.curriculum_scale = deepcopy(self.cfg.rewards.curriculum_init)
        self.control_latency = 0
        self.terrain_curriculum_mode = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.cfg.terrain.curriculum:
            self.terrain_curriculum_mode[:] = True

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    self.custom_torque_limits[i] = self.cfg.control.torque_limits[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                self.custom_torque_limits[i] = 100.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)


        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains, \
                self.randomized_motor_strength = self.compute_randomized_gains(
                    self.num_envs)
        
        self.curriculum_reward_list = self.cfg.rewards.reward_curriculum_list
        print(self.curriculum_reward_list)
        self.command_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.curriculum_thresholds['commands'].keys()}

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.desired_contact_states = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device,
                                                  requires_grad=False, )
    
    def compute_randomized_gains(self, num_envs):
        p_mult = ((
            self.cfg.domain_rand.stiffness_multiplier_range[0] -
            self.cfg.domain_rand.stiffness_multiplier_range[1]) *
            torch.rand(num_envs, self.num_dof, device=self.device) +
            self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
            self.cfg.domain_rand.damping_multiplier_range[0] -
            self.cfg.domain_rand.damping_multiplier_range[1]) *
            torch.rand(num_envs, self.num_dof, device=self.device) +
            self.cfg.domain_rand.damping_multiplier_range[1]).float()
        strength_mult = ((
            self.cfg.domain_rand.motor_strength_range[0] -
            self.cfg.domain_rand.motor_strength_range[1]) *
            torch.rand(num_envs, self.num_dof, device=self.device) +
            self.cfg.domain_rand.motor_strength_range[1]).float()

        return p_mult * self.p_gains, d_mult * self.d_gains, strength_mult * self.motor_strength

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y,
                         device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x,
                         device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points,
                             3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()

        self.height_radius = torch.ones(self.num_envs, self.num_height_points, 1, dtype=torch.float, device=self.device, requires_grad=False) * self.cfg.terrain.height_radius
        limit_x = 5  # TODO
        limit_y = 3  # TODO
        x_mask = (grid_x<=limit_x)*(grid_x>=-limit_x)
        y_mask = (grid_y<=limit_y)*(grid_y>=-limit_y)
        self.center_mask = (x_mask*y_mask).flatten()
        self.center_origin_mask = ((grid_x==0)*(grid_y==0)).flatten()

        return points
    
    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError(
                "Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(
                1, self.num_height_points), self.height_points[env_ids] * self.height_radius[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(
                1, self.num_height_points), self.height_points * self.height_radius) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        heights_scan = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        self.heights_below_base = heights_scan[:,self.center_origin_mask]

        self.heights_shift = heights_scan - self.heights_below_base 
        self.flat_env_mask = self.heights_shift.std(dim=-1)<self.cfg.terrain.flat_std_limit

        return heights_scan
    
    def _get_foot_scan_points_heights(self):       
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.feet_indices.shape[0], self.num_scan_points, device=self.device, requires_grad=False)
 
        points = quat_apply_yaw(self.base_quat.repeat(1, self.num_scan_points * self.feet_indices.shape[0]), self.foot_scan_grids) + self.foot_pos_world.unsqueeze(2)
        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        
        px = points[:, :, :, 0].view(-1)
        py = points[:, :, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        
        return heights.view(self.num_envs, self.feet_indices.shape[0], -1) * self.terrain.cfg.vertical_scale
    
    def _init_foot_scan_height_points(self):
        """ Returns points at which the height measurments are sampled (in foot surrounding)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, num_legs, self.num_scan_points, 3)
        """
        radius=self.cfg.terrain.scan_radius
        y = torch.tensor([-1, 0, 1], device=self.device, requires_grad=False)*radius
        x = torch.tensor([-2, -1, 0, 1, 2, 3, 4], device=self.device, requires_grad=False)*radius
        grid_x, grid_y = torch.meshgrid(x, y) 

        self.num_scan_points = grid_x.numel() # 网格点总数
        scan_points = torch.zeros(self.num_envs, self.feet_indices.shape[0], self.num_scan_points, 3, device=self.device, requires_grad=False)
        scan_points[:, :, :, 0] = grid_x.flatten()
        scan_points[:, :, :, 1] = grid_y.flatten() # 生成平面网格点

        return scan_points

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(
            order='C'), self.terrain.triangles.flatten(order='C'), tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(
            self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.override_com = self.cfg.asset.override_com
        asset_options.override_inertia = self.cfg.asset.override_inertia

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)
        self.default_friction = rigid_shape_props_asset[0].friction
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.body_names = body_names
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        dof_names_lower = []
        for dof in self.dof_names:
            dof_names_lower.append(dof.lower())
        
        torso_inds = [i for i, n in enumerate(dof_names_lower) if ('torso' in n)]
        shoulder_inds = [i for i, n in enumerate(dof_names_lower) if ('shoulder_roll' in n or 'shoulder_yaw' in n)]
        elbow_inds = [i for i, n in enumerate(dof_names_lower) if ('elbow' in n or 'shoulder_pitch' in n)]
        hip_inds = [i for i, n in enumerate(dof_names_lower) if ('hip_roll' in n or 'hip_yaw' in n)]
        print('torso_inds', torso_inds)
        print('shoulder_inds', shoulder_inds)
        print('hip_inds', hip_inds)
        print('elbow_inds', elbow_inds)
        self.torso_inds = torso_inds
        self.shoulder_inds = shoulder_inds
        self.elbow_inds = elbow_inds
        self.hip_inds = hip_inds

        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
        base_name = [s for s in body_names if self.cfg.asset.base_name in s]

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self.high_track_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standing_envs_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])
        
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        base_indice = self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.actor_handles[0], base_name[0])
        self.base_indice = to_torch(
            base_indice, dtype=torch.long, device=self.device)
        
    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
      
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if self.cfg.terrain.curriculum:
                # record env_ids for different terrain types
                self.terrain_type_ids = []
                for i in range(len(self.terrain.proportions)):
                    if i == 0:
                        start_ids = 0
                    else:
                        start_ids = int(self.terrain.proportions[i-1]*self.num_envs)
                    end_ids = int(self.terrain.proportions[i]*self.num_envs)
                    terrain_ids = torch.arange(start_ids, end_ids, device=self.device)
                    self.terrain_type_ids.append(terrain_ids) 
                    if self.terrain.terrain_names[i] in ["random_uniform"]:
                        self.high_track_mode[terrain_ids] = True
            else:
                max_init_level = self.cfg.terrain.num_rows - 1

            self.terrain_levels = torch.randint(
                0, max_init_level+1, (self.num_envs,), device=self.device)
            self.max_reached_level = self.terrain_levels.clone()

            # fixed_gait random slope stairs
            self.first_terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (
                self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (
                self.num_envs/(self.cfg.terrain.num_cols * self.terrain.proportions[0])), rounding_mode='floor').to(torch.long)
            
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(
                self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels,self.terrain_types]

        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(
                self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(
                num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
    
    def teleport_robots(self, env_ids):
        """ Teleports any robots that are too close to the edge to the other side
        """
        thresh = 2

        low_x_ids = env_ids[self.root_states[env_ids, 0] < (-self.cfg.terrain.border_size + thresh)]
        self.root_states[low_x_ids, 0] += self.cfg.terrain.border_size/2

        high_x_ids = env_ids[self.root_states[env_ids, 0] > 
        (self.cfg.terrain.terrain_length * self.cfg.terrain.num_rows + self.cfg.terrain.border_size - thresh)]
        self.root_states[high_x_ids, 0] -= self.cfg.terrain.border_size/2

        low_y_ids = env_ids[self.root_states[env_ids, 1] < (-self.cfg.terrain.border_size + thresh)]
        self.root_states[low_y_ids, 1] += self.cfg.terrain.border_size/2

        high_y_ids = env_ids[self.root_states[env_ids, 1] > 
        (self.cfg.terrain.terrain_width * self.cfg.terrain.num_cols + self.cfg.terrain.border_size - thresh)]
        self.root_states[high_y_ids, 1] -= self.cfg.terrain.border_size/2

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        print(self.reward_scales)
        self.curriculum_thresholds = class_to_dict(self.cfg.curriculum_thresholds)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False

        self.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)
    
    def set_color(self, env_ids, color: str = 'red'):
        color_rgb = {
            'red': gymapi.Vec3(1, 0, 0),
            'green': gymapi.Vec3(0, 1, 0),
            'blue': gymapi.Vec3(0, 0, 1),
            'yellow': gymapi.Vec3(1, 0.8431, 0),
            'gray': gymapi.Vec3(0.1, 0.1, 0.1),
            'purple': gymapi.Vec3(0.63, 0.12, 1),
            'orange': gymapi.Vec3(1, 0.5, 0),
        }
        # differ color to distinguish
        for env_id in env_ids:
            env = self.envs[env_id]
            actor = self.actor_handles[env_id]
            self.gym.set_rigid_body_color(
                env, actor, 0, gymapi.MESH_VISUAL, color_rgb[color])
    
    def training_curriculum(self, num_steps=1, abs_steps=None):
        super().training_curriculum(num_steps=1, abs_steps=abs_steps)
        if self.cfg.rewards.penalize_curriculum and (self.learning_iter % 100 == 0):
            self.curriculum_scale = pow(self.curriculum_scale, self.cfg.rewards.penalize_curriculum_sigma)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        penalzie = torch.square(self.base_lin_vel[:, 2])
        return penalzie
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_standing(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        double_contact = torch.sum(1.*contacts, dim=1) == 2
        double_contact[~self.standing_envs_mask] = 0
        return 1*double_contact
    
    def _reward_standing_air(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        double_contact = torch.sum(1.*contacts, dim=1) == 0
        double_contact[~self.standing_envs_mask] = 0
        return 1*double_contact
    
    def _reward_standing_joint_deviation(self):
        penalize_joint_index = torch.tensor([11, 12, 13, 14, 15, 16, 17, 18], dtype=torch.int, device=self.device)
        reward = torch.square(self.dof_pos - self.default_dof_pos)[:, penalize_joint_index]
        reward[~self.standing_envs_mask] = 0
        return torch.sum(reward, dim=-1)
    
    def _reward_orientation_control(self):
        # Penalize non flat base orientation
        pitch_commands = self.commands[:, 8]
        quat_pitch = quat_from_angle_axis(pitch_commands[:],
                                          torch.tensor([0, 1, 0], device=self.device, dtype=torch.float))
        desired_projected_gravity = quat_rotate_inverse(quat_pitch, self.gravity_vec)

        return torch.sum(torch.square(self.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)
    
    def _reward_waist_control(self):
        waist_commands = self.commands[:, 9]
        reward = torch.square(self.dof_pos[:, self.torso_inds].squeeze(-1) - waist_commands)
        return reward

    def _reward_base_height(self):
        body_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.heights_below_base, dim=-1)
        body_height_target = self.commands[:, 7] + self.cfg.rewards.base_height_target
        reward = torch.square(body_height - body_height_target)
        reward[self.standing_envs_mask] *= 3
        return reward
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):     
        # Penalize changes in actions
        diff_1 = torch.sum(torch.square(self.actions[:,:self.num_dof] - self.last_actions[:,:self.num_dof]), dim=1)
        diff_2 = torch.sum(torch.square(self.actions[:,:self.num_dof] - 
                           2 * self.last_actions[:,:self.num_dof] + self.last_last_actions[:,:self.num_dof]), dim=1)
        return diff_1 + diff_2

    def _reward_dof_vel_limits(self):   
        # Penalize dof velocities too close to the limit
        dof_vel_limits = torch.clip(10 * self.velocity_level.unsqueeze(-1).repeat(1,self.num_dof), min=10, max=20)
        error = torch.sum((torch.abs(self.dof_vel) - dof_vel_limits).clip(min=0., max=15.), dim=1)
        rew = 1 - torch.exp(-1 * error)
        return rew
    
    def _reward_feet_contact_forces(self):       
        # penalize high contact forces
        reward = torch.sum(
            torch.square(self.obs_scales.contact_force*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.)), 
            dim=1).clip(max=2.0)
        reward[self.standing_envs_mask] *= 0.2
        return reward
    
    def _reward_collision(self):    
        # Penalize collisions on selected bodies
        penalize = torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
        return penalize 
    
    def _reward_termination(self):
        # Terminal reward / penalty
        penalize = self.reset_buf * ~self.time_out_buf
        return penalize
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_stumble(self):      
        # Penalize feet hitting vertical surfaces
        stumble = (torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 2.)*\
                  (torch.abs(self.contact_forces[:, self.feet_indices, 2]) < 1.)
        return torch.sum(stumble, dim=1)
        
    def _reward_stand_still(self):      
        self.base_stance_w = 0.38 

        zero_cmd_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1) * (torch.abs(self.commands[:, 2]) < 0.1)
        desired_stance_width = self.base_stance_w * torch.ones(self.num_envs, 1, device=self.device, requires_grad=False)
        desired_ys_nom = torch.cat([desired_stance_width / 2, -desired_stance_width / 2], dim=1)
        err_raibert_heuristic = torch.sum(torch.square(desired_ys_nom[:, :] - self.foot_pos_b_h[:, :, 1]), dim=1)
        err_raibert_heuristic[zero_cmd_mask] *= 10

        return err_raibert_heuristic
    
    def _reward_joint_power_distribution(self):     
        # Penalize torques
        shank_joint_index = torch.tensor([3, 8], dtype=torch.long, device=self.device)
        Penalize = (torch.abs(self.torques) * torch.abs(self.dof_vel))[:, shank_joint_index].var(dim=-1)
        return (torch.square(Penalize) * 1.e-8).clip(max=1)

    def _reward_hip_deviation(self):
        reward = torch.square(self.dof_pos - self.default_dof_pos)[:, self.hip_inds]
        return torch.sum(reward, dim=-1)
    
    def _reward_shoulder_deviation(self):
        shoulder_reward = torch.square(self.dof_pos - self.default_dof_pos)[:, self.shoulder_inds]
        elbow_reward = torch.square(self.dof_pos - self.default_dof_pos)[:, self.elbow_inds]
        return 2 * torch.sum(shoulder_reward, dim=-1) + 0.25 * torch.sum(elbow_reward, dim=-1)
    
    def _reward_no_fly(self):
        zero_cmd_mask = (torch.norm(self.commands[:, :2], dim=1) < 0.1) * (torch.abs(self.commands[:, 2]) < 0.1)
        walking_mask = self.commands[:, 4] == 0.5
        hopping_mask = self.commands[:, 4] == 0
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        none_contact = torch.sum(1.*contacts, dim=1) == 0
        double_contact = torch.sum(1.*contacts, dim=1) == 2
        same_contact = none_contact | double_contact
        single_contact = torch.sum(1.*contacts, dim=1) == 1
        same_contact[walking_mask] = 0
        single_contact[hopping_mask] = 0
        same_contact[zero_cmd_mask] = 0
        single_contact[zero_cmd_mask] = 0

        return 1.*same_contact + 1.*single_contact
    
    def _reward_tracking_contacts_shaped_force(self):
        foot_forces = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        desired_contact = self.desired_contact_states

        reward = 0
        for i in range(2):
            reward += - (1 - desired_contact[:, i]) * (
                        1 - torch.exp(-1 * foot_forces[:, i] ** 2 / self.cfg.rewards.gait_force_sigma))

        return reward / 2

    def _reward_tracking_contacts_shaped_vel(self):
        foot_velocities = torch.norm(self.foot_velocity_world, dim=2).view(self.num_envs, -1)
        desired_contact = self.desired_contact_states
        reward = 0
        for i in range(2):
            reward += - (desired_contact[:, i] * (
                        1 - torch.exp(-1 * foot_velocities[:, i] ** 2 / self.cfg.rewards.gait_vel_sigma)))

        return  reward / 2
    
    def _reward_feet_clearance_cmd_linear(self):   
        phases = 1 - torch.abs(1.0 - torch.clip((self.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
        foot_height = (self.foot_pos_world[:, :, 2]).view(self.num_envs, -1) - self.measured_foot_scan.mean(-1)
        target_height = self.commands[:, 6].unsqueeze(1) * phases + 0.07 
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)
        
        return torch.sum(rew_foot_clearance, dim=1).clip(max=0.1)

    def _reward_feet_clearance_cmd_polynomial(self):   
        phases = torch.clip(0.75 - torch.abs(self.foot_indices - 0.75), 0.0, 1.0)
        coef = _polynomial_planer(0.5, 0.75, 0, 1)
        polynomial_curve = coef[0] + coef[1] * (phases - 0.5) + coef[2] * ((phases - 0.5) ** 2) + \
                            coef[3] * ((phases - 0.5) ** 3) + coef[4] * ((phases - 0.5) ** 4)+ coef[5] * ((phases - 0.5) ** 5)
        even_curve_mask = phases < 0.5
        polynomial_curve[even_curve_mask] = 0
        target_height = self.commands[:, 6].unsqueeze(1) * polynomial_curve + 0.07 
        foot_height = (self.foot_pos_world[:, :, 2]).view(self.num_envs, -1) - self.measured_foot_scan.mean(-1)
        rew_foot_clearance = torch.square(target_height - foot_height) * (1 - self.desired_contact_states)

        return torch.sum(rew_foot_clearance, dim=1).clip(max=0.1)
    
    def _reward_feet_slip(self):     
        # penalize feet_slip
        penalize = torch.zeros(self.num_envs, self.feet_indices.shape[0], device=self.device, requires_grad=False)
        contact_vel = self.foot_velocity_world[self.bool_foot_contact]
        penalize[self.bool_foot_contact] = torch.norm(contact_vel[..., :2], dim=-1)
        rew = 1 - torch.exp(-1 * torch.sum(penalize, dim=-1))
        return rew
    
    def _reward_hopping_symmetry(self):
        walking_mask = self.commands[:, 4] == 0.5
        penalize = torch.abs(self.foot_pos_b_h[:, 0, 0] - self.foot_pos_b_h[:, 1, 0]) + torch.abs(self.foot_pos_b_h[:, 0, 2] - self.foot_pos_b_h[:, 1, 2])
        penalize[walking_mask] = 0
        return penalize

    def _reward_alive(self):
        return 1
    