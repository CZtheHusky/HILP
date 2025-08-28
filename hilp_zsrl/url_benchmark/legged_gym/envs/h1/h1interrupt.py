from legged_gym import LEGGED_GYM_ROOT_DIR, envs
import time
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
from legged_gym.envs.h1.h1 import H1Robot
from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg
from copy import deepcopy

class H1InterruptRobot(H1Robot):
    def __init__(self, cfg: H1InterruptCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.cfg = cfg
        self.initial_disturb(cfg)

        # Reconstruct Command Scale
        command_scale = [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel] 
        command_dims = 3
        if cfg.env.observe_gait_commands:
            command_scale += [self.obs_scales.gait_freq_cmd, self.obs_scales.gait_phase_cmd,
                              self.obs_scales.gait_phase_cmd, self.obs_scales.footswing_height_cmd,] 
            self.command_gait_freq_dim = 3
            self.command_gait_phase_dim = 4
            self.command_gait_duration_dim = 5
            self.command_swing_heights_dim = 6
            command_dims = 7

        if cfg.env.observe_body_height:
            command_scale.append(self.obs_scales.body_height_cmd) 
            self.command_body_height_dim = command_dims 
            command_dims += 1
        if cfg.env.observe_body_pitch:
            command_scale.append(self.obs_scales.body_pitch_cmd) 
            self.command_body_pitch_dim = command_dims
            command_dims += 1
        if cfg.env.observe_waist_roll:
            command_scale.append(self.obs_scales.waist_roll_cmd) 
            self.command_waist_roll_dim = command_dims
            command_dims += 1
        if self.interrupt_in_command:
            command_scale.append(1) 
            self.command_interrupt_flag_dim = command_dims
            command_dims += 1

        self.commands_scale = torch.tensor(command_scale, device=self.device, requires_grad=False)
        for name in self.curriculum_thresholds['disturb'].keys():
            self.command_sums[name] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_envs(self):
        super()._create_envs()
        self.disturb_termination_contact_indices = torch.zeros(len(self.cfg.disturb.disturb_terminate_assets), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.disturb.disturb_terminate_assets)):
            self.disturb_termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.disturb.disturb_terminate_assets[i])
        self.noise_disturb_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.noise_env_nums = int(self.num_envs * self.cfg.disturb.noise_curriculum_ratio)
        self.high_track_mode[:self.noise_env_nums] = False 
        self.noise_disturb_mode[:self.noise_env_nums] = True

    def initial_disturb(self, cfg: H1InterruptCfg):
        self.use_disturb = cfg.disturb.use_disturb
        self.disturb_dim = cfg.disturb.disturb_dim
        self.disturb_scale = cfg.disturb.disturb_scale
        self.disturb_switch_prob = cfg.disturb.switch_prob
        self.disturb_actions = torch.zeros(self.num_envs, self.disturb_dim, dtype=torch.float, device=self.device, requires_grad=False)
        self.executed_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturb_masks = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.disturb_noise_ratio = cfg.disturb.noise_ratio
        self.disturb_isnoise = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.interrupt_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.disturb_replace_action = cfg.disturb.replace_action
        self.disturb_rad = cfg.disturb.disturb_rad
        self.disturb_uniform = cfg.disturb.uniform_noise
        self.disturb_noise_update_step = cfg.disturb.noise_update_step
        self.disturb_noise_scale = torch.tensor(cfg.disturb.noise_scale).to(self.device).unsqueeze(0) 
        self.disturb_noise_lowerbound = torch.tensor(cfg.disturb.noise_lowerbound).to(self.device).unsqueeze(0) #+ 0.15
        self.disturb_uniform_scale = cfg.disturb.uniform_scale
        self.disturb_in_last_action = cfg.disturb.disturb_in_last_action
        self.obs_target_interrupt_in_privilege = cfg.disturb.obs_target_interrupt_in_privilege
        self.obs_executed_actions_in_privilege = cfg.disturb.obs_executed_actions_in_privilege
        if cfg.disturb.disturb_rad_curriculum:
            self.disturb_rad_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.disturb_rad_curriculum = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        if hasattr(cfg.disturb, "disturb_terminate_assets"):
            self.disturb_terminate_assets = cfg.disturb.disturb_terminate_assets
        else:
            self.disturb_terminate_assets = []

        self.num_steps = 0
        self.interrupt_in_command = cfg.disturb.interrupt_in_cmd
        self.stand_interrupt_only = cfg.disturb.stand_interrupt_only
        

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """

        if len(env_ids) == 0:
            return
        
        # update vel commands:
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # update high speed envs
        heading_mask = self.terrain_curriculum_mode[env_ids]
        self.heading_cmd[env_ids[heading_mask]] = torch_rand_float(
            self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids[heading_mask]), 1), device=self.device).squeeze(1)
        
        high_track_mask = self.high_track_mode[env_ids[~heading_mask]]
        high_track_envs = env_ids[~heading_mask][high_track_mask]
        if len(high_track_envs) > 0 and self.cfg.commands.curriculum:
            self.update_command_curriculum_grid(high_track_envs)
        
        update_disturb_mask = self.noise_disturb_mode[env_ids[~heading_mask]]
        update_disturb_envs = env_ids[~heading_mask][update_disturb_mask]
        noise_disturb_mask = self.disturb_masks[env_ids[~heading_mask]]
        noise_disturb_envs = env_ids[~heading_mask][noise_disturb_mask]
        if len(update_disturb_mask) > 0 and self.cfg.disturb.disturb_rad_curriculum: 
            self.update_disturb_curriculum_grid(update_disturb_envs, noise_disturb_envs)
        
        # sample envs as standing
        standing_env_floats = torch.rand(len(env_ids), device=self.device)
        probability_standing = 1 / 10
        standing_env_ids = env_ids[torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)] 
        none_standing_env_ids = env_ids[~torch.logical_and(0 <= standing_env_floats, standing_env_floats < probability_standing)]
        self.standing_envs_mask[standing_env_ids] = True 
        self.standing_envs_mask[none_standing_env_ids] = False 
        self.commands[standing_env_ids, :3] = 0 

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > self.cfg.commands.min_vel).unsqueeze(1)
        self.commands[env_ids, 2] *= (torch.abs(self.commands[env_ids, 2]) > self.cfg.commands.min_vel)

        # Velocity_level. 
        self.velocity_level[env_ids] = torch.clip(1.0*torch.norm(self.commands[env_ids, :2], dim=-1)+0.5*torch.abs(self.commands[env_ids, 2]), min=1)
        
        # clip commands for high speed envs
        # high_speed_env_mask = self.velocity_level[env_ids] > 1.8
        # self.commands[env_ids[high_speed_env_mask], 3] = self.commands[env_ids[high_speed_env_mask], 3].clip(min=2.0)  # Frequency

        if self.cfg.env.observe_gait_commands:
            # update gait commands
            self.commands[env_ids, self.command_gait_freq_dim] = torch_rand_float(self.command_ranges["gait_frequency"][0], self.command_ranges["gait_frequency"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # Frequency
            phases = torch.tensor([0, 0.5], device=self.device)
            random_indices = torch.randint(0, len(phases), (len(env_ids), ), device=self.device)
            self.commands[env_ids, self.command_gait_phase_dim] = phases[random_indices] # phases
            self.commands[env_ids, self.command_gait_duration_dim] = 0.5  # durations
            self.commands[env_ids, self.command_swing_heights_dim] = torch_rand_float(self.command_ranges["foot_swing_height"][0], self.command_ranges["foot_swing_height"][1], (len(env_ids), 1), device=self.device).squeeze(1)  # swing_heights

            hopping_mask = self.commands[env_ids, 4] == 0
            walking_mask = self.commands[env_ids, 4] == 0.5
            hopping_env_ids = env_ids[hopping_mask]
            walking_env_ids = env_ids[walking_mask]

        if self.cfg.env.observe_body_height:
            self.commands[env_ids, self.command_body_height_dim] = torch_rand_float(self.command_ranges["body_height"][0], self.command_ranges['body_height'][1], (len(env_ids), 1), device=self.device).squeeze(1)

        if self.cfg.env.observe_body_pitch:
            self.commands[env_ids, self.command_body_pitch_dim] = torch_rand_float(self.command_ranges["body_pitch"][0], self.command_ranges['body_pitch'][1], (len(env_ids), 1), device=self.device).squeeze(1)
            
            # clip body_pitch for hopping
            if self.cfg.env.observe_gait_commands:
                self.commands[hopping_env_ids, self.command_body_pitch_dim] = self.commands[hopping_env_ids, self.command_body_pitch_dim].clip(max=0.3)

        if self.cfg.env.observe_waist_roll:
            self.commands[env_ids, self.command_waist_roll_dim] = torch_rand_float(self.command_ranges["waist_roll"][0], self.command_ranges['waist_roll'][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # reset command sums
        for key in self.command_sums.keys():
            self.command_sums[key][env_ids] = 0.
        
        if self.interrupt_in_command:
            self.commands[env_ids, self.command_interrupt_flag_dim] = False
    
    def update_disturb_curriculum_grid(self, env_ids, noise_env_ids):
        if len(env_ids)==0: return
        timesteps = int(self.cfg.commands.resampling_time / self.dt)
        ep_len = min(self.max_episode_length, timesteps)

        # only for disturb masks
        curr_is_pass = torch.ones(len(noise_env_ids), dtype=bool, device=self.device)
        curr_is_down = torch.zeros(len(noise_env_ids), dtype=bool, device=self.device)

        for key, value in self.curriculum_thresholds['disturb'].items():
            all_rew = self.command_sums[key][noise_env_ids] / ep_len
            success_threshold = value * self.reward_scales[key]
            if key in self.curriculum_reward_list:
                success_threshold *= self.curriculum_scale

            curr_is_pass *= (all_rew > success_threshold)
            curr_is_down += (all_rew < success_threshold / 2)
        
        self.disturb_rad_curriculum[noise_env_ids] = torch.where(
            curr_is_down,
            (self.disturb_rad_curriculum[noise_env_ids] - 0.05).clip(min=0),
            torch.where(
                curr_is_pass,
                (self.disturb_rad_curriculum[noise_env_ids] + 0.05).clip(max=self.cfg.disturb.max_curriculum),
                self.disturb_rad_curriculum[noise_env_ids]
            )
        ) 

        # resample all noise envs disturb
        self.disturb_masks[env_ids] = (torch.rand(len(env_ids))<=0.5).to(self.device) # Reset with half with disturb.
        is_noise = torch.rand(len(env_ids)) <= self.disturb_noise_ratio
        self.disturb_isnoise[env_ids] = is_noise.to(self.device)
        self.disturb_actions[env_ids] = self.dof_pos[env_ids, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:]
        if self.disturb_replace_action:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids]
        else:
            self.interrupt_mask[env_ids] = self.disturb_masks[env_ids] * (~self.disturb_isnoise[env_ids])

    def _preprocess_obs(self):
        if self.interrupt_in_command:
            self.commands[:, self.command_interrupt_flag_dim] = self.interrupt_mask[:]
        super()._preprocess_obs()
        
    def add_other_privilege(self):
        if self.cfg.env.has_privileged_info and self.obs_target_interrupt_in_privilege:
            obs_target = self.disturb_actions * self.interrupt_mask.unsqueeze(-1)
            obs_target = torch.cat((obs_target, self.disturb_rad_curriculum.unsqueeze(-1)), dim=1)
            self.obs_buf = torch.cat((self.obs_buf, obs_target), dim=1)
        if self.cfg.env.has_privileged_info and self.obs_executed_actions_in_privilege:
            self.obs_buf = torch.cat((self.obs_buf, self.executed_actions), dim=1)

    def Gaussian_disturb_resample(self):
        '''
        Sample Gaussian Disturb Actions. 
        '''
        mean = torch.zeros(8, device=self.device)
        std = torch.ones(8, device=self.device) * self.disturb_scale

        return torch.clamp(
            torch.normal(mean, std) + self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos[:, -self.disturb_dim:]
        )
    
    def Uniform_disturb_resample(self):
        '''Sample Noise from Uniform distribution'''
        scale = self.disturb_uniform_scale
        targets = scale * self.disturb_noise_scale * torch.rand((self.num_envs, 8), device=self.device) + self.disturb_noise_lowerbound + self.disturb_noise_scale * (1-scale)/2
        
        # clip disturb for H1
        left_env_mask = targets[:, 1] < 0.5
        targets[left_env_mask][:, [2, 3]] = 0
        right_env_maks =  targets[:, 5] > 0.5
        targets[right_env_maks][:, [6, 7]] = 0

        return torch.clamp(
            targets - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 0].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos[:, -self.disturb_dim:],
            self.dof_pos_limits[-self.disturb_dim:, 1].view(1,-1).repeat(self.num_envs, 1) - self.default_dof_pos[:, -self.disturb_dim:]
        )

    def reset_idx(self, env_ids):         
        super().reset_idx(env_ids)
        if self.use_disturb and self.cfg.disturb.disturb_rad_curriculum:
            self.extras['episode']['disturb_curriculum']= torch.mean(self.disturb_rad_curriculum[:self.noise_env_nums])
            
    def random_switch_disturb(self):
        switch_rand = torch.rand(self.num_envs, device=self.device)
        switch = switch_rand < self.disturb_switch_prob 
        self.disturb_masks = torch.where(switch, ~self.disturb_masks, self.disturb_masks)
        self.disturb_masks[:] *= self.noise_disturb_mode[:] * (~self.terrain_curriculum_mode[:])# Only disturb noise mode has disturb.
        if self.stand_interrupt_only:
            self.disturb_masks[:] *= self.standing_envs_mask[:]
        if self.disturb_replace_action:
            self.interrupt_mask[:] = self.disturb_masks[:]
        else:
            self.interrupt_mask[:] = self.disturb_masks[:] * (~self.disturb_isnoise[:])
            
    def resample_disturb_noise(self):
        self.disturb_actions = torch.where(
            self.disturb_isnoise.view(-1,1).repeat(1,self.disturb_dim),
            self.Uniform_disturb_resample()/self.cfg.control.action_scale if self.disturb_uniform else self.Gaussian_disturb_resample()/self.cfg.control.action_scale,
            self.disturb_actions
        )

    def post_physics_step(self):
        super().post_physics_step()
        self.num_steps += 1
        if self.use_disturb:
            if self.num_steps % self.disturb_noise_update_step == 0:
                self.resample_disturb_noise()
            self.num_steps %= self.disturb_noise_update_step          
    
    def curriculum_disturb_fusion(self, actions):
        disturb_action = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad + self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:]) / self.cfg.control.action_scale,
            (self.disturb_rad + self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:]) / self.cfg.control.action_scale
        ) # Nosie or traj Target

        fused_disturb_action = self.disturb_rad_curriculum.unsqueeze(-1) * disturb_action +  \
                         (1 - self.disturb_rad_curriculum.unsqueeze(-1)) * actions[:, -self.disturb_dim:]
        
        return fused_disturb_action

    def curriculum_disturb_clipping_mean(self, actions):
        # cliping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:])+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (actions[:, -self.disturb_dim:] * self.cfg.control.action_scale)

        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad + noise_mean)/self.cfg.control.action_scale,
            (self.disturb_rad + noise_mean)/self.cfg.control.action_scale
        )
        return disturb_actions
    
    def curriculum_disturb_clipping_mean_rad(self, actions):
        # clipping mean with curriculum
        noise_mean = self.disturb_rad_curriculum.unsqueeze(-1) * (self.dof_pos[:, -self.disturb_dim:] - self.default_dof_pos[:, -self.disturb_dim:])+ \
                (1-self.disturb_rad_curriculum.unsqueeze(-1))  * (actions[:, -self.disturb_dim:] * self.cfg.control.action_scale)
        
        # clipping action rate with curriculum by rad.
        disturb_actions = torch.clamp(
            self.disturb_actions,
            (- self.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.control.action_scale,
            (self.disturb_rad * self.disturb_rad_curriculum.unsqueeze(-1) + noise_mean)/self.cfg.control.action_scale
        )
        return disturb_actions
        
    def calculate_action(self, actions):
        self.actions = actions.clone()
        clip_actions = self.cfg.normalization.clip_actions
        cliped_actions = torch.clip(actions.clone(), -clip_actions, clip_actions).to(self.device)
        if self.use_disturb:
            if self.cfg.disturb.disturb_curriculum_method == 0:
                disturb_action_clip = self.curriculum_disturb_fusion(cliped_actions)
            elif self.cfg.disturb.disturb_curriculum_method == 1:
                disturb_action_clip = self.curriculum_disturb_clipping_mean(cliped_actions)
            elif self.cfg.disturb.disturb_curriculum_method == 2:
                disturb_action_clip = self.curriculum_disturb_clipping_mean_rad(cliped_actions)

            if self.disturb_replace_action:
                cliped_actions[:, -self.disturb_dim:] = torch.where(
                    self.disturb_masks.view(-1, 1).repeat(1, self.disturb_dim),
                    disturb_action_clip,
                    cliped_actions[:, -self.disturb_dim:]
                )
            else:
                # print("ACTION: ", self.actions[0, -self.disturb_dim:], " DISTURB: ", self.disturb_actions[0])
                cliped_actions[:, -self.disturb_dim:] = torch.where(
                    self.disturb_masks.view(-1, 1).repeat(1, self.disturb_dim),
                    cliped_actions[:, -self.disturb_dim:] + disturb_action_clip,
                    cliped_actions[:, -self.disturb_dim:]
                )
            
            cliped_actions = torch.clip(cliped_actions, -clip_actions, clip_actions).to(self.device)
        if self.disturb_in_last_action:
            self.actions[:] = cliped_actions
        self.executed_actions[:] = cliped_actions
        return cliped_actions

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.reset_buf[self.disturb_masks] = False
        self.large_ori_buf = torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.gravity_termination_buf = torch.any(torch.norm(self.projected_gravity[:, 0:2], dim=-1, keepdim=True) > 0.8, dim=1)
        self.reset_buf |= self.large_ori_buf
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
     
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
            noise_disturb_leave_envs = leave_max_level_envs[self.noise_disturb_mode[leave_max_level_envs]]
            self.terrain_curriculum_mode[noise_disturb_leave_envs] = False

        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids],self.terrain_types[env_ids]]

    def _reward_shoulder_deviation(self):
        return super()._reward_shoulder_deviation() * (~self.interrupt_mask)
    
    def _reward_action_rate_upper(self): 
        diff_1 = torch.sum(torch.square(self.actions[:, 11:self.num_dof] - self.last_actions[:, 11:]), dim=1)
        diff_2 = torch.sum(torch.square(self.actions[:, 11:self.num_dof] - 2 * self.last_actions[:, 11:self.num_dof] + self.last_last_actions[:, 11:self.num_dof]), dim=1)
        return (diff_1 + diff_2) * (~self.interrupt_mask)
    
    def _reward_action_rate_lower(self): 
        diff_1 = torch.sum(torch.square(self.actions[:, :11] - self.last_actions[:, :11]), dim=1)
        diff_2 = torch.sum(torch.square(self.actions[:, :11] - 2 * self.last_actions[:, :11] + self.last_last_actions[:, :11]), dim=1)
        return diff_1 + diff_2

    def _reward_standing_joint_deviation(self):
        return super()._reward_standing_joint_deviation() * (~self.interrupt_mask)

    def _reward_collision(self):    
        # Penalize collisions on selected bodies For those caused by interruption , no penalty.
        return super()._reward_collision() * (~self.interrupt_mask)

    def _reward_feet_contact_forces(self):       
        # penalize high contact forces
        reward = torch.sum(
            torch.square(self.obs_scales.contact_force*(torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.)), 
            dim=1).clip(max=2.0)
        reward[self.standing_envs_mask] *= 0
        return reward

    def _reward_termination(self):
        # Terminal reward / penalty
        penaliez = self.reset_buf * ~self.time_out_buf
        return penaliez
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        out_of_limits[:, 11:] = 0
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        reward = torch.square((self.last_dof_vel - self.dof_vel) / self.dt)
        reward[:, 11:] = 0
        return torch.sum(reward, dim=1)
    
    def _reward_dof_vel_limits(self):       
        # Penalize dof velocities too close to the limit
        dof_vel_limits = torch.clip(10 * self.velocity_level.unsqueeze(-1).repeat(1,self.num_dof), min=10, max=20)
        error = torch.sum((torch.abs(self.dof_vel[:, 11:]) - dof_vel_limits[:, 11:]).clip(min=0., max=15.), dim=1)
        rew = 1 - torch.exp(-1 * error)
        return rew
    