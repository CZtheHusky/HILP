
import sys
from isaacgym import gymapi
from isaacgym import gymutil
import numpy as np
import torch

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless, sample_commands_from_joystick=False):
        self.gym = gymapi.acquire_gym()

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.sim_device = sim_device
        sim_device_type, self.sim_device_id = gymutil.parse_device_str(self.sim_device)
        self.headless = headless

        # env device is GPU only if sim is on GPU and use_gpu_pipeline=True, otherwise returned tensors are copied to CPU by physX.
        if sim_device_type=='cuda' and sim_params.use_gpu_pipeline:
            self.device = self.sim_device
        else:
            self.device = 'cpu'

        # graphics device for rendering, -1 for no rendering
        self.graphics_device_id = self.sim_device_id
        if self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        self.learning_iter = 0
        self.num_partial_obs = cfg.env.num_partial_obs  

        if cfg.env.stack_history_obs:
            self.include_history_steps = ((cfg.env.include_history_steps-1)//cfg.env.obs_interval + 1)
            self.partial_obs_buf = torch.zeros(self.num_envs, self.include_history_steps, self.num_partial_obs, device=self.device, dtype=torch.float)
        else:
            self.include_history_steps = None
            self.partial_obs_buf = torch.zeros(self.num_envs, self.num_partial_obs, device=self.device, dtype=torch.float)
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.evalaute_episode_length_buf = torch.zeros_like(self.episode_length_buf)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.large_ori_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.low_height_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None

        self.extras = {}
        
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None
        self.sample_commands_from_joystick = sample_commands_from_joystick

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            
            # Define the mapping of joystick and keyboard.
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT, "RightStick_LEFT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT, "RightStick_RIGHT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_UP, "RightStick_UP"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_DOWN, "RightStick_DOWN"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_A, "LeftStick_LEFT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_D, "LeftStick_RIGHT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_W, "LeftStick_UP"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_S, "LeftStick_DOWN"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_CONTROL, "L1"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_SHIFT, "L2"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_CONTROL, "R1"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_RIGHT_SHIFT, "R2"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_T, "up"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_G, "down"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_F, "left"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_H, "right"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_4, "X"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_6, "Y"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_8, "A"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_2, "B"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_ENTER, "start"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_NUMPAD_0, "F2"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_SUPER, "F1"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_LEFT_ALT, "select"
            )
        self.key_state = {
            "R1": 0,
            "L1": 0,
            "start": 0,
            "select": 0,
            "R2": 0,
            "L2": 0,
            "F1": 0,
            "F2": 0,
            "A": 0,
            "B": 0,
            "X": 0,
            "Y": 0,
            "up": 0,
            "right": 0,
            "down": 0,
            "left": 0,
        }
        self.stick_state = {
            'lx': 0,
            'ly': 0,
            'rx': 0,
            'ry': 0
        }
        self.last_key_state = self.key_state.copy()
        self.last_stick_state = self.stick_state.copy()

    def training_curriculum(self, num_steps=1):
        self.learning_iter += num_steps

    def get_observations(self):
        return self.partial_obs_buf
    
    def get_privileged_observations(self):
        return self.obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError

    def update_joy_stick_state(self):
        # High frequency.
        self.last_key_state = self.key_state.copy()
        self.last_stick_state = self.stick_state.copy()
        lx = 0
        ly = 0 
        rx = 0
        ry = 0
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value >0:
                print(evt.action)
            if evt.action in self.key_state.keys():
                self.key_state[evt.action] = evt.value
            elif evt.action == "LeftStick_LEFT" and evt.value >0:
                lx -= 1
            elif evt.action == "LeftStick_RIGHT" and evt.value >0:
                lx += 1
            elif evt.action == "LeftStick_UP" and evt.value >0:
                ly += 1
            elif evt.action == "LeftStick_DOWN" and evt.value>0:
                ly -= 1
            elif evt.action == "RightStick_LEFT" and evt.value >0:
                rx -= 1
            elif evt.action == "RightStick_RIGHT" and evt.value >0:
                rx += 1
            elif evt.action == "RightStick_UP" and evt.value >0:
                ry += 1
            elif evt.action == "RightStick_DOWN" and evt.value>0:
                ry -= 1
            elif evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync
        self.stick_state['lx'] = lx
        self.stick_state['rx'] = rx
        self.stick_state['ly'] = ly
        self.stick_state['ry'] = ry
        if lx != 0 or ly != 0 or rx != 0 or ry != 0:
            print(f"lx: {lx}, ly: {ly}, rx: {rx}, ry: {ry}")
        if self.sample_commands_from_joystick:
            self._resample_commands_from_joystick()
        self.gym.poll_viewer_events(self.viewer)
    
    #Key State:        
    def _on_press(self, key):
        return self.key_state[key] and (not self.last_key_state[key])
    def _on_release(self, key):
        return (not self.key_state[key]) and self.last_key_state[key]
    def _pressed(self, key):
        return self.key_state[key] and self.last_key_state[key]
    def _released(self, key):
        return (not self.key_state[key]) and (not self.last_key_state[key])

    def _resample_commands_from_joystick(self):
        # Should be implemented in the inherited class according to different tasks.
        pass

    def render(self, sync_frame_time=True):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()
            # check for keyboard events
            # for evt in self.gym.query_viewer_action_events(self.viewer):
            #     if evt.action == "QUIT" and evt.value > 0:
            #         sys.exit()
            #     elif evt.action == "toggle_viewer_sync" and evt.value > 0:
            #         self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                if sync_frame_time:
                    self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)