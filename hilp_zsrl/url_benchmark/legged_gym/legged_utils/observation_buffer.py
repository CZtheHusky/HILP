import torch

class ObservationBuffer:
    def __init__(self, num_envs, num_obs, include_history_steps, device, zero_pad=True):

        self.num_envs = num_envs
        self.num_obs = num_obs
        self.include_history_steps = include_history_steps  
        self.device = device
        self.zero_pad = zero_pad

        self.obs_buf = torch.zeros(self.num_envs, self.include_history_steps,
                                   self.num_obs, device=self.device, dtype=torch.float)
        self.atten_mask = torch.zeros(
            self.num_envs, self.include_history_steps, device=self.device, dtype=torch.long)

    def reset(self, reset_idxs, new_obs: torch.Tensor):
        if len(reset_idxs) == 0:
            return
        # new_obs shape:[reset_nums, obs_dim]
        if self.zero_pad:
            self.obs_buf[reset_idxs] = 0.
            self.obs_buf[reset_idxs, -1, :] = new_obs.clone()
        else:
            self.obs_buf[reset_idxs] = new_obs.unsqueeze(1).clone()
        self.atten_mask[reset_idxs, :] = 0
        self.atten_mask[reset_idxs, -1] = 1

    def insert(self, new_obs: torch.Tensor):  
        # Shift observations back.
        self.obs_buf[:, 0:(self.include_history_steps - 1), :] = self.obs_buf[:, 1:self.include_history_steps, :].clone()
        self.atten_mask[:, 0:(self.include_history_steps - 1)] = self.atten_mask[:, 1:self.include_history_steps].clone()
        # Add new observation.
        self.obs_buf[:, -1, :] = new_obs.clone()
        self.atten_mask[:, -1] = 1

    def get_obs_tensor_3D(self, history_ids=None):
        """Gets history of observations indexed by obs_ids.
        """
        if history_ids is not None:
            history_ids_sort, _ = torch.sort(history_ids)
            query_idx, _ = torch.sort(self.include_history_steps - history_ids_sort - 1)
            return self.obs_buf[:, query_idx, :].clone(), self.atten_mask[:, query_idx].clone()
        else:
            return self.obs_buf.clone(), self.atten_mask.clone()




