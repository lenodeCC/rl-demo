import torch
from torch.distributions import Categorical
from .model import ActorCritic

class PlayerAgent:
    def __init__(self, state_dim, action_dim, n_latent_var, device):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_latent_var = n_latent_var
        self.device = device

        self.policy = self.create_policy()

    def create_policy(self):
        return ActorCritic(self.state_dim, self.action_dim, self.n_latent_var).to(self.device)
    
    def act(self, state, memory):
        state = torch.tensor(state).float().to(self.device) 
        action_probs = self.policy.forward(state)
        dist = Categorical(action_probs) #按照给定的概率分布来进行采样
        action = dist.sample()
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        
        return action.item()