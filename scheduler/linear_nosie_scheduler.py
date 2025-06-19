import torch

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def add_noise(self, original, noise, t):
        """
        Adds noise to the original data based on the timestep t.
        """
        original_shape = original.shape
        batch_size = original_shape[0]
        
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod.to(original.device)[t].reshape(batch_size)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod.to(original.device)[t].reshape(batch_size)
        
        for _ in range(len(original_shape) - 1):
            sqrt_alpha_cumprod = sqrt_alpha_cumprod.unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod = sqrt_one_minus_alpha_cumprod.unsqueeze(-1)
            
        return sqrt_alpha_cumprod.to(original.device) * original + sqrt_one_minus_alpha_cumprod.to(original.device) * noise
    
    def sample_prev_timestamp(self, xt, noise_pred, t):
        """
        Samples the previous timestamp from the current noisy data xt and predicted noise.
        """
        x0 = ((xt - (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t] * noise_pred)) /
              torch.sqrt(self.alpha_cum_prod.to(xt.device)[t]))
        x0 = torch.clamp(x0, -1., 1.)
        
        mean = xt - ((self.betas.to(xt.device)[t]) * noise_pred) / (self.sqrt_one_minus_alpha_cum_prod.to(xt.device)[t])
        mean = mean / torch.sqrt(self.alphas.to(xt.device)[t])
        
        if t == 0:
            return mean, x0
        else:
            variance = (1 - self.alpha_cum_prod.to(xt.device)[t - 1]) / (1.0 - self.alpha_cum_prod.to(xt.device)[t])
            variance = variance * self.betas.to(xt.device)[t]
            sigma = variance ** 0.5
            z = torch.randn(xt.shape).to(xt.device)
            return mean + sigma * z, x0
        
