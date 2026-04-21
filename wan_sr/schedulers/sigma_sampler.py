from dataclasses import dataclass

import torch


@dataclass
class SigmaSampler:
    mode: str = "mid"

    def sample(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        if self.mode == "clean":
            return torch.zeros(batch_size, device=device)
        if self.mode == "uniform":
            return torch.rand(batch_size, device=device)
        if self.mode != "mid":
            raise ValueError(f"unknown sigma mode: {self.mode}")
        return sample_mid_sigmas(batch_size, device=device)


def sample_mid_sigmas(batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    bucket = torch.rand(batch_size, device=device)
    sigma = torch.empty(batch_size, device=device)

    mid = bucket < 0.70
    broad = (bucket >= 0.70) & (bucket < 0.90)
    late_clean = bucket >= 0.90

    sigma[mid] = torch.empty(mid.sum(), device=device).uniform_(0.35, 0.70)
    sigma[broad] = torch.empty(broad.sum(), device=device).uniform_(0.20, 0.85)
    sigma[late_clean] = torch.empty(late_clean.sum(), device=device).uniform_(0.00, 0.20)
    return sigma
