import torch
import torch.nn.functional as F
from torch import nn


class EppsPulley(nn.Module):
    def __init__(self, points: int = 17, t_max: float = 5.0) -> None:
        super().__init__()
        if points < 3:
            raise ValueError("points must be at least 3")
        t = torch.linspace(0.0, t_max, points)
        dt = t_max / (points - 1)
        weights = torch.full((points,), 2.0 * dt)
        weights[[0, -1]] = dt
        phi = torch.exp(-0.5 * t.square())
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        count = samples.shape[-2]
        values = samples.unsqueeze(-1) * self.t
        cosine = torch.cos(values).mean(dim=-3)
        sine = torch.sin(values).mean(dim=-3)
        error = (cosine - self.phi).square() + sine.square()
        return (error @ self.weights) * count


class SIGReg(nn.Module):
    def __init__(
        self, projections: int = 1024, points: int = 17, t_max: float = 5.0
    ) -> None:
        super().__init__()
        self.projections = projections
        self.test = EppsPulley(points=points, t_max=t_max)
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=embeddings.device)
        generator.manual_seed(int(self.step.item()))
        directions = torch.randn(
            embeddings.shape[-1],
            self.projections,
            device=embeddings.device,
            dtype=torch.float32,
            generator=generator,
        )
        directions = F.normalize(directions, dim=0)
        self.step.add_(1)
        projections = embeddings.float() @ directions
        return self.test(projections).mean()


class DeJEPALoss(nn.Module):
    def __init__(self, sigreg_weight: float = 0.001, projections: int = 1024) -> None:
        super().__init__()
        self.sigreg_weight = sigreg_weight
        self.sigreg = SIGReg(projections=projections)

    def forward(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        prediction = F.mse_loss(first, second)
        regularization = self.sigreg(torch.stack((first, second)))
        total = (
            1.0 - self.sigreg_weight
        ) * prediction + self.sigreg_weight * regularization
        return {"loss": total, "prediction": prediction, "sigreg": regularization}
