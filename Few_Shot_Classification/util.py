import torch

def build_sim(feature):
    feature_norm = feature.div(torch.norm(feature, p=2, dim=-1, keepdim=True))
    sim = torch.matmul(feature_norm, feature_norm.transpose(1, 2))
    return sim


