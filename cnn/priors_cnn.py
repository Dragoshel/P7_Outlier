import torch

from cnn.cnn import CNN


def load_cnn(folder_path: str) -> CNN:
    return torch.load(f'{folder_path}/cnn_model.pth')

def priors_cnn(model, images):
    probabilities = model(images)
    readable_probs = [torch.exp(prob).tolist() for prob in probabilities]
    # readable_probs = torch.exp(probabilities.cpu()).detach().numpy()
    return readable_probs