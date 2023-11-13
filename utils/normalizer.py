import torchvision.transforms as transforms

_norm_factor = 0.5
# Transformations describing how we wish the data set to look
# Normalize is chosen to normalize the distribution of the image tiles
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((_norm_factor,), (_norm_factor,))
])

def get_norm_factor() -> float:
    return _norm_factor

def get_transform() -> transforms.Compose:
    return _transform