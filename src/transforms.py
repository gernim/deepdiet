import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.CenterCrop(800),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])