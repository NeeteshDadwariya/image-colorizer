import torchvision.transforms as T

input_shape_transform = T.Compose([T.ToTensor(),
                                   T.Resize(size=(256, 256)),
                                   T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
