import torchvision.transforms as T

input_shape_transform = T.Compose([T.ToTensor(),
                                   T.Resize(size=(256, 256)),
                                   ])
