import torch
print(torch.__version__)
print('GPU disponible:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))