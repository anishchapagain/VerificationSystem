import torch
ckpt = torch.load('weights/siamese_cedar.pt', map_location='cpu', weights_only=False)
print('Epoch    :', ckpt.get('epoch'))
print('Best EER :', ckpt.get('best_eer'))
print('Config   :', ckpt.get('config'))