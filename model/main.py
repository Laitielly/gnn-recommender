from dataloader import dataset
from model import Model
import torch
import warnings
warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state.")



model = Model(dataset).to('cpu')
model.load_state_dict(torch.load('model/model.pth',
                                 map_location=torch.device('cpu'),
                                 weights_only=True))
model.computer()
raiting = model.predict_user(0)
print(raiting)
