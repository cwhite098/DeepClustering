from main import *
from sklearn.metrics import confusion_matrix, accuracy_score
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#x, y = load_mnist()

x, y, x_test = load_tilts()

test_img = x_test[0,:]
print(test_img)

# Getting the model##############################################
autoencoder = AutoEncoder().to(device)
ae_save_path = 'saves/sim_autoencoder.pth'

if os.path.isfile(ae_save_path):
    print('Loading {}'.format(ae_save_path))
    checkpoint = torch.load(ae_save_path)
    autoencoder.load_state_dict(checkpoint['state_dict'])
else:
    print("=> no checkpoint found at '{}'".format(ae_save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }
dec_save_path='saves/dec.pth'
dec = DEC(n_clusters=2, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
if os.path.isfile(dec_save_path):
    print('Loading {}'.format(dec_save_path))
    checkpoint = torch.load(dec_save_path)
    dec.load_state_dict(checkpoint['state_dict'])
else:
    print("=> no checkpoint found at '{}'".format(dec_save_path))
    checkpoint = {
        "epoch": 0,
        "best": float("inf")
    }
#####################################################

batch = x_test
img = batch.float()
print('img shape:' + str(img.shape))
img = img.to(device)

output = dec(img)
print('output shape: '+ str(output.shape))
print(output)

out = output.argmax(1)
print(out.shape)
print(out)
print(y.shape)

print(confusion_matrix(y.cpu().numpy(),out.cpu().numpy()))