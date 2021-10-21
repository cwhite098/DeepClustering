from main import *
from sklearn.metrics import confusion_matrix, accuracy_score
from metrics import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#x, y = load_mnist()

x, y, x_test = load_tilts()

test_img = x[0,:]
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
dec = DEC(n_clusters=6, autoencoder=autoencoder, hidden=10, cluster_centers=None, alpha=1.0).to(device)
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
print(y)

print(confusion_matrix(y.cpu().numpy(), out.cpu().numpy()))
print(accuracy_score(y.cpu().numpy(), out.cpu().numpy())*100)
print(acc(y.cpu().numpy(), out.cpu().numpy())*100)


crash_ref = y.nonzero()
nocrash_ref = (y-1)*(-1)
nocrash_ref = nocrash_ref.nonzero()


print(x_test.shape)
print(x.shape)
x_class0 = x_test[nocrash_ref,:].reshape(79,72)
x_class1 = x_test[crash_ref,:].reshape(16,72)
print(x_class0.shape)

dec.visualise_labelled(x.float().to(device), x_class0.float().to(device), x_class1.float().to(device))
