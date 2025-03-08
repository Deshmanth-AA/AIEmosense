import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# The model is running on CPU, since it is already pre-trained and doesnt require GPU
device = torch.device('cpu')
print('Running on device: {}'.format(device))

#Define MTCNN module
#Since MTCNN is a collection of neural nets and other code,
#The device must be passed in the following way to enable copying of objects when needed internally.
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709,
    device=device
)
#Function takes 2 vectors 'a' and 'b'
#Returns the cosine similarity according to the definition of the dot product
def cos_sim(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

#cos_sim returns real numbers,where negative numbers have different interpretations.
#So we use this function to return only positive values.
def cos(a,b):
    minx = -1
    maxx = 1
    return (cos_sim(a,b)- minx)/(maxx-minx)

# Define Inception Resnet V1 module (GoogLe Net)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define a dataset and data loader
dataset = datasets.ImageFolder('/content/FaceNet_FR-master/Film/Test')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=lambda x: x[0])

#Perfom MTCNN facial detection
#Detects the face present in the image and prints the probablity of face detected in the image.
aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

# Calculate the 512 face embeddings
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).cpu()

# Print distance matrix for classes.
#The embeddings are plotted in space and cosine distace is measured.
cos_sim = nn.CosineSimilarity(dim=-1, eps=1e-6)
for i in range(0,len(names)):
    emb=embeddings[i].unsqueeze(0)
    # The cosine similarity between the embeddings is given by 'dist'.
    dist =cos(embeddings[0],emb)

dists = [[cos(e1,e2).item() for e2 in embeddings] for e1 in embeddings]
# The print statement below is
#Helpful for analysing the results and for determining the value of threshold.
print(pd.DataFrame(dists, columns=names, index=names))