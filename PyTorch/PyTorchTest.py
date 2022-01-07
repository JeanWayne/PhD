import torch
x = torch.tensor([[[1,2],[3,4],[5,6],[7,8]]])
for i in x:
    print("!")
    print(i)
print("::::::::::::", x[:,0,:])
#print(x.narrow(1, 1, 2))