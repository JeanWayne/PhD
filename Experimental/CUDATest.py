import tensorflow as tf
import torch



##### TENSORFLOW
print("Tesnorflow:")
print("# GPUs Available:\t", len(tf.config.experimental.list_physical_devices('GPU')))


##### PYTORCH
print("\n\nPyTorch:")
a=torch.cuda.is_available()
b=torch.cuda.current_device()
c=torch.cuda.device(0)
d=torch.cuda.device_count()
e=torch.cuda.get_device_name(0)
print("cuda available:\t\t",a)
print("current device ID:\t",b)
print("device Handle:\t\t",c)
print("#device:\t\t\t",d)
print("Name:\t\t\t\t" ,e)