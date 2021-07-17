import os

for i in range(1, 501):

    os.system('python Finalv_get_data.py --batch-size 12 --zdim 256 --data-path ./H5_Format_Data/Sess1/Sub{}  --resume ../Trained_VAE/Checkpoint/checkpoint99.pth.tar  --mode encode --z-path ./Testing_Data_Z/Sess1/Sub{}/ '.format(i,i))
    os.system('python Finalv_get_data.py --batch-size 12 --zdim 256 --data-path ./H5_Format_Data/Sess2/Sub{}  --resume ../Trained_VAE/Checkpoint/checkpoint99.pth.tar  --mode encode --z-path ./Testing_Data_Z/Sess2/Sub{}/ '.format(i,i))


