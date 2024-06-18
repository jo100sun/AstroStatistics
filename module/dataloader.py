import torch as th
import pandas as pd
import torchvision as tv
import os
import yaml


class Galaxydataset(th.utils.data.Dataset):
    def __init__(self,config,train):
        self.patch_len = config["patch_len"]
        self.pad = config["pad"]
        self.len_patches = (config["img_size"]+self.pad*2)//self.patch_len
        self.path = config["path"]
        self.train = train
        self.config = config
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.path):
            os.system(f'mkdir -p {self.path}')
        if not os.path.exists(f"{self.path}/data.pt"):
            self.setup_file()
        file = th.load(f"{self.path}/data.pt")
        if self.train:
            use_idx = file["train"]
        else:
            use_idx = ~file["train"]
        self.images = file["images"][use_idx]
        self.ys = file["ys"][use_idx]
        self.filenames = file["filenames"][use_idx]

    def setup_file(self):
        image_path = "image/images"
        csv_desc=pd.read_csv('zoo2MainSpecz.csv',usecols=['dr7objid','gz2class'])
        csv_file=pd.read_csv('gz2_filename_mapping.csv')
        galaxy_types = {"Ei":0,
                        "Er":1,
                        "Sb":2,
                        "Sc":3,
                        "Ser":4}
        galaxy_types_count = {0:0,
                        1:0,
                        2:0,
                        3:0,
                        4:0}
        setup_bsz = 1024
        images = th.zeros(setup_bsz,self.len_patches*self.len_patches,self.patch_len*self.patch_len*3,dtype = th.uint8)
        ys = th.zeros(setup_bsz,dtype = th.int64)
        filenames = th.zeros(setup_bsz,dtype = th.int64)
        train = th.ones(setup_bsz,dtype=th.bool)
        j = 0
        for i in range(len(csv_desc)):
            num=csv_desc.loc[i]['dr7objid'].astype('int')
            row=csv_file.loc[csv_file['objid']==num]
            if len(row) != 0:
                filename = row['asset_id'].values[0]
                if os.path.exists(f"{image_path}/{filename}.jpg"):
                    if csv_desc.loc[i]['gz2class'] in galaxy_types:
                        image = tv.io.read_image(path=f"{image_path}/{filename}.jpg")
                        image = tv.transforms.Pad(self.pad)(image).unsqueeze(0) #img shape (3, 424+8, 424+8) 
                        y = galaxy_types[csv_desc.loc[i]['gz2class']]
                        if galaxy_types_count[y] >= 10000:
                            continue
                        galaxy_types_count[y] += 1
                        image = image.view(3,self.len_patches,self.patch_len,self.len_patches,self.patch_len).permute(1,3,2,4,0).reshape(self.len_patches*self.len_patches,self.patch_len*self.patch_len*3) # [nbatches, n_patches, patch_dim]
                        ys[j] = y
                        images[j] = image
                        filenames[j] = int(filename)
                        if j%5 == 0:
                            train[j] = False
                        j+=1
                        if (j % setup_bsz) == (setup_bsz - 1):
                            images = th.cat([images,th.zeros(setup_bsz,self.len_patches*self.len_patches,self.patch_len*self.patch_len*3,dtype = th.uint8)],dim=0)
                            ys = th.cat([ys,th.zeros(setup_bsz,dtype = th.int64)],dim=0)
                            filenames = th.cat([filenames,th.zeros(setup_bsz,dtype = th.int64)],dim=0)
                            train = th.cat([train,th.ones(setup_bsz,dtype=th.bool)],dim=0)

                        if j % 1000 == 0:
                            print(j)
                            print(galaxy_types_count)
                        if j > 49990:
                            print(j)
                            print(galaxy_types_count)
                        if j == 50000-1: 
                            break
        file_dict = {"images" : images[:j+1],
                     "ys" : ys[:j+1],
                     "filenames" : filenames[:j+1],
                     "train" : train[:j+1],
                     }
        th.save(file_dict,f'{self.path}/data.pt')
        with open(f'{self.path}/config.yaml','w') as f:
            yaml.safe_dump(self.config,f)

    def __getitem__(self,index):
        return self.images[index].to(th.float32)/255, self.ys[index]
    
    def __len__(self):
        return len(self.images)
    
class GalaxyDataloader:
    def __init__(self, config):
        print(config)
        self.trainset = Galaxydataset(config, train = True)
        self.testset = Galaxydataset(config, train = False)
        self.bsz = config["bsz"]
        self.num_workers = config["num_workers"]
        self.shuffle = config["shuffle"]

    def trainloader(self, device='cuda'):
        loader = th.utils.data.DataLoader(self.trainset, batch_size = self.bsz, num_workers = self.num_workers, shuffle = self.shuffle)
        for event in loader:
            yield [i.to(device) for i in event]

    def testloader(self, device='cuda'):
        loader = th.utils.data.DataLoader(self.testset, batch_size = self.bsz, num_workers = self.num_workers, shuffle = self.shuffle)
        for event in loader:
            yield [i.to(device) for i in event]