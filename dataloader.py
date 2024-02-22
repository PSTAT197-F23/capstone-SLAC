from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TrainingData(Dataset):
    
    def __init__(self, file, transform = None, target_transform = None, **kwargs):
        '''
        A class to initialize our training data.
        Args:
            file: string (master.hdf5)
            transform: callable function to apply to images
            target_transform: callable function to apply to target

        Initiate: data = TrainingData("master...")
        '''
        self.file = file
        self.transform = transform
        self.target_transform = target_transform
        self.imgs, self.labels = self._extract_data()

    def __len__(self):
        '''
        Grabs the number of observations
        '''
        assert len(self.imgs) == len(self.labels)
        return len(self.imgs)

    def __getitem__(self, idx):
        '''
        this is how we can select examples
        '''
        img = self.imgs[idx].astype(np.float32)
        label = self.labels[idx].reshape(2,1).astype(np.float32)
        
        if self.transform:
            img = self.tranform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label

    def _extract_data(self, **kwargs):
        '''
        Extracts the images and the labels from the hdf5 label.
        Returns: Tuple --> (images, labels)
        '''
        # open our master file 
        h = h5py.File(self.file, "r")
        # grab the images 
        imgs = h['images']
        # grab the labels 
          # first find the indexes with the attribbutes 
        idx1 = list(h['labels'].attrs['names']).index('cent_fast_train')
        idx2 = list(h['labels'].attrs['names']).index('cent_slow_train')
        # then we extract 
        labels = h['labels'][:, [idx1, idx2]]
    
        return imgs, labels
