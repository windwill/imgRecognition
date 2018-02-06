### data preparation and loading ############################################################

from libs import *
#from tool import *

# identify the folder containing the images
DATA_DIR ='/home/klshang81/driver/imgs'
# list the label class you want to predict. For example, "safe", "using cell phone", "texting", "fatigue", etc. 
CLASS_NAMES=[
    'c0',
    'c1',
    'c2',
    'c3',
    'c4',
    'c5',
    'c6',
    'c7',
    'c8',
    'c9'
]

### transform image data to tensor
def img_to_tensor(img, mean=0, std=1.):
    img = img.astype(np.float32)
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    tensor = torch.from_numpy(img)   ##.float()
    return tensor

### image augmentation
def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img,random.randint(-1,1))
    return img


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1,0,2)  #cv2.transpose(img)
    return img

def randomContrast(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit*random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha*img  + gray
        img = np.clip(img,0.,1.)
    return img

def randomFilter(img, limit=0.5, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        alpha = limit*random.uniform(0, 1)

        kernel = np.ones((3,3),np.float32)/9*0.2

        img = alpha*cv2.filter2D(img, -1, kernel) + (1-alpha)*img
        img = np.clip(img,0.,1.)
    return img

### get the class name based on predicted probabilities #####################################
def prob_to_class_names(prob, class_names, nil=''):

    N = len(class_names)

    s=nil
    maxprob = -100
    for n in range(N):
        if prob[n]>maxprob:
            maxprob = prob[n]
            if s==nil:
                s = s + class_names[n]
            else:
                s = nil + class_names[n]
    return s

### generate data from image for model input ################################################
def create_image(image, width=256, height=256):
    h,w,c = image.shape

    if c==3:
        jpg_src=0
        M=1
        jpg_dst=0
    else:
        jpg_src=None

    img = np.zeros((h,w*M,3),np.uint8)
    if jpg_src is not None:
        jpg_blue  = image[:,:,jpg_src  ] *255
        jpg_green = image[:,:,jpg_src+1] *255
        jpg_red   = image[:,:,jpg_src+2] *255

        img[:,jpg_dst*w:(jpg_dst+1)*w] = np.dstack((jpg_blue,jpg_green,jpg_red)).astype(np.uint8)

    if height!=h or width!=w:
        img = cv2.resize(img,(width*M,height))

    return img

### transform images based on the augmentation method ################################################## 
def change_images(images, augment):

    num = len(images)
    h,w = images[0].shape[0:2]
    if augment == 'left-right' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,1)

    if augment == 'up-down' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,0)


    if augment == 'center' :
        for n in range(num):
            b=8
            image = images[n, b:h-b,b:w-b,:]
            images[n]  = cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)


    if augment == 'transpose' :
        for n in range(num):
            image = images[n]
            images[n] = image.transpose(1,0,2)


    if augment == 'rotate90' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,1)


    if augment == 'rotate180' :
        for n in range(num):
            image = images[n]
            images[n] = cv2.flip(image,-1)


    if augment == 'rotate270' :
        for n in range(num):
            image = images[n]
            image = image.transpose(1,0,2)  #cv2.transpose(img)
            images[n]  = cv2.flip(image,0)

    return images


### load image file #########################################################################
def load_image(name, width, height, ext, data_dir=DATA_DIR):

    if ext =='jpg':
        image = np.zeros((height,width,3),dtype=np.uint16)  #

        img_file  = data_dir + '/train/' + name
        image_jpg = cv2.imread(img_file,1)

        h,w = image_jpg.shape[0:2]
        if height!=h or width!=w:
            image_jpg = cv2.resize(image_jpg,(height,width))

        image = image_jpg*256

    return image

### dataset class to handle image data ######################################################
class SetDataset(Dataset):

    def __init__(self, split, transform=None, height=64, width=64, ext='jpg', label_csv='driver_imgs_list.csv', is_preload=True):
        data_dir    = DATA_DIR
        class_names = CLASS_NAMES
        num_classes = len(class_names)

        # read image names
        list = data_dir +'/split/'+ split #the list of images stored in a plain file
        with open(list) as f:
            names = f.readlines()
        names = [x.strip()for x in names]
        num   = len(names)

        if ext =='jpg':
            channel=3
        else:
            raise ValueError('SetDataset() : unknown ext !?')

        images  = None
        if is_preload==True:
            images = np.zeros((num,height,width,channel),dtype=np.uint16)
            for n in range(num):
                images[n] = load_image(names[n], width, height, ext)
                pass

        #read image labels
        df     = None
        labels = None
        if label_csv is not None:
            labels = np.zeros((num,num_classes),dtype=np.float32)

            csv_file  = data_dir + '/' + label_csv
            df = pd.read_csv(csv_file)
            for c in class_names:
                df[c] = df['classname'].apply(lambda x: 1 if c in x.split(' ') else 0)

            df1 = df.set_index('img')
            for n in range(num):
                shortname = names[n].split('/')[-1].replace('.<ext>','')
                labels[n] = df1.loc[shortname].values[2:]

        #class member
        self.transform = transform
        self.num       = num
        self.split     = split
        self.names     = names
        self.images    = images
        self.ext       = ext
        self.is_preload = is_preload
        self.width  = width
        self.height = height
        self.height = height
        self.channel = channel

        self.class_names = class_names
        self.df     = df
        self.labels = labels


    def __getitem__(self, index):

        if self.is_preload==True:
            img = self.images[index]
        else:
            img = load_image(self.names[index], self.width, self.height, self.ext)

        img = img.astype(np.float32)/65536
        if self.transform is not None:
            for t in self.transform:
                img = t(img)

        if self.labels is None:
            return img, index

        else:
            label = self.labels[index]
            return img, label, index


    def __len__(self):
        return len(self.names)

