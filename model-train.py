from libs import *
from data import *
from util import *

### import the deep learning model to train #########

from model.resnet import resnet50 as Net
#from model.densenet import densenet121 as Net
#from model.inceptionv3 import Inception3 as Net
#from model.vggnet import vgg16 as Net

### global setting ################
SIZE =  256
EXT  = 'jpg'

### write prediction result to csv
def write_prediction_csv(csv_file, predictions, split):

    class_names = CLASS_NAMES
    num_classes = len(class_names)

    with open(DATA_DIR +'/split/'+ split) as f:
        names = f.readlines()
    names = [x.strip() for x in names]
    num_test = len(names)

    assert((num_test,num_classes) == predictions.shape)
    with open(csv_file,'w') as f:
        header = 'img,'
        for c in CLASS_NAMES:
            header = header + c + ','
        header = header + '\n'
        f.write(header)
        for n in range(num_test):
            shortname = names[n].split('/')[-1]
            prediction = predictions[n]
            prediction = [x / sum(prediction) for x in prediction]
            line = ''
            for c in prediction:
                line = line + str(c) + ','
            line = line[:-1]
            f.write('%s,%s\n'%(shortname,line))

### cross entropy loss function #############################################
def multi_criterion(logits, labels):
    keep, remove = torch.max(labels,1)
    loss = nn.CrossEntropyLoss()(logits, Variable(remove))
    return loss

### f-measure based on precise and recall ###################################
def multi_f_measure( probs, labels, beta=1 ):

    SMALL = 1e-6
    batch_size = probs.size()[0]

    l = labels
    keep, remove = torch.max(probs, 1, keepdim=True)
    p = (probs>=keep).float()
    num_pos     = torch.sum(p,  1)
    num_pos_hat = torch.sum(l,  1)
    tp          = torch.sum(l*p,1)
    precise     = tp/(num_pos     + SMALL)
    recall      = tp/(num_pos_hat + SMALL)

    fs = (1+beta*beta)*precise*recall/(beta*beta*precise + recall + SMALL)
    f  = fs.sum()/batch_size
    return f

### prediction using a calibrated model ###########################################
def predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    for iter, (images, indices) in enumerate(test_loader, 0):

        # forward
        logits, probs = net(Variable(images.cuda(),volatile=True))

        batch_size = len(images)
        test_num  += batch_size
        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    return predictions


### calculate prediction accuracy and cross entropy loss #################################
def evaluate( net, test_loader ):

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

    assert(test_num == test_loader.dataset.num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc

### model prediction, cross entropy loss, model acccuracy evaluation #####################
def evaluate_and_predict(net, test_loader):

    test_dataset = test_loader.dataset
    num_classes  = len(test_dataset.class_names)
    predictions  = np.zeros((test_dataset.num,num_classes),np.float32)

    test_num  = 0
    test_loss = 0
    test_acc  = 0
    for iter, (images, labels, indices) in enumerate(test_loader, 0):

        logits, probs = net(Variable(images.cuda(),volatile=True))
        loss  = multi_criterion(logits, labels.cuda())

        batch_size = len(images)
        test_acc  += batch_size*multi_f_measure(probs.data, labels.cuda())
        test_loss += batch_size*loss.data[0]
        test_num  += batch_size

        start = test_num-batch_size
        end   = test_num
        predictions[start:end] = probs.data.cpu().numpy().reshape(-1,num_classes)

    assert(test_dataset.num==test_num)
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc, predictions


### data augmentation ####################################################################
def augment(x, u=0.75):
    x = randomFlip(x, u=0.5)
    #x = randomTranspose(x, u=0.5)
    #x = randomContrast(x, limit=0.2, u=0.5)
    #x = randomFilter(x, limit=0.5, u=0.2)
    return x

### model training #######################################################################
def training():
    
    # specify the folder where you will save the results
    out_dir ='/home/klshang81/driver/results/debug'
    os.makedirs(out_dir +'/calibrated', exist_ok=True)

    # Write basic run information to run log
    log = Logging()
    log.open(out_dir+'/log.train.txt',mode='a')
    log.write('\n--- [Run Started at %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 32))
    log.write('\tout_dir = %s\n' % out_dir)
    log.write('\n')

    # Preprae the data
    log.write('** dataset **\n')
    num_classes = len(CLASS_NAMES)
    batch_size  = 72

    train_dataset = SetDataset('debug814', #train dataset
                                    transform=[
                                        #lambda x: augment(x),  # you can use a transformed image for the training
                                        lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE, width=SIZE, ext=EXT, is_preload=False,
                                    label_csv='driver_imgs_list.csv' # file includes the labels for each image
                                    )
    train_loader  = DataLoader(
                        train_dataset,
                        sampler = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 3,
                        pin_memory  = True)

    test_dataset = SetDataset('debug814',#validation dataset
                                    transform=[
                                        lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE, width=SIZE, ext=EXT,
                                    is_preload=True,
                                    label_csv='driver_imgs_list.csv' #'train_v2.csv'
                                    )
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 2,
                        pin_memory  = True)


    height, width , in_channels   =  test_dataset.height, test_dataset.width, test_dataset.channel
    log.write('\t(height,width) = (%d, %d)\n'%(height,width)) # image height and width
    log.write('\tin_channels    = %d\n'%(in_channels)) # image channels (could be 1 for gray scale, 3 for RGB and more)
    log.write('\ttrain_dataset  = %s\n'%(train_dataset.split))
    log.write('\ttrain_dataset # of imgs    = %d\n'%(train_dataset.num))
    log.write('\ttest_dataset   = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset # of imgs     = %d\n'%(test_dataset.num))
    log.write('\tbatch_size           = %d\n'%batch_size)
    log.write('\n')

    ## deep learning net ###################################################
    log.write('** net setting **\n')
    net = Net(in_shape = (in_channels, height, width), num_classes=num_classes)

    net.cuda()
    log.write('%s\n\n'%(type(net)))
    log.write('\n')


    ## learning rate and optimiser #########################################
    #LR = SetRate([ (0,0.1),  (10,0.01),  (25,0.005),  (35,0.001), (40,0.0001), (45,-1)])
    LR = SetRate([ (0,0.1), (2,-1)])

    num_epoches = 50  #100 #50
    it_print    = 10
    epoch_test  = 1
    epoch_save  = 5

    #SGD optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)  ###0.0005

    ## load pretrained model ###############################################
    # To use pretrained model, provide the location of the pretrained model file.
    #pretrained_file = None
    pretrained_file = '/home/klshang81/driver/pretrain/resnet50-19c8e357.pth'
    #pretrained_file = '/home/klshang81/driver/pretrain/densenet121-241335ed.pth'
    #pretrained_file = '/home/klshang81/driver/pretrain/vgg16-397923af.pth'
    #pretrained_file = '/home/klshang81/driver/pretrain/inception_v3_google-1a9a5a14.pth'

    # Remove list contains the parameters that is not included in pretrained model.
    skip_list = ['fc.weight', 'fc.bias']
    if pretrained_file is not None:
        pretrained_dict = torch.load(pretrained_file)
        load_valid(net, pretrained_dict, skip_list=skip_list)

    ## model training ######################################################
    log.write('** model training **\n')

    log.write(' epoch   iter   learning_rate  |  train_loss  |  train_accuracy  |  valid_loss  |  valid_accuracy  | min\n')
    log.write('--------------------------------------------------------------------------------------------------------\n')

    smooth_loss = 0.0
    train_loss  = np.nan
    train_acc   = np.nan
    test_loss   = np.nan
    test_acc    = np.nan
    time = 0

    start0 = timer()
    for epoch in range(0, num_epoches):  # loop over the dataset multiple times
        start = timer()

        lr =  LR.get_rate(epoch, num_epoches)
        if lr<0 :break

        adjust_learning_rate(optimizer, lr)
        rate =  get_learning_rate(optimizer)[0] #check

        sum_smooth_loss = 0.0
        sum = 0
        net.cuda().train()
        num_its = len(train_loader)
        for it, (images, labels, indices) in enumerate(train_loader, 0):

            logits, probs = net(Variable(images.cuda()))
            loss  = multi_criterion(logits, labels.cuda())
 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_smooth_loss += loss.data[0]
            sum += 1

            if it % it_print == it_print-1:
                smooth_loss = sum_smooth_loss/sum
                sum_smooth_loss = 0.0
                sum = 0

                train_acc  = multi_f_measure(probs.data, labels.cuda())
                train_loss = loss.data[0]

                print('\r%5.1f   %5d    %0.4f   |  %0.4f  |  %0.4f  | ... ' % \
                        (epoch + it/num_its, it + 1, rate, smooth_loss, train_acc),\
                        end='',flush=True)

        end = timer()
        time = (end - start)/60

        if epoch % epoch_test == epoch_test-1  or epoch == num_epoches-1:
            net.cuda().eval()
            test_loss,test_acc = evaluate(net, test_loader)

            print('\r',end='',flush=True)
            log.write('%5.1f   %5d    %0.4f   |  %0.4f  |  %0.4f  |  %0.4f  |  %6.4f  |  %3.1f min \n' % \
                    (epoch + 1, it + 1, rate, smooth_loss, train_acc, test_loss,test_acc, time))

    end0  = timer()
    time0 = (end0 - start0) / 60

    torch.save(net,out_dir +'/calibrated/final.torch')

### predicting based on calibrated model ##################################################################
def predicting():


    out_dir ='/home/klshang81/driver/results/debug'
    model_file = out_dir +'/calibrated/final.torch'

    log = Logging()
    log.open(out_dir+'/prediction/log.predicting.txt',mode='a')
    log.write('\n--- [prediction started %s] %s\n\n' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 64))
    log.write('\n')

    # read in dataset ############################################################
    log.write('** dataset setting **\n')
    batch_size    = 96  #how many images are predicted in one batch

    test_dataset = SetDataset('debug814',
                                    transform=[
                                         lambda x: img_to_tensor(x),
                                    ],
                                    height=SIZE,width=SIZE,
                                    label_csv=None)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True)

    height, width , in_channels = test_dataset.images[0].shape
    log.write('\t(height,width)    = (%d, %d)\n'%(height,width))
    log.write('\tin_channels       = %d\n'%(in_channels))
    log.write('\ttest_dataset = %s\n'%(test_dataset.split))
    log.write('\ttest_dataset # of imgs  = %d\n'%(test_dataset.num))
    log.write('\tbatch_size        = %d\n'%batch_size)
    log.write('\n')


    # deep learning net
    log.write('** net setting **\n')
    log.write('\tmodel_file = %s\n'%model_file)
    log.write('\n')

    net = torch.load(model_file)
    net.cuda().eval()


    # predicting
    augments = ['default', 'left-right']
    num_augments = len(augments)
    num_classes  = len(test_dataset.class_names)
    test_num     = test_dataset.num
    test_dataset_images = test_dataset.images.copy()

    all_predictions = np.zeros((num_augments,test_num,num_classes),np.float32)
    for a in range(num_augments):
        augment = augments[a]
        log.write('** predict image: %s **\n'%augment)

        test_dataset.images = change_images(test_dataset_images,augment)
        predictions = predict( net, test_loader )
        all_predictions[a] = predictions

    for a in range(num_augments):

        augment = augments[a]
        predictions = all_predictions[a]

        test_dir = out_dir +'/prediction/'+ augment
        os.makedirs(test_dir, exist_ok=True)

        assert(type(test_loader.sampler)==torch.utils.data.sampler.SequentialSampler)
        write_prediction_csv(test_dir + '/results.csv', predictions, test_dataset.split )

    pass


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    training()
    predicting()

    print('\nrun finished!')
