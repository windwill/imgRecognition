from libs import *

### set up learning rate schedule ##########################################################################
class SetRate():
    def __init__(self, pairs):
        super(SetRate, self).__init__()

        N=len(pairs)
        rates=[]
        steps=[]
        for n in range(N):
            steps.append(pairs[n][0])
            rates.append(pairs[n][1])

        self.rates = rates
        self.steps = steps

    def get_rate(self, epoch=None, num_epoches=None):

        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                + 'rates=' + str(['%0.4f' % i for i in self.rates]) + '\n' \
                + 'steps=' + str(['%0.0f' % i for i in self.steps]) + ''
        return string

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr

### open log file for writing ###############################################################################

def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f

### Logging class ############################################################################################
class Logging(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass

### replace model parameters with pretrained models excluding the skip list ###################################
def load_valid(model, pretrained_dict, skip_list=[]):
    model_dict = model.state_dict()
    pretrained_dict1 = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in skip_list }
    model_dict.update(pretrained_dict1)
    model.load_state_dict(model_dict)





