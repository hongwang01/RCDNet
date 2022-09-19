import os
import math
from decimal import Decimal
import utility
import IPython
import torch
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio 
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import torch.utils.data as udata
from torch.utils.data import DataLoader
from numpy.random import RandomState
import cv2


dataloaders = {}
class Dataset(udata.Dataset):
    def __init__(self, name, args):
        super().__init__()
        self.dataset = name
        self.args = args
        self.patch_size = args.patch_size
        self.mat_files = open(self.dataset, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
    def __len__(self):
        return self.file_num * 100

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        gt_file = file_name.split(' ')[1][:-1]
        img_file = file_name.split(' ')[0]
        O = cv2.imread(self.args.dir_data + img_file)
        b, g, r = cv2.split(O)
        input_img = cv2.merge([r, g, b])
        B = cv2.imread(self.args.dir_data + gt_file)
        b, g, r = cv2.split(B)
        gt = cv2.merge([r, g, b])
        im_pair = np.hstack((gt, input_img))
        O, B = self.crop(im_pair, self.patch_size)
        O, B = O.astype(np.float32), B.astype(np.float32)
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        sample = {'O': O, 'B': B}
        return sample
    def crop(self, img_pair, patchsize):
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        p_h, p_w = patchsize, patchsize
        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]
        return O, B

def get_dataloader(dataset_name, args):
    dataset = Dataset(dataset_name, args)
    if not dataset_name in dataloaders:
        dataloaders[dataset_name] = \
            DataLoader(dataset, batch_size=args.batch_size,
                       shuffle=True, num_workers=4, drop_last=True)
    return iter(dataloaders[dataset_name])

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.S = args.stage
        self.ckp = ckp
       # self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        loss_Bs_all = 0
        loss_Rs_all = 0
        loss_B_all = 0
        loss_R_all=0
        cnt = 0
        dt_train = get_dataloader(self.args.dir_data+'real_world.txt',self.args)
        for batch in range(1500):
            try:
                batch_t = next(dt_train)
            except StopIteration:
                dt_train = get_dataloader(self.args.dir_data+'real_world.txt',self.args)
                batch_t = next(dt_train)
            lr, hr = batch_t['O'], batch_t['B']
            loss_Bs = 0
            loss_Rs = 0
            cnt = cnt+1
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.model.zero_grad()
            self.optimizer.zero_grad()
            idx_scale = 0
            B0, ListB, ListR = self.model(lr, idx_scale)
            for j in range(self.S):
                loss_Bs = loss_Bs + 0.1*self.loss(ListB[j], hr)                    # 2022-09-19 fix the float bug
                loss_Rs = loss_Rs + 0.1*self.loss(ListR[j], lr-hr)                 # 2022-09-19 fix the float bug
            loss_B = self.loss(ListB[-1], hr)
            loss_R = 0.9 * self.loss(ListR[-1], lr-hr)
            loss_B0 = 0.1* self.loss(B0, hr)
            loss = loss_B0 + loss_Bs  + loss_Rs + loss_B + loss_R
            loss_Bs_all = loss_Bs_all + loss_Bs
            loss_B_all = loss_B_all + loss_B
            loss_Rs_all = loss_Rs_all + loss_Rs
            loss_R_all = loss_R_all + loss_R
            if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                ttt = 0
                self.optimizer.step()
            else:
                print('Skip this batch {}! (Loss: {})'.format(
                    batch + 1, loss.item()
                ))
            timer_model.hold()
            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    1500 * self.args.batch_size,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))
            timer_data.tic()
        print(loss_Bs_all/cnt)
        print(loss_B_all / cnt)
        print(loss_Rs_all / cnt)
        print(loss_R_all / cnt)
        self.loss.end_log(1500*self.args.batch_size)
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    B0,ListB,ListR = self.model(lr, idx_scale)
                    sr = utility.quantize(ListB[-1], self.args.rgb_range)    # restored background at the last stage
                    save_list = [sr]
                    if not no_eval:
                        eval_acc += utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range,
                            benchmark=self.loader_test.dataset.benchmark
                        )
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale],
                        best[0][idx_scale],
                        best[1][idx_scale] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs
