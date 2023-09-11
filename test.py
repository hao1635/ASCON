import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import html
import util.util as util
from tqdm import tqdm
import torch
import torchvision
import ipdb
import numpy as np
import torchvision 



def save_images(images,root,index,normalize=False,value_range=None):
    saveroot=root+'/'+index+'.png'
    torchvision.utils.save_image(images,saveroot,padding = 1,normalize=normalize,value_range=value_range)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    grayscale=torchvision.transforms.Grayscale(num_output_channels=3)

    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    # dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    val_dataset = create_dataset(opt)
    val_dataloader = val_dataset.dataloader

    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)      # create a model given opt.model and other options

    util.make_dir(opt.results_dir+opt.result_name)
    util.make_dir(opt.results_dir+opt.result_name+'/'+opt.phase)
    save_images_root=opt.results_dir+opt.result_name+'/'+opt.phase
    util.make_dir(save_images_root,refresh=True)
    test_logger = util.get_logger(opt.results_dir+opt.result_name+'/test.log')

    test_running_psnr = []
    test_running_ssim=[]
    test_running_rmse=[]
    test_running_vif=0
    test_running_mad=0
    test_running_gmsd=0
    test_running_fsim=0


    pred_images=[]
    pred_images_clip=[]

    model.setup(opt) 
    model.parallelize()
    model.eval()

    for i, data in enumerate(tqdm(val_dataloader)):
        # if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #     break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        #visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        psnr,ssim,rmse=model.compute_metrics()
        test_running_psnr.append(psnr.detach().cpu().numpy())
        test_running_ssim.append(ssim.detach().cpu().numpy())
        test_running_rmse.append(rmse.detach().cpu().numpy())
        
        #ipdb.set_trace()
        if 'ASCON' in opt.model or 'gan' in opt.model:
            pred=model.fake_B
            y=model.real_B
        else:
            pred=model.output
            y=model.data_B

        pred_clip=torch.clip(pred*3000-1000,-160,240)

        
        if i % 50 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
            test_logger.info('(psnr: %.4f, ssim: %.4f,rmse: %.4f) ' % (psnr, ssim,rmse))
        

    epoch_test_psnr= np.mean(test_running_psnr)
    epoch_test_ssim= np.mean(test_running_ssim)
    epoch_test_rmse= np.mean(test_running_rmse)

    
    test_logger.info('(average: psnr: %.4f, ssim: %.4f,rmse: %.4f)' % (epoch_test_psnr, epoch_test_ssim,epoch_test_rmse))
    test_logger.info('(std_psnr: %.4f,std_ssim: %.4f,std_rmse: %.4f)' % (np.std(test_running_psnr), np.std(test_running_ssim),np.std(test_running_rmse)))
    #webpage.save()  # save the HTML


