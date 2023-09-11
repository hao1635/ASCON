import time
import torch
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import wandb
from tqdm import tqdm
from util.util import get_logger
import ipdb

if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options

    train_logger = get_logger(opt.checkpoints_dir+'/'+opt.name+'/train.log')
    train_dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    train_dataloader = train_dataset.dataloader
    train_dataset_size = len(train_dataset)    # get the number of images in the dataset.

    batch_size = opt.batch_size

    opt.phase='test'
    val_dataset = create_dataset(opt)
    val_dataloader = val_dataset.dataloader
    val_dataset_size = len(val_dataset)

    opt.phase='train'

    model = create_model(opt)      # create a model given opt.model and other options
    train_logger.info(model.netG)
    model.setup(opt)
    model.parallelize()


    if opt.use_wandb:
        wandb.init(project=opt.project,name=opt.name)
        #wandb.watch(model)
    print('The number of training images = %d' % train_dataset_size)

    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []

    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        train_dataset.set_epoch(epoch)

        running_psnr=0
        running_loss = 0
        running_ssim=0
        running_rmse=0

        model.train()
    
        for i, data in enumerate(tqdm(train_dataloader)):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
    
            total_iters += 1
            epoch_iter += 1
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()

            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            #loss_G,psnr,ssim,rmse=model.compute_metrics()
            loss_D,loss_G,psnr,ssim,rmse=model.compute_metrics()
            
            running_loss += loss_G
            running_psnr += psnr
            running_ssim += ssim
            running_rmse += rmse
    
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                message='(epoch: %d, iters: %d,loss_D: %.6f, loss_G: %.6f,,train_psnr: %.4f, train_ssim: %.4f,train_rmse:.%.4f) ' % (epoch, epoch_iter,loss_D,loss_G, psnr, ssim,rmse)
                print(message)
                if opt.use_wandb:
                    wandb.log({ "train_loss_D": loss_D,
                                "train_loss_G": loss_G,
                                'train_psnr':psnr,
                                'train_ssim':ssim,
                                'train_rmse':rmse} )
        
        epoch_loss = running_loss/len(train_dataloader)
        epoch_psnr= running_psnr/len(train_dataloader)
        epoch_ssim=running_ssim/len(train_dataloader)
        epoch_rmse=running_rmse/len(train_dataloader)
        train_logger.info('Epoch: [{}/{}],epoch_loss: {:.6f}, train_psnr: {:.4f}, train_ssim: {:.4f},epoch_rmse:{:.4f}'.format(epoch ,opt.n_epochs, epoch_loss, epoch_psnr, epoch_ssim, epoch_rmse))

        print('validation:')   
        test_running_psnr = 0
        test_running_ssim=0
        test_running_loss = 0 
        test_running_rmse=0

        with torch.no_grad():
            model.eval()  
            for i, data in enumerate(tqdm(val_dataloader)):
                model.set_input(data)  # unpack data from data loader
                model.test()           # run inference
                
                _,loss,psnr,ssim,rmse=model.compute_metrics()
                test_running_loss += loss
                test_running_psnr += psnr
                test_running_ssim += ssim
                test_running_rmse += rmse
            
            epoch_test_loss = test_running_loss /len(val_dataloader)
            epoch_test_psnr= test_running_psnr/len(val_dataloader)
            epoch_test_ssim=test_running_ssim/len(val_dataloader)
            epoch_test_rmse=test_running_rmse/len(val_dataloader)

        train_logger.info('val:Epoch: [{}/{}],epoch_loss: {:.6f}, val_psnr: {:.4f}, val_ssim: {:.4f},test_rmse: {:.4f}'.format(epoch , opt.n_epochs, epoch_test_loss, epoch_test_psnr, epoch_test_ssim,epoch_test_rmse))


        if opt.use_wandb:
            wandb.log({"epoch_train_loss": epoch_loss,
                        'epoch_train_psnr':epoch_psnr,
                        'epoch_train_ssim':epoch_ssim,
                        'epoch_train_rmse':epoch_rmse,
                        "epoch_test_loss":epoch_test_loss,
                        'epoch_test_psnr':epoch_test_psnr,
                        'epoch_test_ssim':epoch_test_ssim,
                        'epoch_test_rmse':epoch_test_rmse,
                        'epoch':epoch
                        } )

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        torch.cuda.empty_cache()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.
    
    data=next(iter(train_dataloader))
    model.set_input(data)
    model.optimize_parameters()
    ipdb.set_trace()
    print('finish training')
