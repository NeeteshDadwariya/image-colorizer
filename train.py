import json
import os
import time
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from basic_model import ColorNet
from colorize_data import ColorizeData
from utils import AverageMeter, visualize_image, create_and_get_path

# Current best losses
best_losses = 1000.0
use_gpu = torch.cuda.is_available()

# Define hparams here or load them from a config file
# Opening JSON file
hyperparams = {}
with open('hyperparams.json') as json_file:
    hyperparams = json.loads(json_file.read(), object_hook=lambda d: SimpleNamespace(**d))

input_dir = os.path.join(hyperparams.data_path, 'input/')
train_dir = os.path.join(input_dir, 'train')
val_dir = os.path.join(input_dir, 'val')

output_dir = create_and_get_path(hyperparams.data_path, 'output/', True)
checkpoint_dir = create_and_get_path('.', 'checkpoints/', False)
best_model_params_path = checkpoint_dir + '/best_model_params.json'
best_model_path = checkpoint_dir + '/best_model.tar'

class Trainer:
    def __init__(self):

        self.train_dataset = ColorizeData(train_dir, split='train')
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=hyperparams.batch_size,
                                           shuffle=False, num_workers=hyperparams.workers)
        self.val_dataset = ColorizeData(val_dir, split='val')
        self.val_dataloader = DataLoader(self.val_dataset,
                                         batch_size=hyperparams.batch_size,
                                         shuffle=False, num_workers=hyperparams.workers)

    def train(self):
        global best_losses, use_gpu

        # Create Model
        #model = Net()
        model = ColorNet()

        # Use GPU if available
        if use_gpu:
            model.cuda()
            print('Loaded model onto GPU.')

        # Loss function to use
        criterion = nn.MSELoss().cuda() if use_gpu else nn.MSELoss()

        # You may also use a combination of more than one loss function
        # or create your own.
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hyperparams.learning_rate,
                                     weight_decay=hyperparams.weight_decay)

        best_state = {}

        # # train loop
        for epoch in range(hyperparams.epochs):
            # Train for one epoch, then validate
            self.train_one_epoch(self.train_dataloader, model, criterion, optimizer, epoch)
            losses = self.validate(model, criterion)

            # Save checkpoint, and replace the old best model if the current model is better
            if losses < best_losses:
                best_losses = losses
                best_state = {
                    'epoch': epoch + 1,
                    'best_losses': best_losses,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }

            print('Best Losses:', best_losses)


        self.save_model(best_state)

        #To safely save model
        time.sleep(5)

        self.predict(model)
        return best_losses

    def train_one_epoch(self, train_loader, model, criterion, optimizer, epoch):
        '''Train model on data in train_loader for a single epoch'''
        print('Starting training epoch {}'.format(epoch))

        # Prepare value counters and timers
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # Switch model to train mode
        model.train()

        # Train for single eopch
        end = time.time()
        for i, (img_gray, img_ab) in enumerate(train_loader):

            # Use GPU if available
            input_gray_variable = Variable(img_gray).cuda() if use_gpu else Variable(img_gray)
            input_ab_variable = Variable(img_ab).cuda() if use_gpu else Variable(img_ab)

            # Record time to load data (above)
            data_time.update(time.time() - end)

            # Run forward pass
            output_ab = model(input_gray_variable)  # throw away class predictions
            loss = criterion(output_ab, input_ab_variable)  # MSE

            # Record loss and measure accuracy
            losses.update(loss.item(), img_gray.size(0))

            # Compute gradient and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record time to do forward and backward passes
            batch_time.update(time.time() - end)
            end = time.time()

            # Print model accuracy -- in the code below, val refers to value, not validation
            if i % hyperparams.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))

        print('Finished training epoch {}'.format(epoch))

    def validate(self, model, criterion):
        '''Validate model on data in val_loader'''
        print('Starting validation.')

        # Prepare value counters and timers
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # Switch model to validation mode
        model.eval()

        # Run through validation set
        end = time.time()
        for i, (input_gray, input_ab) in enumerate(self.val_dataloader):

            # Record time to load data (above)
            data_time.update(time.time() - end)

            input_gray_variable, input_ab_variable = None, None
            # Use GPU if available
            with torch.no_grad():
                input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)
                input_ab_variable = Variable(input_ab).cuda() if use_gpu else Variable(input_ab)

            # Run forward pass
            output_ab = model(input_gray_variable)  # throw away class predictions
            loss = criterion(output_ab, input_ab_variable)  # check this!

            # Record loss and measure accuracy
            losses.update(loss.item(), input_gray.size(0))

            # Record time to do forward passes and save images
            batch_time.update(time.time() - end)
            end = time.time()

            # Print model accuracy -- in the code below, val refers to both value and validation
            if i % hyperparams.print_freq == 0:
                print('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    i, len(self.val_dataloader), batch_time=batch_time, loss=losses))

        print('Finished validation.')
        return losses.avg

    def predict(self, model):
        model_params = torch.load(best_model_params_path)
        epoch = model_params['epoch']

        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        for i, (input_gray, input_ab) in enumerate(self.val_dataloader):
            # Use GPU if available
            with torch.no_grad():
                input_gray_variable = Variable(input_gray).cuda() if use_gpu else Variable(input_gray)

            # Run forward pass
            output_ab = model(input_gray_variable)  # throw away class predictions

            # Save images to file
            for j in range(len(output_ab)):
                save_path = {'grayscale': output_dir, 'colorized': output_dir}
                save_name = 'img-{}-epoch-{}-%.jpg'.format(i * self.val_dataloader.batch_size + j, epoch)
                visualize_image(input_gray[j], ab_input=output_ab[j].data, show_image=False, save_path=save_path,
                                save_name=save_name)

    def save_model(self, model_params):
        torch.save(model_params, best_model_params_path)
        torch.save(model_params['state_dict'], best_model_path)
