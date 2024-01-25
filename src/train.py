import copy
import os
import pdb
import time

import numpy as np
import torch
from data import dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def criterion(outputs, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.to(torch.float32))

def train_epoch(trn_loader, model, optim, device):
    """
        Trains one epoch, return loss and grad history as well as average loss and grad
    """
    model.train()
    loss_hist = []
    grad_hist = []
    for i,(masks, targets) in enumerate(tqdm(trn_loader, dynamic_ncols=True)):
        outputs = model(masks.to(device))
        optim.zero_grad()
        loss = criterion(outputs.to(device), targets.to(device))
        loss_hist.append(loss.item())
        loss.backward()
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        grad_hist.append(norm.item())
        optim.step()
        
    return loss_hist, grad_hist, np.sum(loss_hist) / len(trn_loader),  np.sum(grad_hist) / len(trn_loader)

def eval(loader, model, device):
    """
        Evaluates model on some data loader, return loss, accuracy and predictions
    """
    with torch.no_grad():

        model.eval()
        loss = 0
        acc = 0
        all_preds = []
        all_targets = []
        for masks, targets in loader:
            outputs = model(masks.to(device))
            loss += criterion(outputs.to(device), targets.to(device)).item() * len(targets)
            probs = torch.nn.functional.sigmoid(outputs)
            preds = probs > 0.5
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())
            acc += (preds == targets.to(device)).sum().item()
            
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc, all_preds, all_targets

def eval_test_split(loader, model, device):
    """
        Evaluates the model on each video in the test set separately
    """
    vid_accs = {}
    with torch.no_grad():
        model.eval()
        ds = loader.dataset
        for vid in ds.vids:
            name = os.path.basename(vid.path)
            vid_ds = dataset.TTData([vid], ds.win_size, transforms=ds.transforms.transforms, flip_prob=ds.flip_prob)
            vid_loader = torch.utils.data.DataLoader(vid_ds, batch_size=loader.batch_size, shuffle=False, num_workers=loader.num_workers)
            loss, acc, preds, targets = eval(vid_loader, model, device)
            vid_accs[name] = {'acc': acc, 'loss': loss, 'preds': preds, 'targets': targets}

    for vid in vid_accs:
        acc = vid_accs[vid]['acc']
        loss = vid_accs[vid]['loss']
        preds = vid_accs[vid]['preds']
        targets = vid_accs[vid]['targets']
        print(f'{vid} -- loss: {loss:0.4f}, acc: {acc*100:02.2f}, len: {len(vid_accs[vid]["preds"])}')

    tst_loss, tst_acc, _, _ = eval(loader, model, device)
    print(f'All test -- loss {tst_loss:0.4f}, acc: {tst_acc*100:02.2f}')

    return vid_accs

def train(trn_loader, val_loader, tst_loader, nepochs, model, optim, lr_patience, lr_factor, lr_min, device, summary_writer, has_checkpoint=False, checkpoint_freq=5):
    """

    """
    best_loss = 0
    best_model = model.state_dict()
    start_epoch = 0
    checkpoint = None
    run_dir = os.path.split(summary_writer.log_dir)[0]
    if has_checkpoint:
        checkpoint = torch.load(os.path.join(run_dir, 'model', 'checkpoint.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']
        print(f'Loaded checkpoint from epoch {start_epoch} with loss {best_loss}')

    patience = lr_patience
    t0 = time.time()
    print(f'{len(trn_loader)=}')
    if val_loader is None:
        print(f'* WARNING: val_loader is None. Using trn_loader for validation and early stopping instead.')
    for epoch in range(start_epoch,nepochs):
        # train one epoch
        t = time.time()
        training_loss_hist, training_grad_hist, avg_trn_loss, avg_trn_grad = train_epoch(trn_loader, model, optim, device)
        
        # do validation + patience
        val_loss, val_acc = 0,0
        if epoch % checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': val_loss,
                }, os.path.join(run_dir, 'model', 'checkpoint.pth'))
        
        print(f'\tEpoch {epoch+1:02d}/{nepochs} ({int(time.time()-t):>3d}s) --  ', end='')
        val_loss, val_acc = 0, 0
        if val_loader is not None:
            val_loss, val_acc, _, _ = eval(val_loader, model, device)
            print(f'val_loss: {val_loss:0.4f}, val_acc: {val_acc*100:02.2f}% | ', end='')
        if val_loader is None:
            trn_loss, trn_acc, _, _ = eval(trn_loader, model, device)
            print(f'trn_loss: {trn_loss:0.4f}, trn_acc: {trn_acc*100:02.2f} | ', end='')
            val_loss = trn_loss
        
        # if the validation loss is better than the best loss, save the model and reset patience
        # else, reduce patience and if patience is 0, reduce lr
        lr = optim.param_groups[0]['lr']
        if val_loss < best_loss or epoch == 0:
            best_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience = lr_patience
            torch.save({
                'epoch': epoch,
                'model_state_dict': copy.deepcopy(best_model),
                'optimizer_state_dict': copy.deepcopy(optim.state_dict()),
                'loss': best_loss,
                }, os.path.join(run_dir, 'model', 'best_model.pth'))
            print(f' * best_loss: {best_loss:.4f}', end='')
        else:
            # patience
            patience -= 1
            if patience <= 0 and lr_patience > 0:
                # HURRY
                lr /= lr_factor
                optim.param_groups[0]['lr'] = lr
                patience = lr_patience
                print(f' LR: {lr:.2e}', end='')
                model.load_state_dict(best_model)

        print()
        summary_writer.add_scalar('loss/trn_avg', avg_trn_loss, epoch);
        summary_writer.add_scalar('grad_norm/trn_avg', avg_trn_grad, epoch);
        summary_writer.add_scalar('loss/val', val_loss, epoch)
        summary_writer.add_scalar('acc/val', val_acc, epoch)
        summary_writer.flush()
        
        if lr < lr_min:
            break
    model.load_state_dict(best_model)
    
    print('Evaluating on test set...')
    vid_accs = eval_test_split(tst_loader, model, device)
    for vid in vid_accs:
            summary_writer.add_scalar(f'{vid}/acc', vid_accs[vid]['acc'])
            summary_writer.add_scalar(f'{vid}/loss', vid_accs[vid]['loss'])
            for i, pred in enumerate(vid_accs[vid]['preds']):
                summary_writer.add_scalars(f'{vid}/preds', {
                    'pred': 1 if pred[0] else 0,
                    'target': vid_accs[vid]['targets'][i][0]
                }, i)
                
    _ , tst_acc, _, _ = eval(tst_loader, model, device)
    summary_writer.add_scalar('acc/tst', tst_acc, epoch)
    summary_writer.flush()
    summary_writer.close()
    print('-' * 80)
    print(f'Total time trained: {(time.time() - t0)/3600:.2f}h')
    
