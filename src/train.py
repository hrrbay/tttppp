import torch
import pdb
import os
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def criterion(outputs, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.to(torch.float32))

def train_epoch(trn_loader, model, optim, device):
    model.train()
    loss_hist = []
    grad_hist = []
    for i,(masks, targets) in enumerate(tqdm(trn_loader)):
        outputs = model(masks.to(device))
        optim.zero_grad()
        loss = criterion(outputs.to(device), targets.to(device))
        loss_hist.append(loss.item())
        # TODO: clipgrad
        loss.backward()
        grads = [
            param.grad.detach().flatten()
            for param in model.parameters()
            if param.grad is not None
        ]
        norm = torch.cat(grads).norm()
        grad_hist.append(norm.item())
        optim.step()
        
    # training accuracy as well?
    return loss_hist, grad_hist

def eval(loader, model, device):
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
            # preds = outputs.argmax(dim=1)
            acc += (preds == targets.to(device)).sum().item()
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc, all_preds, all_targets

def train(trn_loader, val_loader, tst_loader, nepochs, model, optim, lr_patience, lr_factor, lr_min, device, summary_writer, has_checkpoint=False, checkpoint_freq=5):
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
        training_loss_hist, training_grad_hist = train_epoch(trn_loader, model, optim, device)
        
        # do validation + patience
        val_loss, val_acc = 0,0#eval(val_loader, model, device)
        if epoch % checkpoint_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': val_loss,
                }, os.path.join(run_dir, 'model', 'checkpoint.pth'))
        
        val_loss, val_acc = 0, 0
        if val_loader is not None:
            val_loss, val_acc, _, _ = eval(val_loader, model, device)
        print(f'Epoch {epoch+1:02d}/{nepochs} ({int(time.time()-t):>3d}s) -- val_loss: {val_loss:0.4f}, val_acc: {val_acc*100:02.2f}% | ', end='')
        if val_loader is None:
            trn_loss, trn_acc, _, _ = eval(trn_loader, model, device)
            print(f'trn_loss: {trn_loss:0.4f}, trn_acc: {trn_acc*100:02.2f} ', end='')
            val_loss = trn_loss
        lr = optim.param_groups[0]['lr']
        if val_loss < best_loss or epoch == 0:
            best_loss = val_loss
            best_model = model.state_dict()
            patience = lr_patience
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model,
                'optimizer_state_dict': optim.state_dict(),
                'loss': best_loss,
                }, os.path.join(run_dir, 'model', 'best_model.pth'))
            print(f' * best_loss: {best_loss:.4f}', end='') # new best model 
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
        for batch, training_loss in enumerate(training_loss_hist):
            summary_writer.add_scalar('loss/trn', training_loss, batch + epoch * len(trn_loader))
        for batch, training_grad in enumerate(training_grad_hist):
            summary_writer.add_scalar('grad_norm/trn', training_grad, batch + epoch * len(trn_loader))

        summary_writer.add_scalar('loss/val', val_loss, epoch)
        summary_writer.add_scalar('acc/val', val_acc, epoch)
        summary_writer.flush()
        
        if lr < lr_min:
            break
    summary_writer.close()
    model.load_state_dict(best_model)
    
    print('Evaluating on test set...')
    tst_loss, tst_acc, _, _ = eval(tst_loader, model, device)
    print('-' * 80)
    print(f'tst_loss: {tst_loss:0.4f}, tst_acc: {tst_acc*100:02.2f}%')
    print(f'Total time trained: {(time.time() - t0)/3600:.2f}h')
    
