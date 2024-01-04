import torch
import pdb
import time

def criterion(outputs, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.to(torch.float32))

def train_epoch(trn_loader, model, optim, device):
    model.train()
    batch = 0
    for i,(masks, targets) in enumerate(trn_loader):
        outputs = model(masks.to(device))
        optim.zero_grad()
        loss = criterion(outputs.to(device), targets.to(device))
        # TODO: clipgrad
        loss.backward()
        optim.step()
    print()

def eval(loader, model, device):
    with torch.no_grad():
        model.eval()
        loss = 0
        acc = 0
        for masks, targets in loader:
            outputs = model(masks.to(device))
            loss += criterion(outputs.to(device), targets.to(device)).item() * len(targets)
            probs = torch.nn.functional.sigmoid(outputs)
            preds = probs > 0.5
            # preds = outputs.argmax(dim=1)
            acc += (preds == targets.to(device)).sum().item()
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc

def train(trn_loader, val_loader, tst_loader, nepochs, model, optim, lr_patience, lr_factor, lr_min, device):
    best_loss = 0
    best_model = model.state_dict()
    patience = lr_patience
    t0 = time.time()
    for epoch in range(nepochs):
        # train one epoch
        t = time.time()
        train_epoch(trn_loader, model, optim, device)

        # do validation + patience
        val_loss, val_acc = eval(val_loader, model, device)
        print(f'Epoch {epoch:03d}/{nepochs} ({int(time.time()-t):>3d}s) -- val_loss: {val_loss:0.4f}, val_acc: {val_acc*100:02.2f}% best_loss:{best_loss:0.4f}', end='')
        lr = optim.param_groups[0]['lr']
        if val_loss < best_loss or epoch == 0:
            print(' * ', end='') # new best model 
            best_loss = val_loss
            best_model = model.state_dict()
            patience = lr_patience
        else:
            # patience
            patience -= 1
            if patience <= 0:
                # HURRY
                lr /= lr_factor
                optim.param_groups[0]['lr'] = lr
                patience = lr_patience
                print(f' LR: {lr:.2e}', end='')
                model.load_state_dict(best_model)
        print()
        if lr < lr_min:
            break
    model.load_state_dict(best_model)
    
    tst_loss, tst_acc = eval(tst_loader, model, device)
    print('-' * 80)
    print(f'tst_loss: {tst_loss:0.4f}, tst_acc: {tst_acc*100:02.2f}%')
    print(f'Total time trained: {(time.time() - t0)/3600:.2f}h')
