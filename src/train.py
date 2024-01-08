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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10000)
        optim.step()

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

def train(trn_loader, val_loader, tst_loader, nepochs, model, optim, lr_patience, lr_factor, lr_min, device):
    best_loss = 0
    best_model = model.state_dict()
    patience = lr_patience
    t0 = time.time()
    print(f'{len(trn_loader)=}')
    if val_loader is None:
        print(f'* WARNING: val_loader is None. Using trn_loader for validation and early stopping instead.')
    for epoch in range(nepochs):
        # train one epoch
        t = time.time()
        train_epoch(trn_loader, model, optim, device)

        # do validation + patience
        val_loss, val_acc = 0, 0
        if val_loader is not None:
            val_loss, val_acc, _, _ = eval(val_loader, model, device)
        print(f'Epoch {epoch+1:02d}/{nepochs} ({int(time.time()-t):>3d}s) -- val_loss: {val_loss:0.4f}, val_acc: {val_acc*100:02.2f}% ', end='')
        if val_loader is None:
            trn_loss, trn_acc, _, _ = eval(trn_loader, model, device)
            print(f'trn_loss: {trn_loss:0.4f}, trn_acc: {trn_acc*100:02.2f} ', end='')
            val_loss = trn_loss
        lr = optim.param_groups[0]['lr']
        if val_loss < best_loss or epoch == 0:
            best_loss = val_loss
            best_model = model.state_dict()
            patience = lr_patience
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
        if lr < lr_min:
            break
    model.load_state_dict(best_model)
    
    tst_loss, tst_acc, _, _ = eval(tst_loader, model, device)
    print('-' * 80)
    print(f'tst_loss: {tst_loss:0.4f}, tst_acc: {tst_acc*100:02.2f}%')
    print(f'Total time trained: {(time.time() - t0)/3600:.2f}h')
