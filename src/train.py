import torch
import pdb

def criterion(outputs, targets):
    return torch.nn.functional.binary_cross_entropy_with_logits(outputs, targets.to(torch.float32))

def train_epoch(trn_loader, model, optim, device):
    model.train()
    batch = 0
    for masks, targets in trn_loader:
        outputs = model(masks.to(device))
        optim.zero_grad()
        loss = criterion(outputs.to(device), targets.to(device))
        # TODO: clipgrad
        loss.backward()
        optim.step()
        batch += 1

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
            pdb.set_trace()
            # preds = outputs.argmax(dim=1)
            acc += (preds == targets.to(device)).sum().item()
        loss /= len(loader.dataset)
        acc /= len(loader.dataset)
        return loss, acc

def train(trn_loader, val_loader, tst_loader, nepochs, model, optim, max_patience, lr_factor, device):
    best_loss = 0
    best_model = model.state_dict()
    for epoch in range(nepochs):
        # train one epoch
        train_epoch(trn_loader, model, optim, device)

        patience = 2
        # validate
        val_loss, val_acc = eval(val_loader, model, device)
        print(f'Epoch {epoch:03d}/{nepochs} patience {patience} -- val_loss: {val_loss:0.4f}, val_acc: {val_acc*100:02.2f}% best_loss:{best_loss:0.4f}', end='')
        if val_loss < best_loss or epoch == 0:
            best_loss = val_loss
            best_model = model.state_dict()
            print(' * ', end='')
            patience = 2
        else:
            # patience
            patience -= 1
            if patience <= 0:
                # HURRY
                lr = optim.param_groups[0]['lr']
                lr /= lr_factor
                optim.param_groups[0]['lr'] = lr
                patience = 2
                print(f' LR: {lr:.4f}', end='')
                model.set_state_dict(best_model)
        print()

        # patience
