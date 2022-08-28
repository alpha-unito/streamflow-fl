import torch

torch.cuda.is_available()
dev = torch.device('cuda')

def eval_model(model, data_loader):
    model = model.to(dev)
    model.eval()
    true_preds, num_preds = 0., 0.

    with torch.no_grad():
        for data_inputs, data_labels in data_loader:
            data_inputs = data_inputs.to(dev)
            data_labels = data_labels.to(dev)
            preds = model(data_inputs)
            preds = preds.squeeze(dim=1)

            _,pred_label = torch.max(preds, dim = 1)
            true_preds += (pred_label == data_labels).sum() 
            
            num_preds += data_labels.shape[0]

    acc = true_preds / num_preds
    print("Accuracy of the model: %4.2f%%" % (100.0*acc))
