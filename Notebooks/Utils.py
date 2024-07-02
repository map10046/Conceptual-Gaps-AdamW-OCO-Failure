import torch 
import torch.nn.functional as F
import torch.optim as optim
import os 

def compute_comparison_matrix(adam_weights_dir=None, sgd_weights_dir=None, device="cuda:0", adam_epochs=0, sgd_epochs=0):
    """
    Computes a comparison/correlation matrix between Adam and SGD weight dictionaries. By considering the weights as a vector in R^N, the correlation is computed as the normalized dot product between the two weight vectors. 
    This is done on a per-epoch basis, giving an N x M matrix where N is the number of Adam epochs and M is the number of SGD epochs.

    Args:
        adam_weights_dir (str): Directory path to the Adam weight dictionaries.
        sgd_weights_dir (str): Directory path to the SGD weight dictionaries.
        device (str): Device to use for computation (default is "cuda:0").
        adam_epochs (int): Number of Adam epochs (default is 0).
        sgd_epochs (int): Number of SGD epochs (default is 0).

    Returns:
        torch.Tensor: Comparison matrix between Adam and SGD weight dictionaries.
    """
    assert adam_weights_dir is not None and sgd_weights_dir is not None, "Please provide both weights directories"

    if adam_epochs == 0:
        adam_epochs = len(os.listdir(adam_weights_dir))

    if sgd_epochs == 0:
        sgd_epochs = len(os.listdir(sgd_weights_dir))

    base_dir_adam = os.path.join(adam_weights_dir, "adam_epoch_")
    base_dir_sgd = os.path.join(sgd_weights_dir, "sgd_epoch_")

    comparison_matrix = torch.zeros(adam_epochs, sgd_epochs).to(device)

    for i in range(adam_epochs):
        for j in range(i, sgd_epochs):
            adam_weights_dict = torch.load(base_dir_adam + str(i) + ".pt")
            sgd_weights_dict = torch.load(base_dir_sgd + str(j) + ".pt")

            assert adam_weights_dict.keys() == sgd_weights_dict.keys(), "The keys of the two weight dictionaries do not match"

            for key in adam_weights_dict.keys():
                adam_param = adam_weights_dict[key].to(device)
                sgd_param = sgd_weights_dict[key].to(device)

                flat_1 = F.normalize(torch.flatten(adam_param), p=2, dim=0)
                flat_2 = F.normalize(torch.flatten(sgd_param), p=2, dim=0)

                comparison_matrix[i, j] = torch.dot(flat_1, flat_2)
                if i < sgd_epochs and j < adam_epochs:
                    comparison_matrix[j, i] = comparison_matrix[i, j]

    return comparison_matrix.cpu().detach()


def save_weights(model, epoch, key, base_dir):
    """
    Save the weights of a model to a file.

    Args:
        model (torch.nn.Module): The model whose weights need to be saved.
        epoch (int): The current epoch number.
        key (str): The key indicating the optimizer used ('adam' or 'sgd').
        base_dir (str): The base directory where the weights will be saved.

    Raises:
        AssertionError: If the key is neither 'adam' nor 'sgd', or if the base directory is not a directory.

    Returns:
        None
    """
    assert key == 'adam' or key == 'sgd', "The key should be either 'adam' or 'sgd'"
    assert os.path.isdir(base_dir), "The base directory is not a directory"

    if key == 'adam':
        torch.save(model.state_dict(), os.path.join(base_dir, "AdamWeights/adam_epoch_" + str(epoch) + ".pt"))
    if key == 'sgd':
        torch.save(model.state_dict(), os.path.join(base_dir, "SGDWeights/sgd_epoch_" + str(epoch) + ".pt"))

    return


def training_step(model, optimizer, criterion, train_loader, device):
    """
    Perform a single training step for the given model.

    Args:
        model (torch.nn.Module): The model to train.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        train_loader (torch.utils.data.DataLoader): The data loader for training data.
        device (torch.device): The device to perform the training on.

    Returns:
        float: The average loss over the training data.
    """
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(train_loader)


def validation_step(model, criterion, val_loader, device):
    """
    Perform a validation step for the given model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        criterion (torch.nn.Module): The loss function.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to run the evaluation on.

    Returns:
        float: The average loss over the validation set.
    """
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
    return running_loss / len(val_loader)
