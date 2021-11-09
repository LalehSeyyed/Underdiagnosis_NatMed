import torch
import os


def Saved_items(epoch_losses_train, epoch_losses_val, time_elapsed, batch_size):
    """
    Saves checkpoint of torchvision model during training.
    Args:

        epoch_losses_train: training losses over epochs
        epoch_losses_val: validation losses over epochs

    """
    print('saving')
    state2 = {
        'epoch_losses_train': epoch_losses_train,
        'epoch_losses_val': epoch_losses_val,
        'time_elapsed': time_elapsed,
        "batch_size": batch_size
    }
    torch.save(state2, 'results/Saved_items')

def checkpoint(model, best_loss, best_epoch, LR):
    """
    Saves checkpoint of torchvision model during training.
    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'LR': LR
    }
    torch.save(state, 'results/checkpoint')


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, LR, dest_dir):
    """
    Saves model checkpoint.
    :param epoch: epoch number
    :param stn: model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :dest_dir
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, "learning rate:": LR }
    filename = 'epoch' + str(epoch) + '.pth.tar'
    filename = os.path.join(dest_dir, filename)
    torch.save(state, filename)
    
# Function to preprocess the data and the subgroups    
def preprocess(split):
    details = pd.read_csv("/scratch/gobi2/projects/ml4h/datasets/CheXpert/map.csv")
    if 'Atelectasis' in split.columns:
        details = details.drop(columns=['No Finding',
       'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
       'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
       'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
       'Support Devices'])
    split = split.merge(details, left_on="Path", right_on="Path")
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81, 'Male', 'Female'], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-", 'M', 'F'])
    return split

def random_split(map_path, total_subject_id, split_portion):
    df = pd.read_csv(map_path)
    subject_df = pd.read_csv(total_subject_id)
    subject_df['random_number'] = np.random.uniform(size=len(subject_df))


    train_id = subject_df[subject_df['random_number'] <= split_portion[0]]
    valid_id = subject_df[(subject_df['random_number'] > split_portion[0]) & (subject_df['random_number'] <= split_portion[1])]
    test_id = subject_df[subject_df['random_number'] > split_portion[1]]

    train_id = train_id.drop(columns=['random_number'])
    valid_id = valid_id.drop(columns=['random_number'])
    test_id = test_id.drop(columns=['random_number'])

    train_id.to_csv("train_id.csv", index=False)
    valid_id.to_csv("valid_id.csv", index=False)
    test_id.to_csv("test_id.csv", index=False)

    train_df = train_id.merge(df, left_on="subject_id", right_on="subject_id")
    valid_df = valid_id.merge(df, left_on="subject_id", right_on="subject_id")
    test_df = test_id.merge(df, left_on="subject_id", right_on="subject_id")

    print(len(train_df))
    print(len(valid_df))
    print(len(test_df))

    train_df.to_csv("new_train.csv", index=False)
    valid_df.to_csv("new_valid.csv", index=False)
    test_df.to_csv("new_test.csv", index=False)





    