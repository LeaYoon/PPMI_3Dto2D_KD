import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from scipy import stats

# function to plot images given grid and list of images
def plot_images(images, grid, figsize=(15, 5), savepath=None):
  len_list = len(images)
  if grid is None:
    grid = (1, len_list)
  
  total_cell = grid[0]*grid[1]
  if len_list > total_cell:
    print(f"Number of images is {len_list} and total cell is {total_cell}.")
    print("Number of images is more than ottal cell. Please check again.")
    return
  fig, ax = plt.subplots(grid[0], grid[1], figsize=figsize)
  count=0
  for i in range(grid[0]):
    for j in range(grid[1]):
      if count < len_list:
        if grid[0]==1:
          ax[j].imshow(images[j])
          ax[j].axis('off')
        else:
          ax[i, j].imshow(images[i*grid[1]+j])
          ax[i, j].axis('off')
        count+=1
      else:
        if savepath is not None:
          plt.savefig(savepath)
        else:
          plt.show()
        break
  if savepath is not None:
    plt.savefig(savepath)
  else:
    plt.show()

def plot_trend(history, save_path = None):

    # for item in trend_dict.items(): # key value pair i.g. "train_acc": 90.0
    #     plt.plot(item[1], label=item[0])
    # plt.legend('upper right')
    # plt.savefig(save_path)
    fig, loss_ax = plt.subplots()
    loss_ax.plot(history.history['loss'], 'y', label='train loss')
    loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    
    try:
        acc_ax = loss_ax.twinx()
        acc_ax.plot(history.history['acc'], 'b', label='train acc')      
        acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
        acc_ax.set_ylabel('accuray')
    except KeyError as ke:
        pass 
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
        plt.cla()
        plt.clf()
    return

# Evaluation modules

def get_performances(target, output, pos_label=1):
    # Evaluation 1 : Accuracy
    _acc = accuracy_score(target, output)

    # Evaluation 2 : F1-score
    _macro_f1_score = f1_score(target, output, average="macro")
    _micro_f1_score = f1_score(target, output, average="micro")
    _weighted_f1_score = f1_score(target, output, average="weighted")

    # Evaluation 3 : geometric mean
    conf_mat = confusion_matrix(target, output)
    specificity = conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1])
    sensitivity = conf_mat[1][1] / (conf_mat[1][0] + conf_mat[1][1])
    _g_mean = stats.gmean([specificity, sensitivity])

    return _acc, _macro_f1_score, _micro_f1_score, _weighted_f1_score, _g_mean

def save_pred_label(label, pred, save_filepath, onehot_label=True, filename_list = None):
    """
    :param label:
    :param pred:
    :param save_filepath:
    :param onehot_label:
    :param filename_list: id indicating for each instance infered from trained model_name, which is helpful to trace data for debugging
    :return: None, this function just store excel data arranging table with label and prediction of trained model_name
    """
    filename_list = np.array(filename_list).tolist()
    if not onehot_label : # when onehot_label is False, the shape is (batch_size, ), i.e, expressing the data with 1 column
        num_classes = len(np.unique(label))
        log_dict = dict()
        log_dict["label"] = np.array(label).tolist()
        log_dict["pred"] = np.array(pred).tolist()
        categorical_label = to_categorical(label, num_classes)
        for ind in range(num_classes):
            log_dict["pred_"+str(ind)] = np.array(categorical_label)[:,ind].tolist()
        if filename_list:
            log_dict["pid"] = filename_list
    else: # when onehot_label is True, the shape of pred AND label is (batch_size, num_classes), i.e expressing the data with many column
        num_classes = len(label[0])

        pred_ = np.array(pred).argmax(axis=1)
        label_ = np.array(label).argmax(axis=1)
        print("pred_", pred_.shape)

        log_dict = dict()
        log_dict["label"] = np.array(label_).tolist()
        log_dict["pred"] = np.array(pred_).tolist()
        for ind in range(num_classes):
            log_dict["pred_"+str(ind)] = np.array(pred)[:,ind].tolist()

        if filename_list:
            log_dict["pid"] = filename_list


    print("label", len(log_dict["label"]))
    print("pred", len(log_dict["pred"]))
    print("pid", len(log_dict["pid"]))

    data = pd.DataFrame(log_dict)
    data.to_excel(save_filepath)
