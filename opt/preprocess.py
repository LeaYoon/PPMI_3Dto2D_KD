import numpy as np
import cv2

def filter_X_with_label_index(X_data, y_label, label_index_list):
  res_X_data = []
  res_y_label = []

  remain_X_data=[]
  remain_y_label = []
  for i in range(len(X_data)):
    if y_label[i] in label_index_list:
      res_X_data.append(X_data[i])
      res_y_label.append(y_label[i])
    else:
      remain_X_data.append(X_data[i])
      remain_y_label.append(y_label[i])
  return np.array(res_X_data), np.array(res_y_label), np.array(remain_X_data), np.array(remain_y_label)



# entropy based 2D weighted iamge
def get_img_entropy(input):
  """
  Parameters
  ----------
  input: ndarray 
  numpy matrix of 3D image with shape (D, H, W, C) whose type is np.uint8
  For each plane (:, H, W, C) of input, the pmf for the pixel distribution is measured.

  Returns
  -------
  result: list
    measured entropy list according to each plane (D, ) 
  """
  input = input.astype(np.uint8)
  plane_entropy_list = []
  for plane in input:
    pix_vals = np.unique(plane)
    pix_probs = [len(plane[plane==pix_val])/len(plane.flatten()) for pix_val in pix_vals]
    
    res_entropy = 0
    for probs in pix_probs:
      res_entropy+= -probs*np.log(probs)
    plane_entropy_list.append(res_entropy)
  
  return plane_entropy_list

def get_entropy_based_weighted_image(input):
  """
  Parameters
  ----------  
  input: ndarray, (D, H, W) or (D, H, W, 1)

  Returns
  -------  
  """
  e=10**-5
  if input.shape[-1] != 1 and len(input.shape)==3:
    input = input[:,:,:,None]
  input_shape = input.shape
  entropy_list = np.array(get_img_entropy(input))
  # print("entropy_list which is nan", entropy_list[np.isnan(entropy_list)])
  # print()
  entropies = np.array(entropy_list).reshape([input_shape[0], 1, 1, 1])
  # _max = entropies.max()
  # _min = entropies.min()
  # norm_entropies = (entropies-_min)/(_max-_min)
  # weighted_input = input*norm_entropies
  weighted_input = input*entropies
  res = np.sum(weighted_input, axis=0)/(np.sum(entropies)+e)
  # if np.any(np.isnan(np.sum(entropies))):
  # # if np.sum(entropies)==0:
  #     print("_plane, ind", ind)
  #     print(np.sum(entropies)[np.isnan(np.sum(entropies))])

  return res



# 3D Image -> 2D dynamic Image
# 논문 깃허브 참고: https://github.com/mvrl/alzheimer-project/blob/master/Dynamic%2BAttention%20for%20AD%20MRI%20classification/scripts/ranksvm.py

def get_dynamic_image(frames, normalized=True):
    """ Adapted from https://github.com/tcvrick/Python-Dynamic-Images-for-Action-Recognition"""
    """ Takes a list of frames and returns either a raw or normalized dynamic image."""
    
    def _get_channel_frames(iter_frames, num_channels):
        """ Takes a list of frames and returns a list of frame lists split by channel. """
        frames = [[] for channel in range(num_channels)]

        for frame in iter_frames:
            for channel_frames, channel in zip(frames, cv2.split(frame)):
                channel_frames.append(channel.reshape((*channel.shape[0:2], 1)))
        for i in range(len(frames)):
            frames[i] = np.array(frames[i])
        return frames


    def _compute_dynamic_image(frames):
        """ Adapted from https://github.com/hbilen/dynamic-image-nets """
        num_frames, h, w, depth = frames.shape

        # Compute the coefficients for the frames.
        coefficients = np.zeros(num_frames)
        for n in range(num_frames):
            cumulative_indices = np.array(range(n, num_frames)) + 1
            coefficients[n] = np.sum(((2*cumulative_indices) - num_frames) / cumulative_indices)

        # Multiply by the frames by the coefficients and sum the result.
        x1 = np.expand_dims(frames, axis=0)
        x2 = np.reshape(coefficients, (num_frames, 1, 1, 1))
        result = x1 * x2
        return np.sum(result[0], axis=0).squeeze()

    num_channels = frames[0].shape[2]
    #print(num_channels)
    channel_frames = _get_channel_frames(frames, num_channels)
    channel_dynamic_images = [_compute_dynamic_image(channel) for channel in channel_frames]

    dynamic_image = cv2.merge(tuple(channel_dynamic_images))
    if normalized:
        dynamic_image = cv2.normalize(dynamic_image, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        dynamic_image = dynamic_image.astype('uint8')

    return dynamic_image

def to_categorical(labels, num_classes):
   _onehots =np.eye(num_classes)
   return _onehots[labels]