import numpy as np
import tensorflow as tf
from tensorflow import keras


def absmaxND(a, axis=None):
    amax = np.max(a, axis)
    amin = np.min(a, axis)
    return np.where(-amin > amax, amin, amax)


def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X, verbose=0)

def pearsonr(vector1, vector2):
    m1 = np.mean(vector1)
    m2 = np.mean(vector2)
    
    diff1 = vector1 - m1
    diff2 = vector2 - m2
    
    top = np.sum(diff1 * diff2)
    bottom = np.sum(np.power(diff1, 2)) * np.sum(np.power(diff2, 2))
    bottom = np.sqrt(bottom)
    
    return top/bottom


def correlation_matrix(model, c_index, mha_index, X, thresh=0.1, random_frac=0.5, limit=None, head_concat=np.max, symmetrize=absmaxND):
    
    """
    * model                  trained tensorflow model
    * c_index                index of the convolutoinal layer (after pooling)
    * mha_index              index of multi-head attention layer
    * X                      test sequences
    * thresh                 attention threshold
    * random_frac            proportion of negative positions in the set of position interactions
    * limit                  maximum number of position interactions processed; sometimes needed to avoid resource exhaustion
    * head_concat            function for concatenating heads; e.g. np.max, np.mean
    * symmetrize             function for symmetrizing the correlation matrix across diagonal
    """
    
    assert 0 <= random_frac < 1
    
    feature_maps = get_layer_output(model, c_index, X)
    o, att_maps = get_layer_output(model, mha_index, X)
    att_maps = head_concat(att_maps, axis=1)
    
    position_interactions = get_position_interactions(att_maps, thresh)
    num_rands = int(random_frac/(1-random_frac))
    random_interactions = [np.random.randint(len(att_maps), size=(num_rands, 1)), np.random.randint(att_maps.shape[1], size=(num_rands, 2))]
    position_pairs = [np.vstack([position_interactions[0], random_interactions[0]]), np.vstack([position_interactions[1], random_interactions[1]])]
    if limit is not None:
        permutation = np.random.permutation(len(position_pairs[0]))
        position_pairs = [position_pairs[0][permutation], position_pairs[1][permutation]]
        position_pairs = [position_pairs[0][:limit], position_pairs[1][:limit]]
    
    filter_interactions = feature_maps[position_pairs].transpose([1, 2, 0])
    correlation_matrix = correlation(filter_interactions[0], filter_interactions[1])
    if symmetrize is not None:
        correlation_matrix = symmetrize(np.array([correlation_matrix, correlation_matrix.transpose()]), axis=0)
    correlation_matrix = np.nan_to_num(correlation_matrix)
    
    return correlation_matrix

    
def get_position_interactions(att_maps, threshold=0.1):
    position_interactions = np.array(np.where(att_maps >= threshold))
    position_interactions = [position_interactions[[0]].transpose(), position_interactions[[1, 2]].transpose()]
    return position_interactions
    
    
def correlation(set1, set2, function=pearsonr):
    combinations = np.indices(dimensions=(set1.shape[0], set2.shape[0])).transpose().reshape((-1, 2)).transpose()[::-1]
    vector_mesh = [set1[combinations[0]], set2[combinations[1]]]
    vector_mesh = np.array(vector_mesh).transpose([1, 0, 2])
    correlations = []
    for i in range(len(vector_mesh)):
        r = function(vector_mesh[i][0], vector_mesh[i][1])
        correlations.append(r)
    correlations = np.array(correlations).reshape((len(set1), len(set2)))
    return correlations





def filter_max_align_batch(X, model, layer=3, window=24, threshold=0.5, batch_size=1024, max_align=1e4, verbose=1):
  """get alignment of filter activations for visualization"""
  if verbose:
    print("Calculating filter PPM based on activation-based alignments")

  N,L,A = X.shape
  num_filters = model.layers[layer].output.shape[2]

  # Set the left and right window sizes
  window_left = int(window/2)
  window_right = window - window_left

  # get feature maps of 1st convolutional layer after activation
  intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output)

  # batch the data
  dataset = tf.data.Dataset.from_tensor_slices(X)
  batches = dataset.batch(batch_size)
  
  # loop over batches to capture MAX activation
  if verbose:
    print('  Calculating MAX activation')
  MAX = np.zeros(num_filters)
  for x in batches:

    # get feature map for mini-batch
    fmap = intermediate.predict(x)

    # loop over each filter to find "active" positions
    for f in range(num_filters):
      MAX[f] = np.maximum(MAX[f], np.max(fmap[:,:,f]))


  # loop over each filter to find "active" positions
  W = []
  counts = []
  for f in range(num_filters):
    if verbose:
      print("    processing %d out of %d filters"%(f+1, num_filters))
    status = 0

    # compile sub-model to get feature map
    intermediate = keras.Model(inputs=model.inputs, outputs=model.layers[layer].output[:,:,f])

    # loop over each batch
    #dataset = tf.data.Dataset.from_tensor_slices(X)
    seq_align_sum = np.zeros((window, A)) # running sum
    counter = 0                            # counts the number of sequences in alignment
    status = 1                            # monitors whether depth of alignment has reached max_align
    for x in batches:
      if status:

        # get feature map for a batch sequences
        fmaps = intermediate.predict(x)

        # Find regions above threshold
        for data_index, fmap in enumerate(fmaps):
          if status:
            pos_index = np.where(fmap > MAX[f] * threshold)[0]

            # Make a sequence alignment centered about each activation (above threshold)
            for i in range(len(pos_index)):
              if status:
                # Determine position of window about each filter activation
                start_window = pos_index[i] - window_left
                end_window = pos_index[i] + window_right

                # Check to make sure positions are valid
                if (start_window > 0) & (end_window < L):
                  seq_align_sum += x[data_index,start_window:end_window,:].numpy()
                  counter += 1
                  if counter == max_align:
                    status = 0
                else:
                  break
          else:
            break
      else:
        if verbose:
          print("      alignment has reached max depth for all filters")
        break

    # calculate position probability matrix of filter
    if verbose:
      print("      %d sub-sequences above threshold"%(counter))
    if counter > 0:
      W.append(seq_align_sum/counter)
    else:
      W.append(np.ones((window,A))/A)
    counts.append(counter)
  return np.array(W), np.array(counts)
