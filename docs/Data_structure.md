# Data Structure

Place your dataset under a common root folder with the following layout:

'''
IMU/
DLC/
sup_IMU/
sup_DLC/
sup_labels/
'''
All files are numpy arrays (`.npy`). Unsupervised recordings are stored as `samples_<session>.npy` inside `IMU/` and `DLC/`. Supervised segments and their labels follow `sup_IMU_<session>.npy`, `sup_DLC_<session>.npy` and `sup_labels_<session>.npy`.

The list of session names used for training and testing is defined in `train.py`.
Each session's arrays must have the same length for IMU and DLC so they can be aligned.