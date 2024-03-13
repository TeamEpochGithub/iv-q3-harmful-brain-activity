from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

data = pd.read_csv('data/raw/train.csv')

def create_stratified_cv_splits(X, y, n_splits=5):
    """
    Create stratified cross-validation splits ensuring:
    - Each fold has proportional representation of the predominant labels.
    - No eeg_id appears in both training and validation sets of a fold.
    
    Parameters:
    - data: The original dataset.
    - eeg_id_to_label: DataFrame mapping eeg_id to its predominant label.
    - n_splits: Number of folds for the cross-validation.
    
    Returns:
    - A list of tuples, each containing the indices for training and validation sets for each fold.
    """

    # create data from X and y
    data = pd.DataFrame({'eeg_id': X.meta['eeg_id'], 'expert_consensus': y.argmax(axis=1)})

    # Group by `eeg_id` and `expert_consensus` to examine the distribution of labels within these groups
    label_distribution_per_eeg_id = data.groupby(['eeg_id', 'expert_consensus']).size().unstack(fill_value=0)

    # Determine the predominant label for each eeg_id
    predominant_labels = label_distribution_per_eeg_id.idxmax(axis=1)

    # Prepare data for stratification: map each eeg_id to its predominant label
    eeg_id_to_predominant_label = predominant_labels.to_frame(name='predominant_label').reset_index()

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Placeholder for the splits
    splits = []
    
    # Generate splits based on the eeg_id's predominant label
    for train_index, test_index in skf.split(eeg_id_to_predominant_label, eeg_id_to_predominant_label['predominant_label']):
        # Get the eeg_ids for the current split
        train_eeg_ids = eeg_id_to_predominant_label.iloc[train_index]['eeg_id']
        test_eeg_ids = eeg_id_to_predominant_label.iloc[test_index]['eeg_id']
        
        # Determine the indices in the original dataset corresponding to these eeg_ids
        train_indices = data[data['eeg_id'].isin(train_eeg_ids)].index
        test_indices = data[data['eeg_id'].isin(test_eeg_ids)].index
        
        # Append the indices to the splits list
        splits.append((train_indices, test_indices))
    
    return splits

