from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd

data = pd.read_csv('data/raw/train.csv')

# Group by `eeg_id` and `expert_consensus` to examine the distribution of labels within these groups
label_distribution_per_eeg_id = data.groupby(['eeg_id', 'expert_consensus']).size().unstack(fill_value=0)

# Check the distribution
label_distribution_per_eeg_id.head(), label_distribution_per_eeg_id.describe()
# Determine the predominant label for each eeg_id
predominant_labels = label_distribution_per_eeg_id.idxmax(axis=1)

# Count the occurrences of each predominant label
predominant_label_counts = predominant_labels.value_counts()

# Prepare data for stratification: map each eeg_id to its predominant label
eeg_id_to_predominant_label = predominant_labels.to_frame(name='predominant_label').reset_index()

# Check the mapping and the counts of predominant labels
eeg_id_to_predominant_label.head(), predominant_label_counts


def create_stratified_cv_splits(data, eeg_id_to_label, n_splits=5):
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
    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Placeholder for the splits
    splits = []
    
    # Generate splits based on the eeg_id's predominant label
    for train_index, test_index in skf.split(eeg_id_to_label, eeg_id_to_label['predominant_label']):
        # Get the eeg_ids for the current split
        train_eeg_ids = eeg_id_to_label.iloc[train_index]['eeg_id']
        test_eeg_ids = eeg_id_to_label.iloc[test_index]['eeg_id']
        
        # Determine the indices in the original dataset corresponding to these eeg_ids
        train_indices = data[data['eeg_id'].isin(train_eeg_ids)].index
        test_indices = data[data['eeg_id'].isin(test_eeg_ids)].index
        
        # Append the indices to the splits list
        splits.append((train_indices, test_indices))
    
    return splits

# Example usage with 5 folds
n_splits_example = 5
cv_splits = create_stratified_cv_splits(data, eeg_id_to_predominant_label, n_splits=n_splits_example)

# Since it's not feasible to display all splits due to their size, let's show the size of the first split

# print the label distribution of the train and validation parts of the first split

train_indices, val_indices = cv_splits[0]
train_label_distribution = data.loc[train_indices, 'expert_consensus'].value_counts(normalize=True)
val_label_distribution = data.loc[val_indices, 'expert_consensus'].value_counts(normalize=True)

print(f"Train label distribution (split 1):\n{train_label_distribution}\n")
print(f"Validation label distribution (split 1):\n{val_label_distribution}")

# verify that there are no overlapping eeg ids between train and validation splitsfor all the folds
overlapping_eeg_ids = []
for i, (train_indices, val_indices) in enumerate(cv_splits):
    train_eeg_ids = data.loc[train_indices, 'eeg_id']
    val_eeg_ids = data.loc[val_indices, 'eeg_id']
    overlapping_eeg_ids.append(train_eeg_ids.isin(val_eeg_ids).sum())

print(overlapping_eeg_ids)
