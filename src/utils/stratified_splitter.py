from sklearn.model_selection import StratifiedKFold
import numpy as np
import pandas as pd


def create_stratified_cv_splits(X: pd.DataFrame, y: np.ndarray, n_splits=5):
    """
    Create stratified cross-validation splits ensuring:
    - Each fold has proportional representation of the predominant labels.
    - No patient_id appears in both training and validation sets of a fold.
    
    Parameters:
    - data: The original dataset.
    - patient_id_to_label: DataFrame mapping patient_id to its predominant label.
    - n_splits: Number of folds for the cross-validation.
    
    Returns:
    - A list of tuples, each containing the indices for training and validation sets for each fold.
    """

    # create data from X and y
    data = pd.DataFrame({'patient_id': X['patient_id'], 'expert_consensus': y.argmax(axis=1)})

    # Group by `patient_id` and `expert_consensus` to examine the distribution of labels within these groups
    label_distribution_per_patient_id = data.groupby(['patient_id', 'expert_consensus']).size().unstack(fill_value=0)

    # Determine the predominant label for each patient_id
    predominant_labels = label_distribution_per_patient_id.idxmax(axis=1)

    # Prepare data for stratification: map each patient_id to its predominant label
    patient_id_to_predominant_label = predominant_labels.to_frame(name='predominant_label').reset_index()

    # Initialize the StratifiedKFold object
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Placeholder for the splits
    splits = []
    
    # Generate splits based on the patient_id's predominant label
    for train_index, test_index in skf.split(patient_id_to_predominant_label, patient_id_to_predominant_label['predominant_label']):
        # Get the patient_ids for the current split
        train_patient_ids = patient_id_to_predominant_label.iloc[train_index]['patient_id']
        test_patient_ids = patient_id_to_predominant_label.iloc[test_index]['patient_id']
        
        # Determine the indices in the original dataset corresponding to these patient_ids
        train_indices = data[data['patient_id'].isin(train_patient_ids)].index
        test_indices = data[data['patient_id'].isin(test_patient_ids)].index
        
        # Append the indices to the splits list
        splits.append((train_indices, test_indices))
    
    return splits

