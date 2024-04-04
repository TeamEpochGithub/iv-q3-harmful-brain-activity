"""Module for smoothing the predictions based on the patient_id"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm

from src.modules.training.verbose_training_block import VerboseTrainingBlock

@dataclass
class SmoothPatient(VerboseTrainingBlock):
    """An example training block."""

    smooth_factor: float = 0 # If this factor is 0, the predictions will not be smoothed. 1 means the predictions will be fully smoothed based on the average of the patient's predictions.

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[Any, Any]:
        """Apply smoothing to the predictions.

        :param x: The input data
        :param y: The target data
        :return: The predictions and the target data
        """
        test_indices = train_args["test_indices"]
        metadata = train_args["metadata"]

        #Slice the metadata first to get the correct rows
        metadata = metadata.iloc[test_indices]

        #Now we make a dict that maps a index in the x (which is already of size test_indices) to the patient_id
        patient_ids = metadata["patient_id"].values

        #Calculate the average prediction for each patient
        patient_predictions = {}
        for i, patient_id in enumerate(patient_ids):
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []
            patient_predictions[patient_id].append(x[i])

        #Now we calculate the average prediction for each patient
        for patient_id in patient_predictions.keys():
            curr_avg = np.mean(np.array(patient_predictions[patient_id]), axis=0)
            patient_predictions[patient_id] = curr_avg

        #Based on the smooth factor, we multiply the current prediction with the average prediction of the patient
        for i, output_probs in tqdm(enumerate(x), desc="Smoothing predictions"):
            patient_id = patient_ids[i]
            x[i] = (1 - self.smooth_factor) * output_probs + self.smooth_factor * patient_predictions[patient_id]

        return x, y

    def custom_predict(self, x: npt.NDArray[np.float32], **pred_args: Any) -> npt.NDArray[np.float32]:
        """Apply the

        :param x: The predictions.
        :return: The softmaxed predictions.
        """

        metadata = pred_args["metadata"]

        #Now we make a dict that maps a index in the x (which is already of size test_indices) to the patient_id
        patient_ids = metadata["patient_id"].values

        #Calculate the average prediction for each patient
        patient_predictions = {}
        for i, patient_id in enumerate(patient_ids):
            if patient_id not in patient_predictions:
                patient_predictions[patient_id] = []
            patient_predictions[patient_id].append(x[i])

        #Now we calculate the average prediction for each patient
        for patient_id in patient_predictions.keys():
            if len(patient_predictions[patient_id]) == 1:
                continue
            curr_avg = np.mean(np.array(patient_predictions[patient_id]), axis=0)
            patient_predictions[patient_id] = curr_avg

        #Based on the smooth factor, we multiply the current prediction with the average prediction of the patient
        for i, output_probs in tqdm(enumerate(x), desc="Smoothing predictions"):
            patient_id = patient_ids[i]
            x[i] = (1 - self.smooth_factor) * output_probs + self.smooth_factor * patient_predictions[patient_id]

        return x
