# Torch_audiomentations augmentations


- _target_: torch_audiomentations.AddColoredNoise
            p: 0.15
            mode: per_channel
            p_mode: per_channel
            max_snr_in_db: 15
            sample_rate: 200

- _target_: torch_audiomentations.ShuffleChannels
            p: 0.15

- _target_: torch_audiomentations.Shift
            p: 0.15
            rollover: true
            mode: per_example
