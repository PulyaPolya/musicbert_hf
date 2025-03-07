# from musicbert_hf.checkpoints import load_musicbert_from_fairseq_checkpoint
#checkpoint_path = "C:/Polina/work/musicbert_hf/musicbert_checkpoint/checkpoint_last_musicbert_base.pt"

# model = load_musicbert_from_fairseq_checkpoint("C:/Polina/work/musicbert_hf/musicbert_checkpoint/checkpoint_last_musicbert_base.pt")

# from musicbert_hf.checkpoints import (
#     load_musicbert_token_classifier_from_fairseq_checkpoint,
# )

# model = load_musicbert_token_classifier_from_fairseq_checkpoint(
#     checkpoint_path,
#     checkpoint_type="token_classifier",
#    #num_labels=42,  # Set number of output labels according to your task
# #)
# from musicbert_hf import MusicBertTokenClassification

# #MusicBertTokenClassification.from_pretrained("path/to/checkpoint/directory")
# from musicbert_hf.checkpoints import (
#     load_musicbert_token_classifier_from_fairseq_checkpoint,
# )

# model = load_musicbert_token_classifier_from_fairseq_checkpoint(
#     "checkpoints/rn_conditioned_checkpoint_40356401.pt",
#     checkpoint_type="token_classifier",
#     #num_labels=42,  # Set number of output labels according to your task
# )
from musicbert_hf import MusicBertTokenClassification


MusicBertTokenClassification.from_pretrained("checkpoints/rn_conditioned_checkpoint_40356401.pt")