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
from musicbert_hf.checkpoints import (
    load_musicbert_multitask_token_classifier_from_fairseq_checkpoint,
    load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint,
    load_musicbert_token_classifier_from_fairseq_checkpoint,
)

#MusicBertTokenClassification.from_pretrained("checkpoints/rn_conditioned_checkpoint_40356401.pt")
checkpoint_path_conditioned = "checkpoints/rn_conditioned_checkpoint_40356401.pt"
checkpoint_path_key = "checkpoints/key_checkpoint_39958320.pt"
checkpoint_path_chained = "checkpoints/rn_chained_checkpoint_45951812.pt"
checkpoint_type="musicbert"
num_labels=[29, 7, 1236]
z_vocab_size=29
model = load_musicbert_multitask_token_classifier_with_conditioning_from_fairseq_checkpoint(
                    checkpoint_path_conditioned,
                    checkpoint_type="token_classifier",
                    # num_labels = num_labels,
                    # z_vocab_size = z_vocab_size
                
                )
# model = load_musicbert_multitask_token_classifier_from_fairseq_checkpoint(
#                         checkpoint_path_chained,
#                         checkpoint_type="musicbert",
#                         num_labels=[29]
#                     )
