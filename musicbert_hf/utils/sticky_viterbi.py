# import numpy as np
import torch
from tqdm import tqdm


def sticky_viterbi(P: torch.Tensor, alpha: float, pbar: bool = True):
    """
    A version of the Viterbi algorithm that discourages switching states.

    In order to encourage self-transitions, we set the transition probabilities to be
    uniform across all states, except for self-transitions, which are scaled by `alpha`
    relative to all other transitions. For example, if we have three states and alpha =
    2.0, the transition probability matrix would be:

    \\begin{bmatrix}
    0.5 & 0.25 & 0.25 \\\\
    0.25 & 0.5 & 0.25 \\\\
    0.25 & 0.25 & 0.5
    \\end{bmatrix}
    
    Args:
        P: an tensor of probabilities, shape [sequence length, state probabilities].
            For example, the probability of a key at each time step.
        alpha: a parameter that controls how self-transitions are weighted.

    Returns:
        A tensor of shape [sequence length] containing the most likely state sequence.

    If alpha = 1.0, then the output is equivalent to argmax:
    >>> P = torch.tensor([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]])
    >>> sticky_viterbi(P, alpha=1.0)
    tensor([0, 1, 0])
    >>> P = torch.tensor([[0.9, 0.1], [0.49, 0.51], [0.51, 0.49], [0.49, 0.51]])
    >>> sticky_viterbi(P, alpha=1.0)
    tensor([0, 1, 0, 1])

    If alpha > 1.0, then switching is disfavored:
    >>> P = torch.tensor([[0.1, 0.9], [0.51, 0.49], [0.51, 0.49], [0.51, 0.49]])
    >>> sticky_viterbi(P, alpha=1.2)
    tensor([1, 1, 1, 1])

    >>> P = torch.tensor([[0.51, 0.49], [0.51, 0.49], [0.51, 0.49], [0.1, 0.9]])
    >>> sticky_viterbi(P, alpha=1.1)
    tensor([0, 0, 0, 1])

    If alpha < 1.0, then switching is actually favored:
    >>> P = torch.tensor([[0.9, 0.1], [0.51, 0.49], [0.51, 0.49], [0.51, 0.49]])
    >>> sticky_viterbi(P, alpha=0.9)
    tensor([0, 1, 0, 1])

    """
    if alpha == 1.0:
        return torch.argmax(P, axis=-1)
    seq_len, n_states = P.shape
    transition_probs = torch.ones((n_states, n_states))
    transition_probs[range(n_states), range(n_states)] *= alpha
    transition_probs /= transition_probs.sum(axis=0, keepdims=True)

    # take logs for numerical stability
    transition_probs = torch.log(transition_probs)
    P = torch.log(P)

    log_prob_of_zero = -1e10
    scores = torch.full((seq_len, n_states), fill_value=log_prob_of_zero)
    scores[0, :] = P[0, :]

    traceback = torch.ones((seq_len - 1, n_states), dtype=torch.long) * -1

    if pbar:
        iterator = tqdm(range(1, seq_len), total=seq_len - 1, desc="Viterbi")
    else:
        iterator = range(1, seq_len)

    for seq_i in iterator:
        for this_state_i in range(n_states):
            for prev_state_i in range(n_states):
                new_score = (
                    scores[seq_i - 1, prev_state_i]
                    + transition_probs[prev_state_i, this_state_i]
                    + P[seq_i, this_state_i]
                )
                if new_score > scores[seq_i, this_state_i]:
                    scores[seq_i, this_state_i] = new_score
                    traceback[seq_i - 1, this_state_i] = prev_state_i

    assert not (traceback == -1).any()

    state_i = scores[-1, :].argmax()
    out = [state_i]
    for seq_i in range(seq_len - 2, -1, -1):
        state_i = traceback[seq_i, state_i]
        out.append(state_i)
    return torch.tensor(out[::-1])
