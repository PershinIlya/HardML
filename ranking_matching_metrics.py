from math import log2

from torch import Tensor, sort


def num_swapped_pairs(ys_true: Tensor, ys_pred: Tensor) -> int:
    num_swapped = 0
    for i in range(len(ys_pred)):
        for j in range(i + 1, len(ys_pred)):
            if ys_true[i] < ys_true[j] and ys_pred[i] > ys_pred[j]:
                num_swapped += 1
            elif ys_true[i] > ys_true[j] and ys_pred[i] < ys_pred[j]:
                num_swapped += 1

    return num_swapped


def compute_gain(y_value: float, gain_scheme: str) -> float:
    if gain_scheme == 'const':
        return float(y_value)
    elif gain_scheme == 'exp2':
        return float(2. ** y_value - 1)


def dcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str) -> float:
    sort_idxs = ys_pred.sort(descending=True).indices
    ys_true_sort = ys_true[sort_idxs]
    dcg_val = 0
    for i in range(len(ys_pred)):
        gain_val = compute_gain(ys_true_sort[i].float(), gain_scheme=gain_scheme)
        dcg_val += gain_val / log2(i + 2)

    return dcg_val
# print(dcg(torch.tensor([3,2,1,1,3,1,2]), torch.tensor(list(range(1, 8))[::-1]), gain_scheme='const'))
# dcg(torch.tensor([3,2,1,1,3,1,2]), torch.tensor(list(range(1, 8))[::-1]), gain_scheme='exp2')


def ndcg(ys_true: Tensor, ys_pred: Tensor, gain_scheme: str = 'const') -> float:
    dcg_val = dcg(ys_true, ys_pred, gain_scheme)
    ys_true_sort, sort_idxs = ys_true.sort(descending=True)
    ys_pred_sort = ys_true[sort_idxs]
    idcg_val = dcg(ys_true_sort, ys_pred_sort, gain_scheme)

    return dcg_val / idcg_val
# print(ndcg(torch.tensor([3,2,1,1,3,1,2]), torch.tensor(list(range(1, 8))[::-1]), gain_scheme='const'))


def precission_at_k(ys_true: Tensor, ys_pred: Tensor, k: int) -> float:
    sort_idxs = ys_pred.sort(descending=True).indices
    ys_true_sort = ys_true[sort_idxs]
    if any(ys_true):
        n = sum(ys_true)
        div = k
        if k > n:
            div = n
        return float(ys_true_sort[:int(k)].sum()) / float(div)
    else:
        return -1
# precission_at_k(torch.tensor([1,0,1,1,0,1,0,0]), torch.tensor([0.9,0.85,0.71,0.63,0.47,0.36,0.24,0.16]), 4)
# precission_at_k(torch.tensor([1,1,1,0,0,0]), torch.tensor([0.9,0.85,0.71,0.63,0.47,0.36]), 4)


def reciprocal_rank(ys_true: Tensor, ys_pred: Tensor) -> float:
    sort_idxs = ys_pred.sort(descending=True).indices
    ys_true_sort = ys_true[sort_idxs]
    return 1. / (float(ys_true_sort.argmax()) + 1)


def p_found(ys_true: Tensor, ys_pred: Tensor, p_break: float = 0.15) -> float:
    sort_idxs = ys_pred.sort(descending=True).indices
    ys_true_sort = ys_true[sort_idxs]
    p_look = 1
    p_found_val = p_look * ys_true_sort[0]
    for i in range(1, len(ys_true_sort)):
        p_look = p_look * (1 - ys_true_sort[i - 1]) * (1 - p_break)
        p_found_val += p_look * ys_true_sort[i]

    return p_found_val


def average_precision(ys_true: Tensor, ys_pred: Tensor) -> float:
    if not any(ys_true):
        return -1
    sort_idxs = ys_pred.sort(descending=True).indices
    ys_true_sort = ys_true[sort_idxs]
    ap_val = 0.
    n_correct = 0.
    for i in range(len(ys_true_sort)):
        if ys_true_sort[i].bool():
            n_correct += 1
            ap_val += n_correct / (i + 1)

    return ap_val / n_correct
