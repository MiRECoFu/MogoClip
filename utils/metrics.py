import torch

def recall_at_k(logits, k):
    # 获取每个 motion 的 top-k 相似的 text 索引
    indices = torch.topk(logits, k=k, dim=-1).indices  # (batch_size, k)
    
    # 目标索引：对角线上的正确匹配
    targets = torch.arange(logits.shape[0], device=logits.device).view(-1, 1)  # (batch_size, 1)
    
    # 检查目标索引是否在 top-k 中
    correct = (indices == targets).sum().item()
    
    # Recall = 正确预测数 / 样本总数
    return correct / logits.shape[0]


def mean_reciprocal_rank(logits):
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    targets = torch.arange(logits.shape[0], device=logits.device)
    ranks = (sorted_indices == targets.view(-1, 1)).nonzero(as_tuple=True)[1] + 1
    return (1 / ranks.float()).mean().item()
