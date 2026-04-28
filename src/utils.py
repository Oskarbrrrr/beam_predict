import torch
import numpy as np

def calculate_topk_accuracy(output, target, topk=(1, 3)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res

def calculate_dba_score(output, target, K=3, delta=5):
    with torch.no_grad():
        N = target.size(0)
        _, pred = output.topk(K, 1, True, True)
        target_expanded = target.view(N, 1)
        diff = torch.abs(pred - target_expanded).float() / delta
        
        eta_k_list = []
        for k in range(1, K + 1):
            min_diff_k = torch.min(diff[:, :k], dim=1)[0]
            clamped_diff = torch.clamp(min_diff_k, max=1.0)
            eta_k = 1.0 - (torch.sum(clamped_diff) / N)
            eta_k_list.append(eta_k.item())
        return sum(eta_k_list) / K

def calculate_apl(output, power_vectors):
    with torch.no_grad():
        _, pred_indices = output.topk(1, 1, True, True)
        loss_list = []
        for i in range(len(pred_indices)):
            pred_idx = pred_indices[i].item()
            p_vec = power_vectors[i].cpu().numpy()
            
            p_opt = np.max(p_vec)
            p_pred = p_vec[pred_idx]
            
            p_pred = max(p_pred, 1e-12)
            p_opt = max(p_opt, 1e-12)
            
            power_loss = 10 * np.log10(p_opt / p_pred)
            loss_list.append(power_loss)
            
        return np.sum(loss_list)