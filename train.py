import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

# 从 src 导入你整理好的模块
from src.dataset import MultimodalDataset
from src.model import BeMambaModel
from src.utils import calculate_topk_accuracy, calculate_dba_score, calculate_apl

# 针对波束不平衡的 FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = -((1 - pt) ** self.gamma) * log_pt
        return loss.mean()

def run_scenario(scenario_name):
    print(f"\n========== 开始训练场景: {scenario_name} ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径映射（确保 data_split.py 已将 CSV 生成在 Data/splits/ 下）
    train_ds = MultimodalDataset(mode='train', scenario_name=scenario_name)
    val_ds   = MultimodalDataset(mode='val', scenario_name=scenario_name)
    test_ds  = MultimodalDataset(mode='test', scenario_name=scenario_name)
    
    # Batch size 设为 16 或 32，视你的 3090 显存而定
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    model = BeMambaModel().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-2)

    best_dba = 0.0
    os.makedirs("checkpoints", exist_ok=True)
    
    epochs = 20 # 先跑 20 轮看看情况
    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        for imgs, radars, lidars, gps, targets, _ in train_loader:
            imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, radars, lidars, gps)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * targets.size(0)
            
        # --- 验证阶段 ---
        model.eval()
        t1_tot, t3_tot, dba_tot, apl_tot = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, radars, lidars, gps, targets, power_vec in val_loader:
                imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
                
                outputs = model(imgs, radars, lidars, gps)
                
                # 计算指标
                acc1, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
                bs = targets.size(0)
                t1_tot += acc1 * bs
                t3_tot += acc3 * bs
                dba_tot += calculate_dba_score(outputs, targets) * bs
                apl_tot += calculate_apl(outputs, power_vec)
                
        n_val = len(val_ds)
        epoch_dba = dba_tot / n_val
        print(f"Epoch {epoch+1:02d} | L: {train_loss/len(train_ds):.4f} | "
              f"Acc@3: {t3_tot/n_val:.2f}% | DBA: {epoch_dba:.4f} | APL: {apl_tot/n_val:.4f} dB")
        
        if epoch_dba > best_dba:
            best_dba = epoch_dba
            torch.save(model.state_dict(), f"checkpoints/best_{scenario_name}.pth")

    # --- 最终 Test 评估阶段 ---
    print(f"\n>>> 载入 {scenario_name} 的最佳模型进行最终 Test 评估...")
    model.load_state_dict(torch.load(f"checkpoints/best_{scenario_name}.pth"))
    model.eval()
    
    t1_tot, t3_tot, dba_tot, apl_tot = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for imgs, radars, lidars, gps, targets, power_vec in test_loader:
            imgs, radars, lidars, gps, targets = imgs.to(device), radars.to(device), lidars.to(device), gps.to(device), targets.to(device)
            outputs = model(imgs, radars, lidars, gps)
            
            acc1, acc3 = calculate_topk_accuracy(outputs, targets, topk=(1, 3))
            bs = targets.size(0)
            t1_tot += acc1 * bs
            t3_tot += acc3 * bs
            dba_tot += calculate_dba_score(outputs, targets) * bs
            apl_tot += calculate_apl(outputs, power_vec)
            
    n_test = len(test_ds)
    print(f"\n[Final Results for {scenario_name} Test Set]")
    print(f"Top-1 Acc: {t1_tot/n_test:.2f}%")
    print(f"Top-3 Acc: {t3_tot/n_test:.2f}%")
    print(f"DBA Score: {dba_tot/n_test:.4f}")
    print(f"APL Loss:  {apl_tot/n_test:.4f} dB\n")

if __name__ == "__main__":
    # 调用前确保你已经执行了: python src/data_split.py
    run_scenario("scenario32")
    # run_scenario("scenario33")
    # run_scenario("scenario34")