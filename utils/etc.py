import torch

def time_check(total_time):
    H = 60*60
    M = 60
    if total_time >= H:
        return f"{int(total_time//H)}h {(int(total_time-H)//M)}m {total_time%M:.2f}s"
    elif total_time >= M:
        return f"{(int(total_time)//M)}m {total_time%M:.2f}s"
    else:
        return f"{total_time%M:.2f}s"


def print_model_params(model, total_params=True, trainable_params=True, non_trainable_params=True):
    """
    Print the number of parameters in the model.
    """
    if total_params:
        print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    if trainable_params:
        # Print the number of trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {trainable_params:,}")
    if non_trainable_params:
        # Print the number of non-trainable parameters
        non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"Total number of non-trainable parameters: {non_trainable_params:,}")


class AUCMeter:
    def __init__(self):
        self.reset()

    def update(self, preds, target):
        if preds.shape[1] == 2:
            probs = torch.softmax(preds, dim=1)[:, 1]
        else:
            probs = preds.squeeze()

        # Fix shape: (N, 1) -> (N,)
        probs = probs.detach().cpu().flatten()
        target = target.detach().cpu().flatten()

        self.preds.append(probs)
        self.targets.append(target)


    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)

        # Sort scores and corresponding true labels
        desc_score_indices = torch.argsort(preds, descending=True)
        targets = targets[desc_score_indices]

        # Compute true positive and false positive rates
        tps = torch.cumsum(targets, dim=0)
        fps = torch.cumsum(1 - targets, dim=0)

        # Append (0,0) at the beginning
        tps = torch.cat([torch.tensor([0], device=tps.device), tps])
        fps = torch.cat([torch.tensor([0], device=fps.device), fps])

        # Normalize
        if tps[-1] == 0 or fps[-1] == 0:
            # 예외 처리: positive 또는 negative 샘플이 하나도 없는 경우
            return 0.0

        tpr = tps / tps[-1]
        fpr = fps / fps[-1]

        # Compute AUC using trapezoidal rule
        auc = torch.trapz(tpr, fpr)

        return auc.item()

    def reset(self):
        self.preds = []
        self.targets = []
