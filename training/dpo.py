import torch
import torch.nn.functional as F


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta * logits).mean()
    return loss


def train_dpo(policy_model, ref_model, optimizer, dataloader, beta=0.1):
    policy_model.train()
    ref_model.eval()
    for step, batch in enumerate(dataloader):
        # compute logps for chosen/rejected under policy and frozen ref model,
        # then call dpo_loss(...). Wire up actual logp computation before use.
        pass


if __name__ == "__main__":
    pass
