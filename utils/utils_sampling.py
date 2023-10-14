import torch
import tqdm
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, b, eta=0., seq_next=None):
    with torch.no_grad():
        n = x.size(0)
        if seq_next == None:
            seq_next = [-1] + list(seq[:-1])

        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            #print(torch.ones(n).device,i.device)
            t = (torch.ones(n).to(x.device) * i).to(x.device)
            next_t = (torch.ones(n).to(x.device) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)

            et = model(xt, t)

            if et.size(1) == 6:
                et = et[:, :3]

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()

            z=torch.randn_like(x0_t)
            xt_next = at_next.sqrt() * x0_t + c1 * z + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds