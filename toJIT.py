
import torch

import lpips



loss_fn = lpips.LPIPS(net='alex', version='0.1')

scriptedm = torch.jit.script(loss_fn)
scriptedm.save('./lpips_alex_0.1.pt')

print('Done.')
