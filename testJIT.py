import torch
import argparse
import lpips

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

# load model & weights by JIT
loss_fn = torch.jit.load('./lpips_alex_0.1.pt')

if(opt.use_gpu):
	loss_fn.cuda()

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

if(opt.use_gpu):
	img0 = img0.cuda()
	img1 = img1.cuda()

# Compute distance
with torch.no_grad():
	dist01 = loss_fn.forward(img0, img1)

print('dist01:', dist01.shape)
print('Distance: %.3f'%dist01)
