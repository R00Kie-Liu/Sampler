# Task-adaptive Spatial-Temporal Video Sampler for Few-shot Action Recognition

Please refer to https://arxiv.org/abs/2207.09759 for this paper.

## Introduction
<div align="center">
<img src="imgs/sampler.png" width="60%" height="50%"/><br/>
</div>
A uniform sampler may overlook frames containing key actions. Critical regions involving the actors and objects may be too small to be properly recognized. (b) Our sampler can (I) select frames from an entire video that contribute most to few-shot recognition, (II) amplify discriminative regions in each frame. This sampling strategy is also dynamically adjusted for each video according to the episode task at hand.

## Overview

<div align="center">
<img src="imgs/overall.png" width="80%" height="80%"/><br/>
</div>


## Todo

- [x] Realease the ActivityNet dataset few-shot split file.
- [x] Realease the core part of Sampler.
- [ ] Code of Sampler + ProtoNet.
- [ ] Realease the whole training and inference code.
- [ ] Sampler + TA2N/TRX/OTAM.

## Usage
Example of spatial-temporal sampling from input query video set
```python
from sampler import Selector

args = ArgsObject() # Refer to sampler.py for details of args
S = Selector(args).cuda()
input = torch.rand(args.way*args.shot*args.seq_len, 3, args.img_size, args.img_size).cuda() 
# Input: way*shot*frame, c, w, h
n, c, w, h = input.size()
print('Input Data shape:', input.shape)

# Indice: way*shot, k, len
indices,_,_,_,_,_ = S(input)
input = input.view(args.way*args.shot, args.seq_len, -1) 
# Output: way*shot, k, c*w*h
subset = torch.bmm(indices, input)
# Output: way*shot*k, c, w, h
subset = subset.view(-1, c, w, h)
print('Data shape output by sampler:', subset.shape)
```

## Bibtex
If you find our work helpful for your research, please consider citing the following BibTeX entry.

```bibtex
@inproceedings{liu2022task,
  title={Task-adaptive Spatial-Temporal Video Sampler for Few-shot Action Recognition},
  author={Liu, Huabin and Lv, Weixian and See, John and Lin, Weiyao},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  pages={6230--6240},
  year={2022}
}
```

## Contact

Please feel free to contact huabinliu@sjtu.edu.cn if you have any questions.

