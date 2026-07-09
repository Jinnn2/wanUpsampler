# 当前进度
本阶段计划通过训练一个LoRA，作用于训练路径，实现让第 3 步带 LoRA 直接完成原本 step3+step4 的 LR clean 收敛，也就是：
```
pred = x_pre_step3_lr - sigma_3 * flow_base + lora(x_pre_step3_lr, t3, prompt)

loss = L1(pred, z4_lr_teacher) + 0.1 * MSE(pred, z4_lr_teacher)
```

在训练过程中，首先构建了5000条视频样本的480p LMDB，结构是：
```
x_pre_step3_lr   # 第 3 次 denoise 调用之前的 LR noisy/intermediate latent
z4_lr_teacher    # base teacher 4 步跑完的 clean LR latent
z0_hr            # 从 clean LMDB 复制来的 HR clean latent，当前 LoRA 训练不用
prompt
seed
meta
```

配置是如下，考虑更新denoise部分，然后对注意力和flow全量更新，大概1e8的参数：
```
vbase_model: dit
target_modules: q,k,v,o,ffn.0,ffn.2
rank: 32
base model frozen，只优化 requires_grad=True 的 LoRA 参数
```

# 当前效果评估
现在使用四卡并行跑minibatch，一步需要30s，对应训练10000步需要3~4天，我在中间取了几个checkpoint生成视频。
[1/10] idx=00 seed=42 l1 original=0.038727 lora=0.034598 winner=lora
[2/10] idx=01 seed=43 l1 original=0.024810 lora=0.022052 winner=lora
[3/10] idx=02 seed=44 l1 original=0.038825 lora=0.032708 winner=lora
[4/10] idx=03 seed=45 l1 original=0.024380 lora=0.021113 winner=lora
[5/10] idx=04 seed=46 l1 original=0.030369 lora=0.027408 winner=lora
[6/10] idx=05 seed=47 l1 original=0.031258 lora=0.030290 winner=lora
[7/10] idx=06 seed=48 l1 original=0.027089 lora=0.024307 winner=lora
[8/10] idx=07 seed=49 l1 original=0.045328 lora=0.038190 winner=lora
[9/10] idx=08 seed=50 l1 original=0.036556 lora=0.034380 winner=lora
[10/10] idx=09 seed=51 l1 original=0.023311 lora=0.020712 winner=lora
当前的训练中，loss能降下来，LoRA实现了部分“补细节”的功能，对照origin3，使用LoRA后的视频细节更丰富，基本实现了语义丰富的目标。
# 当前问题
现在的问题主要有以下几个：
## 1、从输出来看，还是存在模糊
我推测问题有以下几个方面：

1）由于训练集之前是小训练集（64），可能出现了过拟合，所以我计划优先完成5000LMDB（已完成）然后继续训练5000步；

2）可能是LoRA的结构不太对，下面我会尝试去掉attention层，避免空间分布的糊的问题；

3）可能是我的loss设置的不太好：我看论文给的方法是学习对应步的velocity，但是我这里其实是要把两步压成一步，所以注定了要学endpoint，也就是学习最后生成的和加上LoRA的当前态。loss我现在只知道每种都代表啥，但是怎么配怎么选其实还不太清楚，这个我打算前两步做不成功就去读论文。
## 2、没有用论文中的on_policy策略
这个我其实先跑了一版on_policy的，体现出来就是训练的每一次要带上LoRA从头跑student和teacher，然后对应的优化。但是这一版我直接没训成功，loss压不下去，输出基本都是糊的，推测因为前两步比较重要，带LoRA跑可能会出问题。因此现在的版本是step3——only的LoRA，只在第三步训，只在第三步用，这样我就可以缓存一堆跑完step2的，在训练的时候调这个跑，论文说的on_policy策略就没啥用了。
结构大致是（使用GPT生图）：
![alt text](current_lora_upsampler_diagram-1.png)