# 当前进度
本阶段计划通过训练一个LoRA，作用于训练传播路径，实现让第 3 步带 LoRA 直接完成原本 step3+step4 的 LR clean 收敛，也就是：
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
# 当前问题
现在的问题主要有以下几个：
## 1、从输出来看，还是存在模糊的问题
我推测问题有以下几个方面：
1）由于训练集之前是小训练集（64），可能出现了过拟合，所以我计划优先完成5000LMDB（已完成）然后继续训练5000步；
2）可能是
## 2、没有用论文中的on_policy策略
