# DiT DIY
Build a DiT(Diffusion Transformer) from scratch(with the help of Generative AI).

## 训练伪代码
1) 取图片 (B, C, H, W)
2) 构造 (x_t, t, u_target)
3) 模型预测 u_pred (输入x_t, t)
4) 计算loss (针对batch, MSE)
5) 反向传播
6) 优化器optimizer 更新
7) 清空梯度
