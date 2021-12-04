# optimizer
base_lr = 1e-3
warmup_iters = 1000
optimizer = dict(type='AdamW', lr=base_lr)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=base_lr / 1000,
    warmup='linear',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001)
total_epochs = 600
