ModelParams = dict(
    dataset_type='endonerf',
    depth_scale=100.0,
    frame_nums=51,
    test_id=[1, 9, 17, 25, 33, 41, 49],
    is_mask=False,  # Cambiar a False
    depth_initial=True,
    accurate_mask=False,  # Cambiar a False
    is_depth=True,
)

OptimizationParams = dict(
    iterations=40_000,
)
