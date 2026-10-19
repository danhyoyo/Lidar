"""In-memory placement cases; never materialize training experiment configs."""
STAGES = (None, 'c3', 'c4', 'c5')
CASES = [(None, stage, None) for stage in STAGES]


def placement_patch(case, local_kind='simam'):
    _, local, _ = case
    patch = {'local_attention':local_kind if local else 'none'}
    if local:
        patch['local_attention_stage'] = local
    return patch


def configured(case, local_kind='simam'):
    from test_context_grouped_compatibility import fixture
    config, target = fixture(local='none')
    model = config['model']
    for key in list(model):
        if key.startswith('local_attention'):
            model.pop(key)
    model.update(placement_patch(case, local_kind))
    return config, target
