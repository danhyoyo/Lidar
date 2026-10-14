"""In-memory placement cases; never materialize training experiment configs."""
from itertools import product

STAGES = (None, 'c3', 'c4', 'c5')
CASES = [(f, s, 'focal_then_local' if f and f == s else None)
         for f, s in product(STAGES, repeat=2)]
CASES += [(s, s, 'local_then_focal') for s in STAGES[1:]]


def placement_patch(case, local_kind='simam'):
    focal, local, order = case
    patch = {'c4_context':'focal' if focal else 'none',
             'local_attention':local_kind if local else 'none'}
    if focal:
        patch['focal_stage'] = focal
    if local:
        patch['local_attention_stage'] = local
    if order:
        patch['attention_order'] = order
    return patch


def configured(case, local_kind='simam'):
    from test_context_grouped_compatibility import fixture
    config, target = fixture(local='none')
    model = config['model']
    for key in list(model):
        if key.startswith(('c4_context', 'local_attention', 'focal_', 'attention_')):
            model.pop(key)
    model.update(placement_patch(case, local_kind))
    return config, target
