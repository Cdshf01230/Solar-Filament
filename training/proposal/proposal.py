import os

if os.environ['SOLAR_MODEL_KIND'] == 'convnext':
    from proposal_convnext import *
elif os.environ['SOLAR_MODEL_KIND'] == 'vitsplus':
    from proposal_vitsplus import *
else:
    raise ValueError('Unknown backbone')
