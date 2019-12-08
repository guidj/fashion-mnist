LABEL_NAMES = ['t-shirt-or-top', 'trouser', 'pullover', 'dress', 'coat', 'sandals', 'shirt', 'sneaker', 'bag',
               'ankle-boost']

LABEL_INDEX = {name: idx for idx, name in enumerate(LABEL_NAMES)}

LABEL_WEIGHTS = {name: 1.0 for name in LABEL_NAMES}
