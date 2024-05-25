class ExperimentType:
    bc = 'BinaryClassification'
    mcc = 'MultiClassClassification'
    reg = 'Regression'

    def __init__(self, n_classes: int, name: str):
        self.n_classes = n_classes
        self.name = name


