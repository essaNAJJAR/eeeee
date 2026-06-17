import numpy as np


class FuzzyLogicSystem:
    def __init__(self):
        self.inputs = ['DeltaP', 'Sigma', 'Duration', 'Frequency']
        self.output = 'State'
        self.states = ['OFF', 'PARTIAL', 'ON']
        self.rules = self._create_rules()

    def _triangular_mf(self, x, a, b, c):
        if a == b == c:
            return 1.0 if x == b else 0.0
        if x < a or x > c:
            return 0.0
        if a <= x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        if b < x <= c:
            return (c - x) / (c - b) if c != b else 1.0
        return 0.0

    def _create_membership_functions(self):
        mfs = {
            'DeltaP': [
                ('low', 0, 0, 0.33),
                ('medium', 0, 0.33, 0.67),
                ('high', 0.33, 0.67, 1.0),
            ],
            'Sigma': [
                ('low', 0, 0, 0.33),
                ('medium', 0, 0.33, 0.67),
                ('high', 0.33, 0.67, 1.0),
            ],
            'Duration': [
                ('short', 0, 0, 0.33),
                ('medium', 0, 0.33, 0.67),
                ('long', 0.33, 0.67, 1.0),
            ],
            'Frequency': [
                ('low', 0, 0, 0.33),
                ('medium', 0, 0.33, 0.67),
                ('high', 0.33, 0.67, 1.0),
            ],
        }
        return mfs

    def _create_rules(self):
        mfs = self._create_membership_functions()
        rules = []
        input_keys = list(mfs.keys())
        mf_indices = [0, 1, 2]

        for dp in mf_indices:
            for sg in mf_indices:
                for dur in mf_indices:
                    for freq in mf_indices:
                        score = dp + sg + dur + freq
                        if score <= 3:
                            output = 0  # OFF
                        elif score <= 6:
                            output = 1  # PARTIAL
                        else:
                            output = 2  # ON
                        rules.append({
                            'inputs': [(input_keys[0], dp), (input_keys[1], sg),
                                       (input_keys[2], dur), (input_keys[3], freq)],
                            'output': output,
                        })
        return rules

    def evaluate(self, delta_p, sigma, duration, frequency):
        mfs = self._create_membership_functions()
        values = {
            'DeltaP': np.clip(delta_p, 0, 1),
            'Sigma': np.clip(sigma, 0, 1),
            'Duration': np.clip(duration, 0, 1),
            'Frequency': np.clip(frequency, 0, 1),
        }

        output_scores = np.zeros(3)

        for rule in self.rules:
            activation = 1.0
            for input_name, mf_idx in rule['inputs']:
                _, a, b, c = mfs[input_name][mf_idx]
                mf_val = self._triangular_mf(values[input_name], a, b, c)
                activation *= mf_val

            if activation > 0:
                output_scores[rule['output']] += activation

        total = np.sum(output_scores)
        if total > 0:
            output_scores /= total

        state_idx = np.argmax(output_scores)
        return self.states[state_idx], state_idx


def create_fuzzy_logic_system():
    return FuzzyLogicSystem()


def evaluate_fuzzy_logic(delta_p, sigma, duration, frequency):
    fis = FuzzyLogicSystem()
    return fis.evaluate(delta_p, sigma, duration, frequency)
