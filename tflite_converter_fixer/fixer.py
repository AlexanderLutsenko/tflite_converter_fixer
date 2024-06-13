import tensorflow as tf


def invert_permutation(permutation):
    if permutation is None:
        return None
    return [i for i, j in sorted(enumerate(permutation), key=lambda x: x[1])]


def permute_list(xs, permutation):
    if permutation is None:
        return xs
    return [xs[i] for i in permutation]


class OrderFixingLayer(tf.keras.layers.Layer):
    def __init__(self, nested, input_perm=None, output_perm=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nested = nested
        self.input_perm = input_perm
        self.output_perm = output_perm

    def get_config(self):
        config = super().get_config()
        config.update({
            "nested": self.nested,
            "input_perm": self.input_perm,
            "output_perm": self.output_perm,
        })
        return config

    def call(self, inputs):
        inputs = permute_list(inputs, self.input_perm)
        outputs = self.nested(inputs)
        outputs = permute_list(outputs, self.output_perm)
        return outputs


def fix_io_order(model, inputs, inputs_perm=None, outputs_perm=None):
    inputs_perm_inv = invert_permutation(inputs_perm)
    outputs_perm_inv = invert_permutation(outputs_perm)

    fixer_layer = OrderFixingLayer(model, input_perm=inputs_perm, output_perm=outputs_perm_inv)

    inputs_tf_inv = permute_list(inputs, inputs_perm_inv)
    inputs_tf_inv = [tf.keras.Input(batch_shape=t.shape, dtype=t.dtype) for t in inputs_tf_inv]
    outputs_tf = fixer_layer(inputs_tf_inv)
    model = tf.keras.Model(inputs_tf_inv, outputs_tf)
    return model
