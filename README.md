
Converting Tensorflow models to TFLite via `TFLiteConverter` does not necessarily preserve the order of inputs and outputs.

One could think fixing such a nuisance would be an easy task, but the bug persists since at least 2019, 
so we might as well learn to live with it. This tool helps you hack around the bug manually.


### The bug

Create a Tensorflow model with multiple inputs and/or outputs,

```python
input1 = tf.keras.layers.Input(shape=(3, 3, 4))
input2 = tf.keras.layers.Input(shape=(3, 3, 8))
input3 = tf.keras.layers.Input(shape=(3, 3, 16))
input4 = tf.keras.layers.Input(shape=(3, 3, 32))
inputs = [input1, input2, input3, input4]

x = tf.concat(inputs, axis=-1)

output1 = tf.keras.layers.Conv2D(4, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(x)
output2 = tf.keras.layers.Conv2D(8, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(x)
output3 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(x)
outputs = [output1, output2, output3]

model = tf.keras.Model(inputs, outputs)
tf.keras.models.save_model(model, "saved_model")
```

```python
converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()
open("saved_model.tflite", "wb").write(tflite_model)
```

convert it to TFLite and...

```python
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input shapes:', *[d['shape'].tolist() for d in input_details])
print('Output shapes:', *[d['shape'].tolist() for d in output_details])
```

inputs and outputs of the `.tflite` model are shuffled!

```console
Input shapes: [1, 3, 3, 4] [1, 3, 3, 32] [1, 3, 3, 16] [1, 3, 3, 8]
Output shapes: [1, 3, 3, 4] [1, 3, 3, 16] [1, 3, 3, 8]
```

### The remedy

Okay, the input and output tensors are permuted by some permutation `perm`.
The idea is to permute them by `inverse perm` beforehand, 
so later when we convert the model to TFLite and the bug kicks in, it would just put them back in the original order.
> perm ∘ perm<sup>-1</sup>(inputs) = inputs

#### 1) See exactly how the inputs/outputs orders are wrong when adding these lines before conversion: 

```python
from tflite_converter_fixer import fixer

inputs_perm = None
outputs_perm = None
model = fixer.fix_io_order(model, inputs, inputs_perm, outputs_perm)
```
At this point, we just put our model inside another one without permuting anything.
This step is necessary since messing with the model may change how the bug manifests itself.
Indeed, that's what happens in our example:

```console
Input shapes: [1, 3, 3, 8] [1, 3, 3, 32] [1, 3, 3, 16] [1, 3, 3, 4]
Output shapes: [1, 3, 3, 16] [1, 3, 3, 4] [1, 3, 3, 8]
```

#### 2) Specify how the inputs/outputs need to be rearranged to return in correct order
* input #0 should go to 2nd place (#1, counting from 0)
* input #1 --> #3
* input #2 --> #2
* input #3 --> #0


* output #0 --> #2
* output #1 --> #0
* output #2 --> #1

```python
inputs_perm = [1, 3, 2, 0]
outputs_perm = [2, 0, 1]
model = fixer.fix_io_order(model, inputs, inputs_perm, outputs_perm)
```

```console
Input shapes: [1, 3, 3, 4] [1, 3, 3, 8] [1, 3, 3, 16] [1, 3, 3, 32]
Output shapes: [1, 3, 3, 4] [1, 3, 3, 8] [1, 3, 3, 16]
```

