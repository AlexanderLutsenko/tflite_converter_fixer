import tensorflow as tf
from tflite_converter_fixer import fixer


input1 = tf.keras.layers.Input(shape=(3, 3, 4))
input2 = tf.keras.layers.Input(shape=(3, 3, 8))
input3 = tf.keras.layers.Input(shape=(3, 3, 16))
input4 = tf.keras.layers.Input(shape=(3, 3, 32))

input = tf.concat([input1, input2, input3, input4], axis=-1)

o1 = tf.keras.layers.Conv2D(4, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(input)
o2 = tf.keras.layers.Conv2D(8, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(input)
o3 = tf.keras.layers.Conv2D(16, (1, 1), activation='relu', input_shape=(1, 3, 3, 64))(input)

model = tf.keras.Model(inputs=[input1, input2, input3, input4], outputs=[o1,o2,o3])


# inputs_perm = None
# outputs_perm = None
inputs_perm = [1, 3, 2, 0]
outputs_perm = [2, 0, 1]
model = fixer.fix_io_order(model, [input1, input2, input3, input4], inputs_perm, outputs_perm)


tf.keras.models.save_model(model, "saved_model")


converter = tf.lite.TFLiteConverter.from_saved_model("saved_model")
tflite_model = converter.convert()
open("saved_model.tflite", "wb").write(tflite_model)


interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('Input shapes:', *[d['shape'].tolist() for d in input_details])
print('Output shapes:', *[d['shape'].tolist() for d in output_details])
