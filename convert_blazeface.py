#This file takes the tflite model of blazeface and converts it to coreml, tf format
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import *
from layers import *
from utils import *
import coremltools
import tfcoreml
from PIL import Image
import cv2
import matplotlib.pyplot as plt
print(tf.__version__)

tflite_path = "./tflite_models/face_detection_front.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
tf_lite_mapping = {}
for i in interpreter.get_tensor_details():
    if (("ker" in i["name"].lower()) or ("bia" in i["name"].lower())) and not ("dequant" in i["name"].lower()):
        tf_lite_mapping[i['name']] = interpreter.get_tensor(i["index"])
        
def create_blazeface(input_shape, grid_8, grid_16, batch_size = 1, pos1=512, pos2=384, data_format="channels_last"):
    x_scale = input_shape[1]
    y_scale = input_shape[2]
    
    inp_tensor = Input(shape=input_shape, batch_size=batch_size, name="input_image")
    pre_conv_out = Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding="SAME", data_format=data_format)(inp_tensor)
    act_out = ReLU()(pre_conv_out)
    
    block1 = face_block(act_out, 24)
    block2 = face_block(block1, 28)
    block3 = face_block(block2, 32, strides=(2,2))
    block4 = face_block(block3, 36)
    block5 = face_block(block4, 42)
    block6 = face_block(block5, 48, strides=(2,2))
    block7 = face_block(block6, 56)
    block8 = face_block(block7, 64)
    block9 = face_block(block8, 72)
    block10 = face_block(block9, 80)
    block11 = face_block(block10, 88)
    
    block12 = face_block(block11, 96, strides=(2,2))
    block13 = face_block(block12, 96)
    block14 = face_block(block13, 96)
    block15 = face_block(block14, 96)
    block16 = face_block(block15, 96)
    
    classifier_16 = Conv2D(filters=2, kernel_size=(1,1), strides=(1,1), name="classificator_8")(block11)
    classifier_16 = Reshape((pos1,1))(classifier_16)
    classifier_8 = Conv2D(filters=6, kernel_size=(1,1), strides=(1,1), name="classificator_16")(block16)
    classifier_8 = Reshape((pos2,1))(classifier_8)
    classifier = Concatenate(axis=1)([classifier_16, classifier_8])
    classifier = tf.clip_by_value(classifier, -100, 100)
    classifier = tf.math.sigmoid(classifier)
    #All the post processing stage involving anchor multiplication and offset adjustment is done as part of the model itself
    points_16 = Conv2D(filters=32, kernel_size=(1,1), strides=(1,1), name="regressor_8")(block11)
    points_16 = Reshape((16, 16, 2, 16))(points_16)
    point_16_box_xy = points_16[:, :, :, :, :2] / x_scale + grid_16[..., :2]
    point_16_box_wh = points_16[:, :, :, :, 2:4] / x_scale
    point_16_land_slice = points_16[:, :, :, :, 4:]

    point_16_land_xy = point_16_land_slice / x_scale + grid_16
    point_16_all = Concatenate(axis=-1)([point_16_box_xy, point_16_box_wh, point_16_land_xy])
    point_16_all = Reshape((-1, 16))(point_16_all)
    
    points_8 = Conv2D(filters=96, kernel_size=(1,1), strides=(1,1), name="regressor_16")(block16)
    points_8 = Reshape((8, 8, 6, 16))(points_8)
    point_8_box_xy = points_8[:, :, :, :, :2] / x_scale + grid_8[..., :2]
    point_8_box_wh = points_8[:, :, :, :, 2:4] / x_scale
    point_8_land_slice = points_8[:, :, :, :, 4:]
    point_8_land_xy = point_8_land_slice / x_scale + grid_8
    point_8_all = Concatenate(axis=-1)([point_8_box_xy, point_8_box_wh, point_8_land_xy])
    point_8_all = Reshape((-1, 16))(point_8_all)
    
    points = Concatenate(axis=1)([point_16_all, point_8_all])
    all_output = Concatenate(axis=-1)([points, classifier])
    
    return tf.keras.Model(inputs=[inp_tensor], outputs=[all_output])
    

data_format = "channels_last"
grid_8 = np.array(np.meshgrid(np.arange(0, 1, 1/8), np.arange(0, 1, 1/8))).transpose((1,2, 0)) + 1/16
grid_8 = np.tile(np.expand_dims(grid_8, 2), 6)
grid_16 = np.array(np.meshgrid(np.arange(0, 1, 1/16), np.arange(0, 1, 1/16))).transpose((1,2, 0)) + 1/32
grid_16 = np.tile(np.expand_dims(grid_16, 2), 6)
blazeface_tf = create_blazeface((128, 128, 3),grid_8, grid_16, batch_size=None, data_format=data_format)
restore_variables(blazeface_tf, tf_lite_mapping, data_format)
blazeface_tf.save("./keras_models/blazeface_tf.h5")

tf.keras.backend.clear_session()
coreml_tf = tf.keras.models.load_model("./keras_models/blazeface_tf.h5")
inp_node = coreml_tf.inputs[0].name[:-2].split('/')[-1]
out_node = coreml_tf.outputs[0].name[:-2].split('/')[-1]
print(inp_node, out_node)
blazeface_coreml = tfcoreml.convert(
    "./keras_models/blazeface_tf.h5",
    output_feature_names = [out_node],
    input_name_shape_dict = {inp_node: [1, *list(coreml_tf.inputs[0].shape[1:])]},
    image_input_names = [inp_node],
    image_scale = 1/127.5,
    red_bias = -1,
    green_bias = -1,
    blue_bias=-1,
    minimum_ios_deployment_target='13'
)

blazeface_coreml._spec.description.output[0].type.multiArrayType.shape.extend([1, 896, 17])
blazeface_coreml._spec.description.output[0].name = "box_landmarks_conf"
blazeface_coreml._spec.neuralNetwork.layers[-1].name = "box_landmarks_conf"
blazeface_coreml._spec.neuralNetwork.layers[-1].output[0] = "box_landmarks_conf"

blazeface_coreml.save("./coreml_models/blazeface.mlmodel")

inp_image = Image.open("./sample.jpg")
inp_image = inp_image.resize((128, 128))
inp_image_int = np.array(inp_image)
inp_image_np = inp_image_int.astype(np.float32)
inp_image_np = np.expand_dims((inp_image_np/127.5) - 1, 0)
blazeface_coreml = coremltools.models.MLModel("./coreml_models/blazeface.mlmodel")

print("Checking model sanity across tensorflow, and coreml")
tf_out = coreml_tf.predict(inp_image_np)[0]
coreml_output = blazeface_coreml.predict({"input_image": inp_image}, useCPUOnly=True)["box_landmarks_conf"][0] #currently runs only on CPU
print("Tensorflow output mean: {}".format(tf_out.mean()))
print("CoreMl output mean: {}".format(coreml_output.mean()))

box_tlbr = xywh_to_tlbr(tf_out[:, 0:4], y_first=True)
out_boxes = tf.image.non_max_suppression(box_tlbr, tf_out[:, -1], 5, score_threshold=0.75)
final_boxes = (tf_out[out_boxes.numpy(), :]*128).astype(np.int32)
final_boxes = xywh_to_tlbr(final_boxes).astype(np.int32)
for bx in final_boxes:
    cv2.rectangle(inp_image_int, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 255), 1)
landmarks = final_boxes[:, 4:-1].reshape(6,2)
plt.imshow(inp_image_int)
plt.scatter(landmarks[:, 0], landmarks[:, 1], marker="+")
plt.savefig("blazeface_out.jpg")
plt.show()