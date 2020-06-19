"""
this file takes the converted blazeface coreml model from convert_blazeface.py 
and adds Non-maximum suppresion to create a pipeline.
currently multiple outputs in coreML with tf2.0 Keras is not working so working around to change the single
output in coreml to multiple output
"""
import coremltools
from coremltools.models import datatypes
from coremltools.models.pipeline import *
from PIL import Image

include_landmarks = False
blazeface_coreml = coremltools.models.MLModel("./coreml_models/blazeface.mlmodel") 
blazeface_coreml._spec.description.output.pop(-1)
blazeface_coreml._spec.neuralNetwork.layers.pop(-1)

#adding the boxes output layer
blazeface_coreml._spec.neuralNetwork.layers.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.MergeFromString(b'')
blazeface_coreml._spec.neuralNetwork.layers[-1].name = "boxes_pre"
blazeface_coreml._spec.neuralNetwork.layers[-1].input.append("model/concatenate_3/concat")
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].rank = 3
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend([1, 896, 16])
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].rank = 3
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend([1,896, 4])
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.strides.extend([1,1,1])
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.beginIds.extend([0, 0, 0])
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.endIds.extend([2147483647, 2147483647, 4])
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.beginMasks.extend([True, True, True])
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.endMasks.extend([True, True, False])
blazeface_coreml._spec.neuralNetwork.layers[-1].output.append("boxes_pre")
#squeezing the first dimension
blazeface_coreml._spec.neuralNetwork.layers.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].squeeze.MergeFromString(b'')
blazeface_coreml._spec.neuralNetwork.layers[-1].name = "boxes"
blazeface_coreml._spec.neuralNetwork.layers[-1].input.append("boxes_pre")
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].rank = 3
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend([1, 896, 4])
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].rank = 2
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend([896, 4])
blazeface_coreml._spec.neuralNetwork.layers[-1].squeeze.squeezeAll = True
blazeface_coreml._spec.neuralNetwork.layers[-1].output.append("boxes")

#creating the landmarks output layer
confidence_index = -6
if include_landmarks:
    confidence_index = -8
    blazeface_coreml._spec.neuralNetwork.layers.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.MergeFromString(b'')
    blazeface_coreml._spec.neuralNetwork.layers[-1].name = "landmarks_pre"
    blazeface_coreml._spec.neuralNetwork.layers[-1].input.append("model/concatenate_3/concat")
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].rank = 3
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend([1, 896, 16])
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].rank = 3
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend([1,896, 12])
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.strides.extend([1,1,1])
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.beginIds.extend([0, 0, 4])
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.endIds.extend([2147483647, 2147483647, 16])
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.beginMasks.extend([True, True, False])
    blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.endMasks.extend([True, True, True])
    blazeface_coreml._spec.neuralNetwork.layers[-1].output.append("landmarks_pre")
    
    blazeface_coreml._spec.neuralNetwork.layers.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].squeeze.MergeFromString(b'')
    blazeface_coreml._spec.neuralNetwork.layers[-1].name = "landmarks"
    blazeface_coreml._spec.neuralNetwork.layers[-1].input.append("landmarks_pre")
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].rank = 3
    blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend([1, 896, 12])
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor.add()
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].rank = 2
    blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend([896, 12])
    blazeface_coreml._spec.neuralNetwork.layers[-1].squeeze.squeezeAll = True
    blazeface_coreml._spec.neuralNetwork.layers[-1].output.append("landmarks")

# creating a new layer by squeezing confidence output
blazeface_coreml._spec.neuralNetwork.layers.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].sliceStatic.MergeFromString(b'')
blazeface_coreml._spec.neuralNetwork.layers[-1].name = "box_confidence"
blazeface_coreml._spec.neuralNetwork.layers[-1].input.append(blazeface_coreml._spec.neuralNetwork.layers[confidence_index].output[0])
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].rank = 3
blazeface_coreml._spec.neuralNetwork.layers[-1].inputTensor[0].dimValue.extend([1, 896, 1])
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor.add()
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].rank = 2
blazeface_coreml._spec.neuralNetwork.layers[-1].outputTensor[0].dimValue.extend([896, 1])
blazeface_coreml._spec.neuralNetwork.layers[-1].squeeze.squeezeAll = True
blazeface_coreml._spec.neuralNetwork.layers[-1].output.append("box_confidence")

#adding the output nodes to description
#adding box score layers
blazeface_coreml._spec.description.output.add()
blazeface_coreml._spec.description.output[0].name = "box_confidence"
blazeface_coreml._spec.description.output[0].type.multiArrayType.shape.extend([896, 1])
blazeface_coreml._spec.description.output[0].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE

#adding box output
blazeface_coreml._spec.description.output.add()
blazeface_coreml._spec.description.output[1].name = "boxes"
blazeface_coreml._spec.description.output[1].type.multiArrayType.shape.extend([896, 4])
blazeface_coreml._spec.description.output[1].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE

#adding landmark output
if include_landmarks:
    blazeface_coreml._spec.description.output.add()
    blazeface_coreml._spec.description.output[2].name = "landmarks"
    blazeface_coreml._spec.description.output[2].type.multiArrayType.shape.extend([896, 12])
    blazeface_coreml._spec.description.output[2].type.multiArrayType.dataType = datatypes._FeatureTypes_pb2.ArrayFeatureType.DOUBLE

nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3
for i in range(2):
    blazeface_output = blazeface_coreml._spec.description.output[i].SerializeToString()
    nms_spec.description.input.add() 
    nms_spec.description.input[i].ParseFromString(blazeface_output)
    nms_spec.description.output.add() 
    nms_spec.description.output[i].ParseFromString(blazeface_output)
    
nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [1, 4] 
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType 
    ma_type.shapeRange.sizeRanges.add() 
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0 
    ma_type.shapeRange.sizeRanges[0].upperBound = -1 
    ma_type.shapeRange.sizeRanges.add() 
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i] 
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i] 
    del ma_type.shape[:]
    
nms = nms_spec.nonMaximumSuppression 
nms.confidenceInputFeatureName = "box_confidence" 
nms.coordinatesInputFeatureName = "boxes" 
nms.confidenceOutputFeatureName = "confidence" 
nms.coordinatesOutputFeatureName = "coordinates" 
nms.iouThresholdInputFeatureName = "iouThreshold" 
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.5
default_confidence_threshold = 0.75
nms.iouThreshold = default_iou_threshold 
nms.confidenceThreshold = default_confidence_threshold
nms.stringClassLabels.vector.extend(["face"])
nms_model = coremltools.models.MLModel(nms_spec) 


input_features = [("input_image", datatypes.Array(3,128,128)), ("iouThreshold", datatypes.Double()),
("confidenceThreshold", datatypes.Double())] #cannot directly pass imageType as input type here. 
output_features = [ "confidence", "coordinates" ]
pipeline = Pipeline(input_features, output_features)

pipeline.add_model(blazeface_coreml._spec)
pipeline.add_model(nms_model._spec)
pipeline.spec.description.input[0].ParseFromString(blazeface_coreml._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(nms_model._spec.description.output[1].SerializeToString())

final_model = coremltools.models.MLModel(pipeline.spec) 
final_model.save("./coreml_models/blazeface_pipeline.mlmodel")

inp_image = Image.open("./sample.jpg")
inp_image = inp_image.resize((128, 128))

predictions = final_model.predict({'input_image': inp_image}, useCPUOnly=True)
print(predictions)