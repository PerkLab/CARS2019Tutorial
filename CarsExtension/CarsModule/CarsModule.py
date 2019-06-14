import os
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging
import numpy as np
import scipy.ndimage

from keras.models import load_model
import cv2


#
# CarsModule
#

class CarsModule(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "CarsModule" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
It performs a simple thresholding on the input volume and optionally captures a screenshot.
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# CarsModuleWidget
#

class CarsModuleWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    self.logic = CarsModuleLogic()
    
    ScriptedLoadableModuleWidget.setup(self)
    
    self.updateTimer = qt.QTimer()
    self.updateTimer.setInterval(100)
    self.updateTimer.setSingleShot(True)

    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ["vtkMRMLStreamingVolumeNode"]
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    self.modelPathEdit = ctk.ctkPathLineEdit()
    parametersFormLayout.addRow("Keras model: ", self.modelPathEdit)
    
    #
    # threshold value
    #
    self.imageThresholdSliderWidget = ctk.ctkSliderWidget()
    self.imageThresholdSliderWidget.singleStep = 0.05
    self.imageThresholdSliderWidget.minimum = 0
    self.imageThresholdSliderWidget.maximum = 1.0
    self.imageThresholdSliderWidget.value = 0.5
    self.imageThresholdSliderWidget.setToolTip("Set threshold value for class probability.")
    parametersFormLayout.addRow("Prediction threshold", self.imageThresholdSliderWidget)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = True
    parametersFormLayout.addRow(self.applyButton)
    
    self.classLabel = qt.QLabel("0")
    classFont = self.classLabel.font
    classFont.setPointSize(32)
    self.classLabel.setFont(classFont)
    parametersFormLayout.addRow(self.classLabel)
    
    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    
    # Add vertical spacer
    self.layout.addStretch(1)


  def onUpdateTimer(self):
    newText = self.logic.getLastClass()
    self.classLabel.setText(newText)
    self.updateTimer.start()
    

  def cleanup(self):
    pass

  def onApplyButton(self):
    imageThreshold = self.imageThresholdSliderWidget.value
    modelFilePath = self.modelPathEdit.currentPath
    
    # Try to load Keras model
    
    success = self.logic.loadKerasModel(modelFilePath)
    if not success:
      logging.error("Failed to load Keras model: {}".format(modelFilePath))
      return
    
    inputVolumeNode = self.inputSelector.currentNode()
    if inputVolumeNode is None:
      logging.error("Please select a valid image node!")
      return
    
    success = self.logic.run(inputVolumeNode, imageThreshold)
    if not success:
      logging.error("Could not start classification!")
      return
    
    self.updateTimer.connect('timeout()', self.onUpdateTimer)
    self.updateTimer.start()
    

#
# CarsModuleLogic
#

class CarsModuleLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
  
  def __init__(self):
    self.model = None
    self.observerTag = None
    self.lastObservedVolumeId = None
    self.lastClass = ""
    self.model_input_size = None
    self.classes = ['A', 'C', 'None', 'R', 'S']

  
  def getLastClass(self):
    return self.lastClass
  
  
  def loadKerasModel(self, modelFilePath):
    """
    Tries to load Keras model for classifiation
    :param modelFilePath: full path to saved model file
    :return: True on success, False on error
    """
    try:
      self.model = load_model(modelFilePath)
    except:
      self.model = None
      return False
    
    return True
  

  def hasImageData(self,volumeNode):
    """This is an example logic method that
    returns true if the passed in volume
    node has valid image data
    """
    if not volumeNode:
      logging.debug('hasImageData failed: no volume node')
      return False
    if volumeNode.GetImageData() is None:
      logging.debug('hasImageData failed: no image data in volume node')
      return False
    return True

  def isValidInputOutputData(self, inputVolumeNode, outputVolumeNode):
    """Validates if the output is not the same as input
    """
    if not inputVolumeNode:
      logging.debug('isValidInputOutputData failed: no input volume node defined')
      return False
    if not outputVolumeNode:
      logging.debug('isValidInputOutputData failed: no output volume node defined')
      return False
    if inputVolumeNode.GetID()==outputVolumeNode.GetID():
      logging.debug('isValidInputOutputData failed: input and output volume is the same. Create a new volume for output to avoid this error.')
      return False
    return True


  def run(self, inputVolumeNode, imageThreshold):
    """
    Run the classification algorithm on each new image
    """
    
    if self.model is None:
      logging.error('Cannot run classification without model!')
      return False

    image = inputVolumeNode.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    self.model_input_size = self.model.layers[0].input_shape[1]
    
    if self.observerTag is None:
      self.lastObservedVolumeId = inputVolumeNode.GetID()
      self.observerTag = inputVolumeNode.AddObserver(vtk.vtkCommand.ModifiedEvent, self.onImageModified)
      logging.info('Processing started')
    else:
      lastVolumeNode = slicer.util.getNode(self.lastObservedVolumeId)
      if lastVolumeNode is not None:
        lastVolumeNode.RemoveObserver(self.observerTag)
        self.observerTag = None
        self.lastObservedVolumeId = None
      logging.info('Processing ended')
    
    return True
  
  
  def onImageModified(self, caller, event):
    image_node = slicer.util.getNode(self.lastObservedVolumeId)
    image = image_node.GetImageData()
    shape = list(image.GetDimensions())
    shape.reverse()
    components = image.GetNumberOfScalarComponents()
    if components > 1:
      shape.append(components)
      shape.remove(1)
    input_array = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    
    # Resize image and scale between 0.0 and 1.0
    
    resized_input_array = cv2.resize(input_array, (self.model_input_size, self.model_input_size))
    resized_input_array = resized_input_array / (resized_input_array.max())
    resized_input_array = np.expand_dims(resized_input_array, axis=0)
    
    # Run prediction and print result
    
    prediction = self.model.predict(resized_input_array)
    maxPredictionIndex = prediction.argmax()
    self.lastClass = self.classes[maxPredictionIndex]
    
    print("Prediction: {} at {:2.2%} probability".format(self.lastClass, prediction[0, maxPredictionIndex]))
    

class CarsModuleTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_CarsModule1()

  def test_CarsModule1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import SampleData
    SampleData.downloadFromURL(
      nodeNames='FA',
      fileNames='FA.nrrd',
      uris='http://slicer.kitware.com/midas3/download?items=5767')
    self.delayDisplay('Finished with download and loading')

    volumeNode = slicer.util.getNode(pattern="FA")
    self.assertIsNotNone( self.logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
