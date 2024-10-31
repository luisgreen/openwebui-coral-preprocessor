---------------------------------------------------------------
Coral TPU Preprocessor Filter
---------------------------------------------------------------
This filter is designed to preprocess a message containing images through a Coral TPU for augmenting a non-vision model. The filter utilizes the Coral TPU's Edge TFLite runtime and provides support for both EfficientNet and COCO models, which can be obtained from [Coral AI Models](https://coral.ai/models). These models are not included in this repository.

### Features:
- Augment input data with object detection information from the Coral TPU.
- Support for EfficientNet and COCO models.
- Ability to define custom confidence thresholds, context window size, and model paths through environment variables.
- Generates a prompt that can be used for answering questions based on the identified objects in the image(s).

### Requirements:
- tflite-runtime
- numpy

### Installation (Linux):
1. Clone this repository to your local machine.
2. Download and compile a suitable version of libedgetpu from [here](https://github.com/google-coral/libedgetpu.git). You may need to copy or move the library to the correct location, so that open-webui can access it. The recommended path is `/app/backend/data/coral`. I added a precompiled x86_64 Linux version in this repo.
3. Install Open Web UI (open-webui), which can be found at [Open Web UI](https://openwebui.com/) and follow the installation instructions available in their documentation: [Open Web UI Documentation](https://docs.openwebui.com/)
4. Configure your application to use this filter within Open Web UI (open-webui).

### Customization:
You can customize the behavior of the filter by modifying the environment variables defined in the Valves class:
- `EFFICIENTNET_MODEL_FILE`: Path to the EfficientNet TFLite model file.
- `EFFICIENTNET_LABEL_FILE`: Path to the EfficientNet labels file.
- `COCO_MODEL_FILE`: Path to the COCO TFLite model file.
- `COCO_LABEL_FILE`: Path to the COCO labels file.
- `CONFIDENCE_THRESHOLD`: Confidence threshold for the inference.
- `CONTEXT_WINDOW`: Number of tokens to use in the context window.

For more information, please refer to the source code comments and documentation on the GitHub repository.

Thank you to Google Coral AI for providing the Edge TFLite runtime and the EfficientNet and COCO models available at Coral AI Models. Also, thank you to Open Web UI (open-webui) team for their work on creating the open-source web application framework, which can be found at Open Web UI, and providing documentation at Open Web UI Documentation.

Enjoy using the Coral TPU Preprocessor Filter!
