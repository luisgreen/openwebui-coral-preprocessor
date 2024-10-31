"""
title:  Coral TPU Preprocessor
author: Luis Chacón
description: Filter to preprocess a message with images through a Coral TPU to augment a non-vision model.
author_url: https://github.com/luisgreen
funding_url: https://github.com/luisgreen
requirements: tflite-runtime, numpy
version: 0.1.0
changelog:
    v0.1.0 - Added efficientnet support for more augmentation and more specifics.
    v0.0.4 - Handle a proper response if there is an image but nothing could not be identified.
           - Enhaced response prompt.
    v0.0.3 - Added Counter so that the question can gather more context on cluster of things.
    v0.0.2 - Improved query prompt.
    v0.0.1 - Initial
"""

from PIL import Image
from io import BytesIO
import os
import numpy as np
import base64
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from collections import Counter
from pydantic import BaseModel, Field
from typing import Optional
from open_webui.utils.misc import (
    get_last_user_message_item,
    get_last_user_message,
    get_content_from_message,
)
import base64
import logging


class Filter:
    class Valves(BaseModel):
        EFFICIENTNET_MODEL_FILE: str = Field(
            os.getenv(
                "EFFICIENTNET_MODEL_FILE",
                "efficientnet-edgetpu-L_quant_edgetpu.tflite",
            ),
            description="Coral TPU TFlite Model",
        )
        EFFICIENTNET_LABEL_FILE: str = Field(
            os.getenv("EFFICIENTNET_LABEL_FILE", "imagenet_labels.txt"),
            description="Labels supplied with the model",
        )
        COCO_MODEL_FILE: str = Field(
            os.getenv(
                "MODEL_FILE",
                "tf2_ssd_mobilenet_v1_fpn_640x640_coco17_ptq_edgetpu.tflite",
            ),
            description="Coral TPU TFlite Model",
        )
        COCO_LABEL_FILE: str = Field(
            os.getenv("LABEL_FILE", "coco_labels.txt"),
            description="Labels supplied with the model",
        )
        CONFIDENCE_THRESHOLD: str = Field(
            os.getenv("CONFIDENCE_THRESHOLD", "0.35"),
            description="Confidence threshold for the inference",
        )
        CONTEXT_WINDOW: int = Field(
            os.getenv("CONTEXT_WINDOW", 30000),
            description="The number of tokens to use in the context window",
        )
        pass

    def __init__(self):
        self.valves = self.Valves()
        efnt_model_path = (
            f"/app/backend/data/coral/{self.valves.EFFICIENTNET_MODEL_FILE}"
        )
        efnt_labels_path = (
            f"/app/backend/data/coral/{self.valves.EFFICIENTNET_LABEL_FILE}"
        )

        coco_model_path = f"/app/backend/data/coral/{self.valves.COCO_MODEL_FILE}"
        coco_labels_path = f"/app/backend/data/coral/{self.valves.COCO_LABEL_FILE}"

        # Initialize the interpreters
        self.efnt_interpreter = Interpreter(
            efnt_model_path,
            experimental_preserve_all_tensors=True,
            experimental_delegates=[
                load_delegate("/app/backend/data/coral/libedgetpu.so.1.0")
            ],
        )

        self.coco_interpreter = Interpreter(
            coco_model_path,
            experimental_preserve_all_tensors=True,
            experimental_delegates=[
                load_delegate("/app/backend/data/coral/libedgetpu.so.1.0")
            ],
        )

        # Initialize the labels
        with open(efnt_labels_path, "r") as f:
            self.efnt_labels = {i: line.strip()
                                for i, line in enumerate(f.readlines())}

        with open(coco_labels_path, "r") as f:
            self.coco_labels = {i: line.strip()
                                for i, line in enumerate(f.readlines())}

    def analyze_results(self, boxes, classes, scores, num_boxes):
        objects_detected = list()
        for i in range(int(num_boxes)):
            score = scores[i]
            if float(score) >= float(self.valves.CONFIDENCE_THRESHOLD):
                objects_detected.append(self.efnt_labels[classes[i]])
        return objects_detected

    def summarize_counts(self, objects):
        counts = Counter(objects)
        summary = []
        for obj, count in counts.items():
            if count > 1:
                summary.append(f"{count} {obj}s")
            else:
                summary.append(f"{count} {obj}")

        return ", ".join(summary)

    def preprocess_image_for_coco(self, image_data, size):
        image = image_data.resize((size))
        image = np.expand_dims(image, axis=0)
        return image

    def preprocess_image_for_efficient_net(self, image_data, size, input_shape):
        image = image_data.convert("RGB").resize(size, Image.LANCZOS)
        image = np.asarray(image).astype(np.uint8)
        image = np.expand_dims(image, axis=0)
        image = np.reshape(image, input_shape)
        return image

    def get_efnt_objects(self, output_details):
        output_data = self.efnt_interpreter.get_tensor(output_details)[0]

        return [
            self.efnt_labels.get(value[0], "Unknown")
            for value in enumerate(output_data)
            if value[1] > 0
        ]

    def summarize_ccoco_elements(self, objects):
        counts = Counter(objects)
        summary = []
        for obj, count in counts.items():
            if count > 1:
                summary.append(f"{count} {obj}s")
            else:
                summary.append(f"{count} {obj}")

        return ", ".join(summary)

    def get_coco_elements(self, output_details):
        objects_detected = list()
        scores = self.coco_interpreter.get_tensor(output_details[0]["index"])[
            0
        ]  # Confidence of detected objects
        boxes = self.coco_interpreter.get_tensor(output_details[1]["index"])[
            0
        ]  # Bounding box coordinates of detected objects
        num_boxes = self.coco_interpreter.get_tensor(output_details[2]["index"])[
            0
        ]  # Number of boxed detected
        classes = self.coco_interpreter.get_tensor(output_details[3]["index"])[
            0
        ]  # Class index of detected objects

        for i in range(int(num_boxes)):
            score = scores[i]
            if float(score) >= float(self.valves.CONFIDENCE_THRESHOLD):
                objects_detected.append(self.coco_labels[classes[i]])

        return objects_detected

    def assemble_prompt(self, message_prompt, coco_objects, efnt_objects):
        elements = self.summarize_counts(coco_objects)
        specific = ", ".join(efnt_objects)
        return f"""
        Your task is to create an answer to the question, based on the context given here. This context is a serie of elements that
        were identified by a TPU previously, so dont make assumptions on the likelyhood of the elements.

        Also you may or may not have a list of specific objects which are possible exact names of the elements found so you can know
        what the elements are about. This list comes from another TPU model too.

        Pay attention to the given question, and elaborate an answer that correlates the identified elements with the question and
        in an accurate, clear and concise manner. Do your best to return a descriptive answer that would consider and explain what
        the question contain.

        If there are elements identified, try to infere, if possible, a very detailed scene where the elements can be present together
        and explain with details where all the elements can coexist. If you dont know, just skip this step.

        You are allowed to mention something like 'My Coral TPU identified the following elements: <elements> and the following specific
        objects: <specific objects here>'

        If the list of elements empty, you can say that 'My Coral TPU was not able to identify anything'. If you consider there is no
        other relevant information, you can stop responding inmediately. Otherwise if you find there is something interesting you can
        share based on the question, focus on it and answer.

        <CONDITIONS>
        - Don't include previous answers as context for this analysis.
        - Attached to this message there is one or more images, ignore them, just focus in the context.
        - Dont mention that the elements where separated by comma.
        - Don't mention that you are only a text based model, just use the context to answer.
        - Always mention which elements were given in the response.

        <RESPONSE>
        Explanation here if any
        Final answer here

        Question: {message_prompt}
        Elements: {elements}
        Specific Objects: {specific}
        """

    async def inlet(
        self,
        body: dict,
        __user__: Optional[dict] = None,
        __event_emitter__=None,
    ) -> dict:
        images_to_process = set()

        coco_objects_collection = list()
        efnt_objects_collection = list()
        exist_images = False

        messages = body.get("messages")
        message = get_last_user_message_item(messages)
        message_prompt = get_last_user_message(messages)

        if "images" in message:
            exist_images = True
            images_to_process = [image for image in message["images"]]

            self.efnt_interpreter.allocate_tensors()
            efnt_input_details = self.efnt_interpreter.get_input_details()[0]
            efnt_output_details = self.efnt_interpreter.get_output_details()
            efnt_size = (
                efnt_input_details["shape"][1],
                efnt_input_details["shape"][2],
            )

            self.coco_interpreter.allocate_tensors()
            coco_input_details = self.coco_interpreter.get_input_details()[0]
            coco_output_details = self.coco_interpreter.get_output_details()
            coco_size = (
                coco_input_details["shape"][1],
                coco_input_details["shape"][2],
            )

            for image in images_to_process:
                image_data = Image.open(BytesIO(base64.b64decode(image)))

                efnt_input_data = self.preprocess_image_for_efficient_net(
                    image_data, efnt_size, efnt_input_details["shape"]
                )

                self.efnt_interpreter.set_tensor(
                    efnt_input_details["index"], efnt_input_data
                )

                coco_input_data = self.preprocess_image_for_coco(
                    image_data, coco_size)
                self.coco_interpreter.set_tensor(
                    coco_input_details["index"], coco_input_data
                )

                self.efnt_interpreter.invoke()
                self.coco_interpreter.invoke()

                efnt_objects_found = self.get_efnt_objects(
                    efnt_output_details[0]["index"]
                )
                coco_objects_found = self.get_coco_elements(
                    coco_output_details)

                efnt_objects_collection = efnt_objects_collection + efnt_objects_found
                coco_objects_collection = coco_objects_collection + coco_objects_found

        if exist_images:
            message_to_replace = body["messages"].pop()
            message_to_replace["content"] = self.assemble_prompt(
                message_prompt, coco_objects_collection, efnt_objects_collection
            )
            body["messages"].append(message_to_replace)
            return body

        return body

    def outlet(self, body: dict, __user__: Optional[dict] = None) -> dict:
        return body
