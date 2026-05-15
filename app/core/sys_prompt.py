# sys_prompt.py
# Pre-defined system prompts for different tasks in the
# Vision Understanding Utility (VisPROC) framework.

# type hinting
from __future__ import annotations

# standard library imports
from PIL import Image
import io
from typing import Any

# System prompts for different tasks
class SystemPrompts:

    # Caption Generation
    class CaptionGeneration:

        # system prompt for caption generation
        # NOTE: this is shared across all objects of this class
        SYS_PROMPT: str = \
        """System Prompts: You will assist with image analysis tasks. Your role is to analyze the given image and generate a descriptive caption for the provided image. Please follow the rules:

        - The output format should be: "Caption: *."
        - Cover all primary objects in the given image.
        - Focus on the relationship between the objects.
        - Strictly keep the caption in one phrase.
        - Do not use a celebrities name such as "harry potter" in the caption, use "boy", "girl", "man", "woman" instead.
        - Do not include any unnecessary commentary or explanations.
        - Do not say anything else from the output format."""
        
        # Method to get the full prompt for processor
        # input: bytes
        def get_message(self, input: bytes) -> list[Any]:
            
            # Open the provided image, convert to RGB, and resize to 256x256
            image = Image.open(io.BytesIO(input)).convert("RGB").resize((256, 256), Image.LANCZOS)

            # Create the full prompt
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "Analyze the given image and generate a descriptive caption."}
                    ]
                }
            ]

            # Return the full message
            return messages

    # Primary Object Detection
    class PrimaryObjectDetection:

        # system prompt for primary object detection
        # NOTE: this is shared across all objects of this class
        SYS_PROMPT: str = \
        """System Prompts: You will assist with image analysis tasks. Your role is to analyze the given image and identify the primary objects and list their names. Please follow the rules:
        
            - Strictly follow the output format: “Objs: [***, ***, ...]”.
            - Cover all primary objects in the given image.
            - Only provide the object names of the primary objects.
            - Make sure the object names are in one-word noun format, do
            not add any descriptive adjectives or descriptive nouns.
            - Keep only the object name in its singular form.
            - Do not use a persons' name as object name such as "harry
            potter", use "boy", "girl", "man”, "woman" instead.
            - Separate each object name with a comma.
            - Do not repeat any same word.
            - Do not include any unnecessary commentary or explanations.
            - Do not say anything else from the output format."""
        
        # Method to get the full prompt for processor
        # input: bytes
        def get_message(self, input: bytes) -> list[Any]:
            
            # Open the provided image, convert to RGB, and resize to 256x256
            image = Image.open(io.BytesIO(input)).convert("RGB").resize((256, 256), Image.LANCZOS)

            # Create the full prompt
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": "Analyze the given image and identify the primary objects and list their names."}
                    ]
                }
            ]

            # Return the full message
            return messages
        
    # Primary Concept Generation
    class PrimaryConceptGeneration:

        # concept list
        CONCEPTS: list[str] = \
            ['animal', 'human', 'vehicle', 'building', 'clothing',
            'weapon', 'plant', 'appliance', 'tool', 'container',
            'body part', 'furniture', 'device', 'fabric', 'fruit',
            'vegetable', 'insect', 'landscape feature', 'water body',
            'accessory', 'sports equipment', 'food', 'drink', 'light source',
            'weather phenomenon', 'jewelry', 'musical instrument',
            'structure', 'flying vehicle', 'toy', 'kitchen item',
            'writing tool', 'gardening tool', 'scientific equipment',
            'furniture accessory', 'roadway', 'weaponry accessory',
            'sports field', 'money', 'timekeeping device', 'decoration',
            'art', 'stationery', 'kitchen appliance', 'rock/mineral',
            'soil/substrate', 'climate/atmosphere component']

        # system prompt for primary concept generation
        # NOTE: this is shared across all objects of this class
        SYS_PROMPT: str = \
        "System Prompts: You will assist with text-related tasks. I will give you a list of object names in the following form: Objs: [***, ***, ***, ...]. Your role is to categorized each object into the concept lists:" \
        + "\n\n[" + ", ".join(CONCEPTS) + "]\n\n" + \
        """Please follow the rules:
        - Strictly follow the output format: “Concept: [***, ***, ...]”.
        - Only output the concept name in the given list.
        - You can categorize the object into more than one concepts.
        - If two objects belong to one concept, only list concept once."""

        # Method to get the full prompt for processor
        # input: tuple of (bytes, str)
        # Object string is in the format: "Objs: [***, ***, ...]"
        def get_message(self, input: tuple[bytes, str]) -> list[Any]:

            # unpack the input tuple
            image_input, object_string = input

            # Open the provided image, convert to RGB, and resize to 256x256
            image = Image.open(io.BytesIO(image_input)).convert("RGB").resize((256, 256), Image.LANCZOS)

            # Create the full prompt
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": f"Categorize each object into the concept lists based on the given image and object names. {object_string}"}
                    ]
                }
            ]

            # Return the full message
            return messages


# test-code
if __name__ == "__main__":

    # set cache env variable
    import os
    # HF_HOME and TRANSFORMERS_CACHE
    os.environ["HF_HOME"] = "./cache/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = "./cache/huggingface/transformers"
    # TORCH_HOME for PyTorch models
    os.environ["TORCH_HOME"] = "./cache/torch"

    # import Auto Processor for testing
    from transformers import AutoProcessor
    from qwen_vl_utils import process_vision_info

    # set up processor
    model_dir = "data/Qwen/Qwen3-VL-8B-Instruct"
    processor = AutoProcessor.from_pretrained(model_dir)

    # Example usage of the system prompts
    caption_gen = SystemPrompts.CaptionGeneration()
    primary_obj_det = SystemPrompts.PrimaryObjectDetection()
    primary_concept_gen = SystemPrompts.PrimaryConceptGeneration()

    # Example input
    image = Image.open("testing_image/chureito_pagoda.jpg")
    image_input: bytes = io.BytesIO()
    image.save(image_input, format = "JPEG")
    image_input = image_input.getvalue()
    object_string = "Objs: [cat, sofa, lamp]"

    # Get messages for each task
    caption_message = caption_gen.get_message(image_input)
    obj_det_message = primary_obj_det.get_message(image_input)
    concept_gen_message = primary_concept_gen.get_message((image_input, object_string))

    # Process the messages with the processor
    for msg in [caption_message, obj_det_message, concept_gen_message]:

        # Verbose
        print(f"\033[1;34m[INFO]\033[0m Processing message for task:\n       content: {msg[0]['content']}")

        # process
        text = processor.apply_chat_template(
        msg, tokenize = False, add_generation_prompt = True
        )
        image_inputs, video_inputs = process_vision_info(msg)

        # print the results
        print(f"\033[1;32m[OK]\033[0m Processed Text:")
        print(text)
