
from utils.LLM.gpt_utils import Chat_w_Vision, Chat
from PIL import Image
import json

def extract_SG_answer(text):
    import re
    # Extracting Answer2
    answer2_match = re.search('Answer2:(.*?)Question', text, re.DOTALL)
    answer2 = answer2_match.group(1).strip() if answer2_match else None
    if answer2 is None:
        answer2_match = re.search('Answer 2:(.*?)Question', text, re.DOTALL)
        answer2 = answer2_match.group(1).strip() if answer2_match else None

    # Extracting Answer3
    answer3_match = re.search('Answer3:(.*?)$', text, re.DOTALL)
    answer3 = answer3_match.group(1).strip() if answer3_match else None
    if answer3 is None:
        answer3_match = re.search('Answer 3:(.*?)$', text, re.DOTALL)
        answer3 = answer3_match.group(1).strip() if answer3_match else None
    
    d = {}
    if answer2 and answer3:
        answer2 = answer2.lower()
        answer3 = answer3.lower()
        objects = []
        tuples = []
        for line in answer3.split("\n"):
            line = line.lower()
            relation_match = re.search('\d+\.(.*)', line)
            relation = relation_match.group(1) if relation_match else None
            if relation is not None:
                relation = [s.strip() for s in relation.split("->")]
                tuples.append(relation)
                objects.append(relation[0])
                objects.append(relation[-1])
        objects = list(set(objects))
        d["objects"] = objects
        d["tuples"] = tuples
    return d


def filter_valid_desc(desc):
    filtered_desc = []
    for text in desc:
        if not text.startswith("A photo of a <asset0>."):
            filtered_desc.append("")
        else:
            filtered_desc.append(text)
    return filtered_desc


def construct_basic_SG(image_path, show_chats=False):
    from templates.parsing_prompts import scene_desc_question, construct_sg_question
    image = Image.open(image_path)
    
    # Ask GPT to describe scene
    gpt = Chat_w_Vision(image)
    
    response = gpt.ask_GPT(scene_desc_question, show_chats)

    # Then extract the scene graph
    response = gpt.ask_GPT(construct_sg_question, show_chats)
    basic_SG = extract_SG_answer(response)
    return basic_SG


def construct_node_desc(image_path, basic_SG_path):
    from templates.parsing_prompts import classify_template, animate_question, nonanimate_question

    image = Image.open(image_path)
    with open(basic_SG_path) as f:
        basic_SG = json.load(f)
        
    # classify nodes into three categories: Animate object(0), Non-Animate object(1), Background object(2).
    text = classify_template + str(basic_SG["objects"]) + "\noutputs: "
    result = Chat().ask_GPT(text)
    node_types = [res.strip() for res in result.strip()[1:-1].split(",")]
    
    count = 0
    descs = []
    alias = []
    for name, node_type in zip(basic_SG["objects"], node_types):
        text=f"describe the {name} in the image briefly. Here are some examples:"
        name_alias = f"<{name.split(' ')[-1]}-{count}>"
        if node_type=="1":
            question = text + nonanimate_question
            response = Chat_w_Vision(image).ask_GPT(question)
        elif node_type=="0":
            question = text + animate_question
            response = Chat_w_Vision(image).ask_GPT(question)
        else:
            response = "" # skip background objects
            name_alias = name.split(' ')[-1]
            count -= 1

        descs.append(response)
        alias.append(name_alias)
        count += 1

    descs = filter_valid_desc(descs) # use the basic prompt if GPT fails
    basic_SG["desc"] = descs
    basic_SG["alias"] = alias

    return basic_SG
    