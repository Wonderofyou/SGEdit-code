
import json
import re
import copy
from utils.LLM.gpt_utils import Chat

TEMPLATE='''Learn from these examples and Complete your task.
## Your task:
Position Information:
POS_INFO
Relation Information:
REL_INFO
User Modification: 
MODIFICATION'''

TO_JSON = '''Output your answer in a json format result directly:
Am example of expect output:
{"apple": [0.55, 0.43, 0.27, 0.37]}
Note: direcly output the json format result, where the key is the object and value is the bounding box.'''

IS_INTERACTION = '''Your task is to examine a tuple containing three elements: (entity1, relation, entity2), and identify the type of relationship between the entities. There are two categories: 
1. Spatial Relationships: Indicates physical position, such as "beside", "on", "to the left of", "outside", "in front of", "landing on", "standing on" and "sitting on". 
2. Interaction Relationships: Describes actions or interactions, such as "riding", "holding", "reading".

Here are some examples to guide you:
## Examples
Input: (person, seated on, boat)
Output: spatial
Input: (person, beside, bike)
Output: spatial
Input: (person, playing, guitar)
Output: interaction
Input: (person, holding, wallet)
Output: interaction

## Your task:
Input: '''

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox
    bbox = ", ".join(list(map(lambda a: "{:.2f}".format(a), [x1, y1, x2-x1, y2-y1])))
    return bbox

def xywh2xyxy(bbox):
    x, y, w, h = bbox
    return [x, y, x+w, y+h]

def prepare_box_question(operation, modification, full_scene_graph):
    from templates.updating_prompts import box_proposal_instruction, replace_examples, add_examples, edge_examples
    # initialize question with instruction
    question = box_proposal_instruction

    # concat examples
    if operation == "replace":
        examples = replace_examples
    elif operation == "add":
        examples = add_examples
    elif operation == "edit_edge":
        examples = edge_examples
    question = question + examples

    # concat target inut
    pos_info = ""
    for obj, bbox in zip(full_scene_graph["objects"],full_scene_graph["bboxes"]):
        info = f"'{obj}': [{xyxy2xywh(bbox)}]\n"
        pos_info = pos_info + info
    pos_info = pos_info.strip()

    rel_info = ""
    for i, (s, p, o) in enumerate(full_scene_graph["tuples"]):
        info = f"{i+1}. '{s}' -> {p} -> '{o}'\n"
        rel_info = rel_info + info
    rel_info = rel_info.strip()

    target_input = TEMPLATE.replace("POS_INFO", pos_info).replace("REL_INFO", rel_info).replace("MODIFICATION", modification)
    question = question + target_input
    
    return question

def prepare_prompt_question(gen_objects, edit_info, relationships):
    from templates.updating_prompts import single_instruction, multiple_instruction
    gen_objects_text = ", ".join([f'"{obj}"' for obj in gen_objects])

    # Do additional relationship filterting for editing relationship 
    tgt_rel = edit_info.get("tgt_rel")
    if tgt_rel:
        _rel = []
        for tp in relationships:
            if tp != tgt_rel:
                isaltered = (tp[0] in gen_objects) or (tp[2] in gen_objects)
                if not isaltered:
                    _rel.append(tp)
            else:
                _rel.append(tp)
    else:
        _rel = relationships
    scene_info = "\n".join([f"{i+1}: " + " -> ".join(tp) for i, tp in enumerate(_rel)])

    text_input = f"Target objects: {gen_objects_text}\nScene information:\n{scene_info}\nExpected prompt:"
    question = single_instruction if len(gen_objects) == 1 else multiple_instruction
    question = question + text_input
    return question

def extract_edit(s):
    if s.startswith("Add"):
        pattern = "Add an object: '(.*?)'. Add tuple: \['(.*?)' -> (.*?) -> '(.*?)'\]"
        match = re.search(pattern, s)
        add_object = match.group(1)
        add_tuple = [match.group(2), match.group(3), match.group(4)]
        info = {
            "operation": "add",
            "add_object": add_object,
            "add_tuple": add_tuple,
        }
    elif s.startswith("Remove"):
        pattern = "Remove: '(.*)'"
        match = re.search(pattern, s)
        remove_object = match.group(1)
        info = {
            "operation": "remove",
            "rm_object": remove_object,
        }
    elif s.startswith("Replace"):
        pattern = "Replace '(.*)' by '(.*)'"
        match = re.search(pattern, s)
        replace_src_object = match.group(1).strip()
        replace_tgt_object = match.group(2).strip()
        info = {
            "operation": "replace",
            "replace_src_object": replace_src_object,
            "replace_tgt_object": replace_tgt_object
        }
    elif s.startswith("Modify"):
        pattern = "Modify a relationship: from \['(.*?)' -> (.*?) -> '(.*?)'\] to \['(.*?)' -> (.*?) -> '(.*?)'\]"
        match = re.search(pattern, s)
        src_rel = [match.group(1).strip(), match.group(2).strip(), match.group(3).strip()]
        tgt_rel = [match.group(4).strip(), match.group(5).strip(), match.group(6).strip()]
        info = {
                "operation": "edit_edge",
                "src_rel": src_rel,
                "tgt_rel": tgt_rel
        }
    info["modification"] = s
    return info

def update_scene_graph(sg_dict, operation, edit_info, box_updated):
    if operation == "add":
        add_object = edit_info["add_object"]
        add_tuple = edit_info["add_tuple"]
        sg_dict["objects"].append(add_object)
        sg_dict["tuples"].append(add_tuple)
        sg_dict["bboxes"].append([])
        sg_dict["alias"].append(add_object)
        sg_dict["desc"].append("")
    elif operation == "remove":
        remove_object = edit_info["rm_object"]
        for i, object in enumerate(sg_dict["objects"]):
            if object == remove_object:
                sg_dict["objects"].pop(i)
                sg_dict["alias"].pop(i)
                sg_dict["bboxes"].pop(i)
                sg_dict["desc"].pop(i)
                break
        sg_dict["tuples"] = [[s, p, o] for s, p, o in sg_dict["tuples"] if (s != remove_object and o != remove_object)]
    elif operation == "replace":
        replace_src_object = edit_info["replace_src_object"]
        replace_tgt_object = edit_info["replace_tgt_object"]
        for i, (s, _ ,o) in enumerate(sg_dict["tuples"]):
            if s == replace_src_object:
                sg_dict["tuples"][i][0] = replace_tgt_object
            if o == replace_src_object:
                sg_dict["tuples"][i][2] = replace_tgt_object
        for i, object in enumerate(sg_dict["objects"]):
            if object == replace_src_object:
                sg_dict["objects"][i] = replace_tgt_object
                sg_dict["alias"][i] = replace_tgt_object
                sg_dict["desc"][i] = ""
                break
    elif operation == "edit_edge":
        src_rel = edit_info["src_rel"]
        tgt_rel = edit_info["tgt_rel"]
        print(src_rel)
        for i, tuple in enumerate(sg_dict["tuples"]):
            print(tuple)
            if tuple == src_rel:
                sg_dict["tuples"][i] = tgt_rel
                break
    
    # update the bounding boxes
    if box_updated != None:
        for i, object in enumerate(sg_dict["objects"]):
            if object in box_updated:
                sg_dict["bboxes"][i] = box_updated[object]

    return sg_dict

def need_reverse(boxes, threshold=0.3):
    def calculate_area(x1, y1, x2, y2):
        """Calculate the area of a bounding box."""
        return abs(x2 - x1) * abs(y2 - y1)

    def calculate_overlap_area(box1, box2):
        """Calculate the overlapping area between two bounding boxes."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        return calculate_area(x_left, y_top, x_right, y_bottom)

    box1, box2 = boxes
    """Check if the overlapping area exceeds half the area of either bounding box."""
    overlap_area = calculate_overlap_area(box1, box2)
    box1_area = calculate_area(*box1)
    box2_area = calculate_area(*box2)
    
    # check whether it is large overlapping
    if overlap_area > threshold * box1_area or overlap_area > threshold * box2_area:
        if box1_area >  box2_area:
            return True
    return False

def get_edit_execute(edit_info, gen_objects, prompt, update_sg):
    operation = edit_info["operation"]
    if operation == "add":
        remove_objects = []
    elif operation == "remove":
        remove_objects = [edit_info["rm_object"]]
    elif operation == "replace":
        remove_objects = [edit_info["replace_src_object"]]
    elif operation == "edit_edge":
        remove_objects = gen_objects
    
    name2token = dict(zip(update_sg["objects"], update_sg["alias"]))
    name2desc = dict(zip(update_sg["objects"], update_sg["desc"]))
    name2bbox = dict(zip(update_sg["objects"], update_sg["bboxes"]))

    gen_desc = [name2desc[obj] for obj in gen_objects]
    gen_desc = [desc.split(".")[1] if desc.startswith("A photo of a <asset0>") else desc for desc in gen_desc]
    gen_bboxes = [name2bbox[obj] for obj in gen_objects]
    gen_bboxes = [xywh2xyxy(box) for box in gen_bboxes]

    if operation == "edit_edge":
        for obj in gen_objects:
            prompt = prompt.replace(f'\"{obj}\"', name2token[obj])        
        gen_objects = [name2token[obj] for obj in gen_objects]
    else:
        prompt = prompt.replace('\"',"")

    edit_execute = {
        "remove_objects": remove_objects,
        "gen_objects": gen_objects,
        "gen_bboxes": gen_bboxes,
        "gen_desc": gen_desc,
        "prompt": prompt,
        "info": edit_info,
        "modified_sg": {"objects": update_sg["objects"], "tuples": update_sg["tuples"]}
    }

    # handle overlapping (make sure the large object not occlude small object)
    if len(gen_objects) == 2 and need_reverse(gen_bboxes):
        edit_execute["reverse"] = True
    
    return edit_execute

def construct_edits(sg_path, mod_txt, out_dir):
    with open(sg_path, 'r') as f:
        sg_dict = json.load(f)

    with open(mod_txt, "r") as f:
        modis = f.read().splitlines()
        
    for mod_num, mod in enumerate(modis):
        edit_info = extract_edit(mod)
        operation = edit_info["operation"]
        if operation != "remove":
            # Update box
            question = prepare_box_question(operation, mod, sg_dict)
            box_proposal_chats = Chat()
            response = box_proposal_chats.ask_GPT(question, show_chats=True)
            box_updates = json.loads(box_proposal_chats.ask_GPT(TO_JSON))

            # Update scene graph
            update_sg = update_scene_graph(copy.deepcopy(sg_dict), operation, edit_info, box_updates)

            # Determine generate objects: (For 'add object' or 'modify relationship', it might introduce new interaction, generate both objects in this new relationship)
            gen_objects = list(box_updates.keys())
            if operation in ["add", "edit_edge"]:
                new_relation = edit_info["add_tuple"] if operation == "add" else edit_info["tgt_rel"]
                question = IS_INTERACTION + f"({','.join(new_relation)})"
                relation_type = Chat().ask_GPT(question)[len("Output: "):]
                gen_objects = [new_relation[0], new_relation[1]] if relation_type == "interaction" else gen_objects

            # Obtain generate prompt
            question = prepare_prompt_question(gen_objects, edit_info, update_sg["tuples"])
            prompt = Chat().ask_GPT(question, show_chats=True)

            edit_execute = get_edit_execute(edit_info, gen_objects, prompt, update_sg)
        else:
            update_sg = update_scene_graph(copy.deepcopy(sg_dict), operation, edit_info, None)
            edit_execute = get_edit_execute(edit_info, [], "", update_sg)
        
        with open(f"{out_dir}/{mod_num}.json", "w") as f:
            json.dump(edit_execute, f, indent=4)