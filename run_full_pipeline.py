import json
import argparse
import os
import torch
# torch.cuda.set_device(0)

# Inputs
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--modification_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--remove_blend_step", type=int, default=15)
    parser.add_argument("--insert_mtSampler_step", type=int, default=10)
    parser.add_argument("--insert_attnMod_step", type=int, default=15)
    parser.add_argument("--num_samples", type=int, default=5)
    args = parser.parse_args()
    return args
args = parse_args()

image_path = args.image_path
modification_path = args.modification_path
out_dir = args.out_dir

# Outputs
scene_graph_dir = os.path.join(out_dir, "scene_graph")
mask_output_dir = os.path.join(out_dir, "preprocessed")
model_save_dir = os.path.join(out_dir, "save_model")
llm_edit_dir = os.path.join(out_dir, "edit_jsons")
edit_image_dir = os.path.join(out_dir, "edit_images")
for dir in [scene_graph_dir, mask_output_dir, model_save_dir, llm_edit_dir, edit_image_dir]:
    os.makedirs(dir,exist_ok=True)
basic_scene_graph_path = os.path.join(scene_graph_dir, "sg_dict.json")
update_scene_graph_path = os.path.join(scene_graph_dir, "sg_dict_updated.json")

# 1. Scene Graph Construction: 
from components.query_scene_graph import construct_basic_SG
sg_dict = construct_basic_SG(image_path, show_chats=True)
with open(basic_scene_graph_path, 'w') as f:
    json.dump(sg_dict, f, indent=4)


# # 2. Node Attribute Extraction:  
# # a) construct masks
# from utils.Segment.seg_utils import construct_node_masks
# construct_node_masks(image_path, basic_scene_graph_path, mask_output_dir)

# # b) construct descs
# from components.query_scene_graph import construct_node_desc
# annotated_sg_dict = construct_node_desc(image_path, basic_scene_graph_path)
# with open(update_scene_graph_path, 'w') as f:
#     json.dump(annotated_sg_dict, f, indent=4)

# # c) concept learning
# from components.train_model import concept_learning
# concept_learning(mask_output_dir, update_scene_graph_path, model_save_dir)


# # 3. Editing Control:
# from components.query_edits import construct_edits
# construct_edits(update_scene_graph_path, modification_path, llm_edit_dir)


# # 4. Attention-modulated Object Removal and Insertion
# from components.run_edits import execute_edits
# execute_edits(args, model_save_dir, mask_output_dir, llm_edit_dir, edit_image_dir)
