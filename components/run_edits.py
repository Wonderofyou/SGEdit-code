import torch
import json
import os
from torchvision.utils import save_image
from PIL import Image
import numpy as np

from torchvision import transforms
from diffusers import DiffusionPipeline, DDIMScheduler
from utils.Segment.seg_utils import initialize_sam_model

from utils.Pipeline.diffuser_utils import SGPipeline
from utils.Attention.attn_utils import AttentionBase, MutualSelfAttentionControlMask, DenseMultiDiff
from utils.Attention.attn_processor import register_attention_control
from utils.basic_utils import load_torch_image, bbox2layouts, parse_text, region_prompts, dilate_mask, erase_on_image_level

class Generator():
    def __init__(self, model_path, sam_predictor, device, weight_dtype):
        self.weight_dtype = weight_dtype
        self.device = device
        self.pipeline = self._load_pipeline(model_path, device)
        self.sam_predictor = sam_predictor
    
    def _load_pipeline(self, model_path, device):
        pipeline = SGPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        pipeline.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        pipeline = pipeline.to(device)
        return pipeline

    def dense_forward(self, editor, prompt, regional_prompts, bg_latents, save_dir, seed,
                      bg_preserve_start=25, bg_preserve_end=45):
        register_attention_control(self.pipeline, editor)
        generator = torch.Generator(device='cuda').manual_seed(seed)
    
        image, latents_list, vis_image, fg_blend_mask = self.pipeline(
            editor=editor,
            prompt=prompt,
            regional_prompts=regional_prompts,
            generator=generator,
            sam_predictor=self.sam_predictor,
            bg_latents=bg_latents,
            bg_preserve_start=bg_preserve_start,
            bg_preserve_end=bg_preserve_end,
        )
        
        img = Image.fromarray(image)
        img.save(os.path.join(save_dir, f"{seed}.png"))
        
        # vis_image.save(os.path.join(save_dir, f"vis_{seed}.png"))
        # Image.fromarray(fg_blend_mask[0][0].cpu().numpy()).save(os.path.join(save_dir, f"mask_{seed}.png"))
        
        editor.reset()
        
    def erase_forward(self, editor, prompts, mask, start_code, latents_arr, save_dir, seed, blend_step):
        register_attention_control(self.pipeline, editor)
        
        image, intermediate_latents = self.pipeline.erase(prompts,
                latents=start_code,
                mask_s=mask,
                blend_step=blend_step,
                ref_intermediate_latents=latents_arr,
                guidance_scale=7.5,
                save_latents=True)
        torch.save(intermediate_latents, os.path.join(save_dir, f"erased_latents_{seed}.pt"))
        return image
    
    def reverse_forward(self, source_image, source_prompt):
        editor = AttentionBase()
        register_attention_control(self.pipeline, editor)

        start_code, latents_arr = self.pipeline.invert(source_image,
                                                source_prompt,
                                                guidance_scale=7.5,
                                                num_inference_steps=50,
                                                return_intermediates=True)
        return start_code, latents_arr
        
            
class SG_Editor():
    def __init__(self, model_path, input_dir):
        self.input_dir = input_dir
        self.input_image = Image.open(os.path.join(input_dir, "input.png"))
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.weight_dtype = torch.float16
        sam_predictor = initialize_sam_model("cuda")
        self.generator = Generator(model_path, sam_predictor, self.device, self.weight_dtype)
        self.bg_prompt = "A photo of nothing, no object, no people, just the background."
      
    def _update_gen_setting(self, gen_dict):
        self.prompt = gen_dict["prompt"]
        self.boxes = gen_dict["gen_bboxes"]
        self.insert_objects = gen_dict["gen_objects"]
        self.remove_objects = gen_dict["remove_objects"]
        self.gen_desc = gen_dict["gen_desc"]
        self.reverse = True if "reverse" in gen_dict else False  
        
    def _invert(self):
        # DDIM inversion
        source_prompt = ""
        image = np.array(self.input_image)
        image = load_torch_image(image).to(self.device, self.weight_dtype)

        _, latents_arr = self.generator.reverse_forward(image, source_prompt)
        torch.save(latents_arr[::-1], os.path.join(self.invert_dir, "invert_latents.pt"))
    
    def _remove(self, num_samples=5, blend_step=15, pre_erased=True):
        # set the prompt
        source_prompt = ""
        target_prompt = "A photo of nothing, no object, no people, just the background."
        prompts = [source_prompt, target_prompt]

        # dialate the erased mask
        mask_s = []
        for removed_name in self.remove_objects:
            mask_path = os.path.join(self.input_dir, f"{removed_name}_mask.png")
            mask = Image.open(mask_path)
            mask = dilate_mask(mask, dilate_kernel=10, target_type="numpy") > 0.5
            mask_s.append(mask)
        mask_s = ~np.stack(mask_s).any(axis=0)
        
        torch_transforms = transforms.ToTensor()
        
        # load/resize the image and mask
        image = np.array(self.input_image)
        image = erase_on_image_level(image, mask) if pre_erased == True else image
        image = load_torch_image(image).to(self.device, self.weight_dtype)
        mask_s = Image.fromarray(mask_s).resize((64,64))
        mask_s = torch_transforms(mask_s)[0].to(self.device, self.weight_dtype)
        
        # invert the erased image
        start_code, latents_arr = self.generator.reverse_forward(image, source_prompt)
        for i in range(num_samples):
            # erased on the latent level
            rand_noise = torch.randn_like(start_code)
            start_code_erase = torch.where((mask_s==0)[None, None, :, :], rand_noise, start_code) 
            start_code_erase = torch.cat([start_code, start_code_erase])
            
            # erase_forward
            editor = MutualSelfAttentionControlMask(mask_s=mask_s)
            image = self.generator.erase_forward(editor, 
                                         prompts=prompts,
                                         mask=mask_s,
                                         start_code=start_code_erase,
                                         latents_arr=latents_arr,
                                         save_dir=self.remove_dir,
                                         seed=i,
                                         blend_step=blend_step)
            
            save_path = os.path.join(self.out_dir, f"erased{i}.png")
            save_image(image[1:], save_path)
        return image
    
    def _add(self, num_samples=5, multi_step=10, attnMod_step=15):
        tokenizer = self.generator.pipeline.tokenizer
        
        # convert from bounding boxes to the layout
        layouts = bbox2layouts(self.boxes, self.reverse, self.insert_objects) 
        
        # multi-sampling condition 
        use_multi_sampler = multi_step > 0 and len(self.insert_objects) > 1
        if use_multi_sampler:
            assert attnMod_step > multi_step, "Error: 'attnMod_step' must be greater than 'multi_step'."
            regional_prompts, supr_idxs_list = region_prompts(tokenizer, self.gen_desc, self.insert_objects)
        else:
            regional_prompts, supr_idxs_list = [], []
        
        # add the bg_prompt and gloabl suppress to the end of list
        main_prompt, supr_idxs = parse_text(tokenizer, self.prompt, self.gen_desc, self.insert_objects)
        # print(main_prompt, supr_idxs)
        regional_prompts.append(self.bg_prompt)
        supr_idxs_list.append(supr_idxs)


        # initialize the attn control
        bg_latents = None
        for i in range(num_samples):
            # initialize editor
            editor = DenseMultiDiff(supr_idxs=supr_idxs_list, layouts=layouts, boxes=self.boxes, weight_dtype=self.weight_dtype, 
                                 use_multi_sampler=use_multi_sampler, multi_step=multi_step, attnMod_step=attnMod_step)
            
            # load bg_latents
            if len(self.remove_objects)==0:
                if bg_latents is None:
                    bg_latents = torch.load(os.path.join(self.invert_dir, "invert_latents.pt"))
            else:
                bg_latents = torch.load(os.path.join(self.remove_dir, f"erased_latents_{i}.pt"))
            
            self.generator.dense_forward(editor, 
                                         prompt = main_prompt, 
                                         regional_prompts = regional_prompts,
                                         bg_latents = bg_latents, 
                                         save_dir = self.out_dir,
                                         seed = i)

    def gen_union(self, args, gen_dict, out_dir):
        # make directories
        self.out_dir = out_dir
        self.invert_dir = os.path.join(out_dir, "inverted")
        self.remove_dir = os.path.join(out_dir, "removal")
        os.makedirs(self.out_dir, exist_ok = True)
        os.makedirs(self.invert_dir, exist_ok = True)
        os.makedirs(self.remove_dir, exist_ok = True)
        
        self._update_gen_setting(gen_dict)
        
        # Removal
        if len(self.remove_objects) == 0:
            self._invert()
        else:
            self._remove(num_samples=args.num_samples, 
                         blend_step=args.remove_blend_step)
        
        # Insertion
        if len(self.insert_objects) == 0:
            return
        else:
            self._add(num_samples=args.num_samples,
                      multi_step=args.insert_mtSampler_step,
                      attnMod_step=args.insert_attnMod_step)


def execute_edits(args, model_path, input_dir, llm_edit_dir, out_dir):
    sg_editor = SG_Editor(model_path=model_path, input_dir=input_dir)
    edit_jsons = [f"{llm_edit_dir}/{fp}" for fp in os.listdir(llm_edit_dir) if fp[-4:]=="json"]
    for file in edit_jsons:
        sub_dir = "edits_" + file.split("/")[-1].split(".")[0]
        sub_dir = os.path.join(out_dir, sub_dir)
        with open(file) as f:
            gen_dict = json.load(f)
        sg_editor.gen_union(args, gen_dict, sub_dir)
    
    
    
    