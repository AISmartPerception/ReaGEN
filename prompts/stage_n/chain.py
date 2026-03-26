from typing import Dict, Any
import logging
from prompts.stage_n.stage import Stage
from prompts.stage_n.blackboard import blackboard_template
from utils.bboxes_tok import _maybe_to_pil, _pad_box
from PIL import Image
import json

import numpy as np

def layerwise_importance(stage_results, final_key="ANSWER.CONSOLIDATION", normalize=True, stage_order=None, logger=None, decay=0.6):
    """
    Build A_{i,j} and A_{i,F} from dict-style attention logs and compute stage importance.

    stage_results:
      {
        "SCENE.SUMMARY": {"mean_attention_mass": {}},
        "BBOX":          {"mean_attention_mass": {"SCENE.SUMMARY": 0.0068}},
        "COUNT":         {"mean_attention_mass": {"SCENE.SUMMARY": 0.0117, "BBOX": 0.0238}},
        "FINAL":         {"mean_attention_mass": {"SCENE.SUMMARY": x1, "BBOX": x2, "COUNT": x3}}
      }

    Returns:
      A_to_stage  : (N,N) upper-tri matrix with A[i,j] (i<j else 0)
      a_to_final  : (N,)   vector with A[i,F]
      importance  : (N,)   vector with Importance(i)
    """
    # ---- Stage order (exclude FINAL) ----
    if stage_order is None:
        stages = [k for k in stage_results.keys() if k != final_key and k != "DIRECT_ANSWER"]
    else:
        stages = list(stage_order)
    N = len(stages)
    idx = {name: i for i, name in enumerate(stages)}

    # ---- Build A_to_stage from dicts ----
    A_to_stage = np.zeros((N, N), dtype=float)
    for j, stage in enumerate(stages):
        incoming = stage_results[stage].get("mean_attention_mass", {}) or {}
        # logger.info(f"incoming: {incoming}")
        mem_dict = {stage_name: 0 for stage_name in incoming[0].keys()}
        for layer_name, layer_dict in incoming.items():
            for stage_name, val in layer_dict.items():
                mem_dict[stage_name] += val
        mem_dict = {stage_name: mem_dict[stage_name] / 4 for stage_name in mem_dict.keys()}

        incoming = mem_dict
        # print(incoming)
        # incoming is a dict: {prev_stage_name: mass}
        for prev_name, val in incoming.items():
            if prev_name not in idx:
                continue  # ignore unknown keys
            i = idx[prev_name]
            if i < j:                     # only earlier stages contribute
                A_to_stage[i, j] = float(val)

    # ---- Build a_to_final from dict or list ----
    fin = stage_results[final_key]["mean_attention_mass"]

    a_to_final = np.zeros(N, dtype=float)
    if isinstance(fin, dict):
        for layer_name, layer_dict in fin.items():
            if int(layer_name) < 4:
                for name, val in layer_dict.items():
                    if name in idx:
                        a_to_final[idx[name]] += float(val)
    else:
        # fallback if it's a list aligned with stages
        arr = list(fin)
        if len(arr) != N:
            raise ValueError(f"FINAL expects {N} masses, got {len(arr)}")
        a_to_final = np.array(arr, dtype=float)

    # logger.info(f"A_to_stage: {A_to_stage}")
    # logger.info(f"a_to_final: {a_to_final}")
    # ---- Optional: normalize incoming per destination j ----
    # if normalize:
    #     for j in range(1, N):
    #         s = A_to_stage[:j, j].sum()
    #         if s > 0:
    #             A_to_stage[:j, j] /= s
    if normalize:
        for i in range(N):
            s = A_to_stage[i, :].sum()
            if s > 0:
                A_to_stage[i, :] /= s


    # ---- Backward recursion: Imp(i) = A_{i,F} + sum_{j>i} A_{i,j} * Imp(j) ----
    importance = np.zeros(N, dtype=float)
    direct_contribution = np.zeros(N, dtype=float)
    importance[N-1] = a_to_final[N-1]  # last stage only flows to final
    for i in range(N-2, -1, -1):
        # contribution = direct flow to final + indirect flow via later stages
        importance[i] = a_to_final[i] + float((A_to_stage[i, i+1:] * (decay * importance[i+1:])).sum())


    importance_dict = {}
    for i, stage_name in enumerate(stages):
        importance_dict[stage_name] = {
            "direct": float(a_to_final[i]),
            "indirect": float(importance[i] - a_to_final[i]),
            "total": float(importance[i]),
        }

    return A_to_stage, a_to_final, importance, importance_dict


class Chain:
    def __init__(self, 
                stages: list, 
                logger: logging.Logger | None = None, 
                model: Any = None,
                processor: Any = None,
                config: dict = None,
                base_system_prompts: dict = None):

        self.stages = [Stage(stage, model, processor, config, base_system_prompts[stage]) for stage in stages]
        self.final_stage = Stage("ANSWER.CONSOLIDATION", model, processor, config, base_system_prompts["ANSWER.CONSOLIDATION"])
        if "DIRECT_ANSWER" in base_system_prompts:
            self.direct_answer_stage = Stage("DIRECT_ANSWER", model, processor, config, base_system_prompts["DIRECT_ANSWER"])
        else:
            self.direct_answer_stage = None

        # Initialize shared blackboard as a dictionary with templates
        self.shared_blackboard = {stage.name: blackboard_template[stage.name] for stage in self.stages}
        self.logger = logger
        self.config = config
        self.cropped_image = None  # Store cropped image after BBOX stage
        self.bbox_coords = None    # Store bbox coordinates
        self.dataset_name = config["dataset"]["data_id"].split("/")[-1]


    def blackboard_to_text(self, current_stage_idx: int) -> str:
        """
        Convert the current blackboard state to a formatted text string.
        This includes all previous stage outputs filled into their templates.
        """
        blackboard_text = ""
        mem_names = []
        for i, (stage_name, content) in enumerate(self.shared_blackboard.items()):
            if i == current_stage_idx:
                break

            if content.strip():  # Only include stages that have content
                blackboard_text += f"<MEM:{stage_name}>\n{content}\n </MEM:{stage_name}>\n"
                mem_names.append(stage_name)
        return blackboard_text, mem_names


    def extract_bbox_and_crop(self, image: Image.Image, bbox_coords: tuple) -> tuple[Image.Image, tuple]:
        """
        Extract bbox coordinates from raw text and crop theq image.
        Returns the cropped image and bbox coordinates.
        """
        x1, y1, x2, y2 = map(int, bbox_coords)
        cropped = image.crop((x1, y1, x2, y2))
            
        return cropped, (x1, y1, x2, y2)
        


    def run(self, sample_id, sample: Dict[str, Any], return_aux: bool = True, return_debug: bool = True):
        """
        Run the chain of stages, updating the shared blackboard with rendered outputs from each stage.
        After BBOX stage, subsequent stages receive both original and cropped images.
        """
        stage_results = {}
        stage_uncertainties = {}  # Track uncertainty for each stage
        if self.dataset_name == "MMMU_Pro":
            if "vision" in sample['config_name']:
                original_image = _maybe_to_pil(sample["image"])
                current_images = [original_image]
            else:
                num_images = 8
                original_images = []
                for i in range(num_images-1):
                    if sample[f"image_{i+1}"]:
                        original_images.append(_maybe_to_pil(sample[f"image_{i+1}"]))
                current_images = original_images
        else:
            original_image = _maybe_to_pil(sample["image"])
            current_images = [original_image]

        # original_image = _maybe_to_pil(sample["image"])
        # current_images = [original_image]  # Start with just original image
        
        # Run each stage and update the blackboard with rendered outputs
        for i, stage in enumerate(self.stages):
            # if self.logger:
            #     self.logger.info(f"Running stage: {stage.name}")
            
            blackboard_text, mem_names = self.blackboard_to_text(i)
            
            # Run the stage with current images and blackboard text
            stage_result = stage.run(sample_id, sample, blackboard_text, current_images, mem_names, logger=self.logger)
            stage_results[stage.name] = stage_result

            if stage_result["output"] == "Invalid JSON format":
                stage_uncertainties[stage.name] = 0.0
                self.shared_blackboard[stage.name] += stage_result["output"]
                self.shared_blackboard[stage.name] += stage_result["raw_text"]

                # if stage.name == "BBOX":
                #     self.bbox_coords = (0, 0, original_image.width, original_image.height)
                #     self.cropped_image = original_image
                #     current_images = [original_image]
            else:
                stage_uncertainties[stage.name] = stage_result.get("uncertainty_score", 0.0)
                self.shared_blackboard[stage.name] += stage_result["output"]
                
                # if stage.name == "BBOX":
                #     cropped_img, bbox_coords = self.extract_bbox_and_crop(original_image, stage_result["coords"])
                #     if cropped_img is not None and bbox_coords is not None:
                #         self.cropped_image = cropped_img
                #         self.bbox_coords = bbox_coords
                #         current_images = [original_image, cropped_img]
                
            if self.logger:
                self.logger.info(f"Stage {i+1}, {stage.name} output: {stage_result['output']}")
        

        final_blackboard_text, mem_names = self.blackboard_to_text(len(self.stages))

        final_result = self.final_stage.run(sample_id, sample, final_blackboard_text, current_images, mem_names, logger=self.logger)
        stage_results["ANSWER.CONSOLIDATION"] = final_result

        if self.direct_answer_stage is not None:
            direct_answer_result = self.direct_answer_stage.run(sample_id, sample, final_blackboard_text, current_images, mem_names, logger=self.logger)
            stage_results["DIRECT_ANSWER"] = direct_answer_result
        
        if len(self.stages) > 0:
            A, a_to_final, imp, impor_dict = layerwise_importance(stage_results, logger=self.logger)
            # A, a_to_final, imp, impor_dict = {}, {}, {}, {}
       
        # Track uncertainty for final stage
        stage_uncertainties["ANSWER.CONSOLIDATION"] = final_result.get("uncertainty_score", 0.0)

        # Compute overall uncertainty metrics
        all_uncertainties = list(stage_uncertainties.values())
        # mean_uncertainty = sum(all_uncertainties) / len(all_uncertainties) if all_uncertainties else 0.0
        
        out = {
            "answer_raw": final_result["raw_text"],
            "rendered_answer": final_result["output"],
            # "final_validity": final_result["final_validity"],
            "stage_outputs": stage_results,
            "stage_uncertainties": stage_uncertainties,
            "images": current_images,
            # "mean_uncertainty": mean_uncertainty,
            # "total_stages": len(stage_uncertainties),
            "gt": final_result["gt"],
        }

        if len(self.stages) > 0:
            out["importance_dict"] = impor_dict
            out["importance"] = imp
            out["A"] = A
            out["a_to_final"] = a_to_final
            out['shared_blackboard'] = self.shared_blackboard
            out["blackboard_text"] = final_blackboard_text

            out["question_emb"] = final_result["question_emb"]
            out["image_emb"] = final_result["image_emb"]
            

        # if "BBOX" in [stage.name for stage in self.stages]:
        #     out["bbox_coords"] = self.bbox_coords
        #     out["cropped_image"] = self.cropped_image

        if self.direct_answer_stage is not None:
            out["direct_answer_raw"] = direct_answer_result["raw_text"]
            out["direct_answer_rendered"] = direct_answer_result["output"]

        return out
    


