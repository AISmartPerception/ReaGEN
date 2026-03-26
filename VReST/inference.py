import os
import logging
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time
import json

import sys
sys.path.append("/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST")
sys.path.append("/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/data/MathVista")

print("Sys path:", sys.path)
from mllms.base_models import Model, Llama_Model, Qwen25_Model
from utils.generate import generate
from utils.extract_answer import extract
from utils.get_scores import scoring

# A logger for this file
logging.getLogger("httpx").setLevel(logging.WARNING) 
log = logging.getLogger(__name__)
@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg):

    # print the config
    log.info(f"Working directory : {os.getcwd()}")
    output_dir = HydraConfig.get().runtime.output_dir
    # 允许动态键
    OmegaConf.set_struct(cfg, False)
    # 添加新的键
    cfg.output_dir = output_dir
    extract_result_dir = os.path.join(output_dir, f"ext_llm={cfg.ext_llm.name}")
    extract_result_dir = os.path.join(extract_result_dir, f"method={cfg.extract_method}")
    if hasattr(cfg, "extract_field"):
        extract_result_dir = os.path.join(extract_result_dir, f"field={cfg.extract_field}")
    extract_result_dir = os.path.join(extract_result_dir, f"n={cfg.ext_llm.n}")
    cfg.extract_result_dir = extract_result_dir
    os.makedirs(extract_result_dir, exist_ok=True)
    # 重新启用结构化模式
    OmegaConf.set_struct(cfg, True)
    log.info(f"Output directory  : {output_dir}")
    log.info("\n" + OmegaConf.to_yaml(cfg))
    print("Configuration:", cfg)

    if cfg.mode == "generate":
        # Load the model
        model = Model(cfg)
        generate(cfg, model)
    
    if cfg.mode == "extract":
        # Load the model
        model = Qwen25_Model(cfg.ext_llm)
        extract(cfg, model)

    if cfg.mode == "scoring":
        scoring(cfg)

if __name__ == "__main__":
    main()