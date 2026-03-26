# VReST
[ACL 2025] VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

## Prepare Dataset
You need to download the official codes and data of [MATH-V](https://github.com/mathllm/MATH-V), [VStar](https://huggingface.co/datasets/craigwu/vstar_benchhttps://github.com/lupantech/MathVista), [MMMU-Pro](https://huggingface.co/datasets/MMMU/MMMU_Pro), etc... Then organize them in the following format:
```
VReST
    ├── data
        ├── VStar
        ├── MATH-V
        ├── MMMU-Pro
```

## Start Inference

First of all, you need to deploy an LVLM using vllm.
```
vllm serve --port 8000 Qwen/Qwen3-VL-4B-Instruct --served-model-name Qwen3-VL-4B-Instruct --gpu_memory_utilization 0.8 --max_model_len 8192
```

Then, run VReST using the following script. Then the generated result will be saved in the path "./outputs/data={data.name}/llm={llm.name}/prompt_method={prompt_method.name}"
```
python inference.py data=mathvision mode=generate prompt_method=mctsv8
```

Then you need to deploy an LLM to extract the answers. 
```
vllm serve --port 8001 Qwen/Qwen2.5-7B-Instruct --served-model-name Qwen2.5-7B-Instruct --max_model_len 8192
```

Use the following script to extract the answer.
```
python inference.py data=mathvision mode=extract prompt_method=mctsv8 extract_field=max_mean_terminal extract_method=yes_or_no
```

Finally, use the following script to calculate the accuracy.
```
python inference.py data=mathvision mode=scoring prompt_method=mctsv8 extract_field=max_mean_terminal extract_method=yes_or_no
```

## Citation
```
@article{zhang2025vrest,
  title={VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism},
  author={Zhang, Congzhi and Peng, Jiawei and Wang, Zhenglin and Lai, Yilong and Sun, Haowen and Chang, Heng and Ma, Fei and Yu, Weijiang},
  journal={arXiv preprint arXiv:2506.08691},
  year={2025}
}
```

