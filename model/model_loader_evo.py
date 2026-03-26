import os
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, AutoModelForSeq2SeqLM, AutoConfig, BitsAndBytesConfig
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoModelForVision2Seq
from transformers import pipeline
# from transformers.models.qwen2_vl_vpt import Qwen2VLVPTForConditionalGeneration
from dotenv import load_dotenv
from openai import OpenAI
# from google import genai
from transformers import AutoModel, AutoTokenizer
import httpx



def load_student_on_gpu(name: str, gpu_index: int = 0, use_flash: bool = True):
    kwargs = {}
    if use_flash:
        kwargs["attn_implementation"] = "flash_attention_2"
    # Map the entire model to a single local GPU index
    kwargs["device_map"] = {"": gpu_index}  # 0 or 1
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype="auto", **kwargs)
    proc  = AutoProcessor.from_pretrained(name, use_fast=True)
    return model, proc


def _maxmem_int_keys(gpu_gib_per_device: int, cpu_gib: int, num_gpus: int = 2):
    mm = {i: f"{gpu_gib_per_device}GiB" for i in range(num_gpus)}  # 0,1
    mm["cpu"] = f"{cpu_gib}GiB"
    return mm

def try_load_teacher_robust(
    name: str = "Qwen/Qwen2.5-VL-32B-Instruct",
    use_flash: bool = True,
    offload_dir: str = "./offload_teacher",
    gpu_cap_gib: int = 78,
    cpu_cap_gib: int = 900,
    allow_8bit: bool = True,
):
    os.makedirs(offload_dir, exist_ok=True)
    num_local = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_local < 2:
        print(f"[TeacherLoader] WARNING: only {num_local} GPU(s) visible. Expecting 2 (remapped 1,2).")

    attn_impl = {"attn_implementation": "flash_attention_2"} if use_flash else {}

    attempts = []

    # Attempt 1: 4-bit NF4 across GPUs 0,1 + CPU offload
    qconf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    attempts.append(dict(
        note="4bit NF4, device_map='auto' on GPUs 0,1 + CPU offload",
        kwargs=dict(
            quantization_config=qconf4,
            device_map="auto",
            max_memory=_maxmem_int_keys(gpu_cap_gib, cpu_cap_gib, num_gpus=max(1,num_local)),
            offload_folder=offload_dir,
            offload_state_dict=True,
            **attn_impl,
        )
    ))

    # Attempt 2: 8-bit across GPUs 0,1 + CPU offload
    if allow_8bit:
        qconf8 = BitsAndBytesConfig(load_in_8bit=True)
        attempts.append(dict(
            note="8bit, device_map='auto' on GPUs 0,1 + CPU offload",
            kwargs=dict(
                quantization_config=qconf8,
                device_map="auto",
                max_memory=_maxmem_int_keys(gpu_cap_gib, cpu_cap_gib, num_gpus=max(1,num_local)),
                offload_folder=offload_dir,
                offload_state_dict=True,
                **attn_impl,
            )
        ))

    # Attempt 3: CPU-only (last resort)
    attempts.append(dict(
        note="CPU-only fallback",
        kwargs=dict(
            device_map={"": "cpu"},
            low_cpu_mem_usage=True,
        )
    ))

    last_err = None
    for i, att in enumerate(attempts, 1):
        try:
            print(f"[TeacherLoader] Attempt {i}/{len(attempts)}: {att['note']}")
            k = {kk: vv for kk, vv in att["kwargs"].items() if vv is not None}
            teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, **k)
            tproc = AutoProcessor.from_pretrained(name, use_fast=True)
            return teacher, tproc, att["note"]
        except Exception as e:
            last_err = e
            print(f"[TeacherLoader] Attempt {i} failed: {type(e).__name__}: {e}")
            continue
    raise RuntimeError("Failed to load teacher") from last_err


def load_model(model_id, device=1, API = True, local_dir = None, 
                quant_conf = "16bit", use_flash = False,
                gpu_cap_gib = 78, cpu_cap_gib = 900):
    if API: 
        if "gpt" in model_id:
            load_dotenv("utils/open_key.env")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            print(f"Loaded API model {model_id}")
            return (model_id, client)
        elif "gemini" in model_id:
            load_dotenv("utils/gemini.env")
            # client = genai.Client(
            #     api_key=os.getenv("GEMINI_API_KEY"),
            #     # http_client=httpx.Client(verify=False)
            # )
            print(f"Loaded API model {model_id}")
            return (model_id, client)
    else:
        if "InternVL" in model_id:
            dtype = torch.bfloat16

            device = {"": device}
            model_name = "OpenGVLab/InternVL3_5-4B"
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=dtype,   # or torch.bfloat16 / torch.float16
                device_map=device,
            )
            model.eval()
            return (tokenizer, model)
            
        if not os.path.exists(local_dir):
            snapshot_download(repo_id=model_id, 
                                local_dir=local_dir,
                                local_dir_use_symlinks=False,
                                resume_download=True,
                                max_workers=1)
            print(f"Downloaded model {model_id} to {local_dir}")

        if torch.cuda.is_available():
            
            dtype = torch.bfloat16

            device = {"": device}
            

        if model_id == "Qwen/Qwen2.5-VL-7B-Instruct":
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_dir, dtype='auto', device_map=device, attn_implementation="eager"
            )
        elif model_id == "Qwen/Qwen2.5-VL-32B-Instruct":

            kwargs = {}
            if use_flash:
                kwargs["attn_implementation"] = "flash_attention_2"
            else:
                kwargs["attn_implementation"] = "eager"

            kwargs["device_map"] = {"": device}  # 0 or 1
            if quant_conf == "4bit":
                qconf = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                )
            elif quant_conf == "8bit":
                qconf8 = BitsAndBytesConfig(load_in_8bit=True)
            elif quant_conf == "16bit":
                qconf16 = BitsAndBytesConfig(load_in_16bit=True)
            
            kwargs["quantization_config"] = qconf
            kwargs["device_map"] = device
            kwargs["max_memory"] = _maxmem_int_keys(gpu_cap_gib, cpu_cap_gib, num_gpus=1)
              

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                local_dir, **kwargs
            )

        elif model_id == "Qwen/Qwen2-VL-2B-Instruct":
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                local_dir, dtype='auto', device_map=device
            )

        elif model_id == "Qwen/Qwen3-VL-8B-Instruct" or model_id == "Qwen/Qwen3-VL-4B-Instruct":
            kwargs = {}
            # if use_flash:
            #     kwargs["attn_implementation"] = "flash_attention_2"
            # else:
            #     kwargs["attn_implementation"] = "eager"
            
            # if quant_conf == "4bit":
            #     qconf = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_quant_type="nf4",
            #         bnb_4bit_use_double_quant=True,
            #         bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            #     )
            # elif quant_conf == "8bit":
            #     qconf = BitsAndBytesConfig(load_in_8bit=True)
            # elif quant_conf == "16bit":
            #     qconf = BitsAndBytesConfig(load_in_16bit=True)

            # kwargs["quantization_config"] = qconf
              
            model = AutoModelForVision2Seq.from_pretrained( 
                local_dir, dtype="auto", device_map=device, **kwargs
            )

        elif model_id == "Qwen/Qwen3-VL-30B-A3B-Thinking" or model_id == "Qwen/Qwen3-VL-30B-A3B-Instruct":
            kwargs = {}
            if use_flash:
                kwargs["attn_implementation"] = "flash_attention_2"
            else:
                kwargs["attn_implementation"] = "eager"

            kwargs["device_map"] = {"": device}  # 0 or 1
            if quant_conf == "4bit":
                qconf = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
                )
            elif quant_conf == "8bit":
                qconf = BitsAndBytesConfig(load_in_8bit=True)
            elif quant_conf == "16bit":
                qconf = BitsAndBytesConfig(load_in_16bit=True)

            kwargs["quantization_config"] = qconf
            kwargs["device_map"] = device
            kwargs["max_memory"] = _maxmem_int_keys(gpu_cap_gib, cpu_cap_gib, num_gpus=1)

            model = AutoModelForVision2Seq.from_pretrained( 
                local_dir, **kwargs
            )
            

        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_id,
        #     dtype="auto",
        #     device_map="auto",
        # )

        # model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype="auto")  

        processor = AutoProcessor.from_pretrained(local_dir, use_fast=True)
        print(f"Loaded model {model_id} from {local_dir}")

        return (processor, model)