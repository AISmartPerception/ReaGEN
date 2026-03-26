import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
from utils.vector_cot import encode_stages
from utils.search_space_analysis import analyze_search_space
from prompts.stage_n.search_space import flatten_search_space

def evo_collate_fn(batch):
    """
    Pads variable-length tensors within a batch.
    """
    idx, image_embs, question_embs, cot_inits, As, cot_finals = zip(*batch)

    # Pad image/question embeddings
    def pad_seq(tensors):
        max_len = max(t.shape[0] for t in tensors)
        dim = tensors[0].shape[1]
        padded = torch.zeros(len(tensors), max_len, dim)
        mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
        for i, t in enumerate(tensors):
            l = t.shape[0]
            padded[i, :l, :] = t
            mask[i, :l] = 1
        return padded, mask

    image_padded, image_mask = pad_seq(image_embs)
    question_padded, question_mask = pad_seq(question_embs)

    idx = torch.stack(idx)
    cot_inits = torch.stack(cot_inits)
    As = torch.stack(As)
    cot_finals = torch.stack(cot_finals)
    
    return {
        "idx": idx,
        "image_emb": image_padded,
        "image_mask": image_mask,
        "question_emb": question_padded,
        "question_mask": question_mask,
        "cot_initial": cot_inits,
        "A": As,
        "cot_final": cot_finals,
    }

        
def evo_collate_fn_test(batch):
    """
    Pads variable-length tensors within a batch.
    """
    
    idx, cot_inits, cot_finals, scores = zip(*batch)
    return {
        "idx": idx,
        "cot_initial": cot_inits,
        "cot_final": cot_finals,
        "score": scores,
    }
    
def get_gen_dataloader(evo_ds, config, logger = None, test=False):
    if not test:
        dataset = GENDataset(evo_ds, config, logger)
        # val_idx = GENDataset()
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.9, 0.1])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["gen_training"]["batch_size"],
            shuffle=config["gen_training"]["shuffle"],
            num_workers=config["gen_training"]["num_workers"],
            collate_fn=evo_collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["gen_training"]["batch_size"],
            shuffle=config["gen_training"]["shuffle"],
            num_workers=config["gen_training"]["num_workers"],
            collate_fn=evo_collate_fn,
            pin_memory=True
        )
        return train_loader, val_loader, dataset.stage_pool
    
    else:
        test_dataset = GENDataset(evo_ds, config, logger, test)
        test_loader = DataLoader(
            test_dataset,
            # batch_size=config["gen_training"]["batch_size"],
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=evo_collate_fn_test,
            pin_memory=False
        )

        # test_ds = [evo_ds['train'][i.item()] for i in test_dataset.test_ids]
        # test_ds = evo_ds['train']
        return test_loader, test_dataset, test_dataset.test_ids, test_dataset.stage_pool


# Construct the data for GEN model
# Each sample is a tuple of (Image_emb+Question_emb+Initial CoT vector+Initial_A+Initial_a_to_Final, final CoT vector)
class GENDataset(Dataset):
    def __init__(self, evo_ds, config, logger=None, test=False):
        self.samples = []
        self.config = config
        self.test_ids = {}
        self.logger = logger
        self.test = test
        
        self.search_spaces = {}
        self.stage_pool = None

        
        # evo_pair_dir = os.path.join(self.config["paths"]["output_dir"], dataset_name)
        # evo_pair_dir = os.path.join(config["paths"]["output_dir"], f"{config['dataset']['data_id'].split('/')[-1]}_cot_init_attn")
        prefix = "cot_init_attn"

        setup = config['gen_training']['setup']
        # save_dir = os.path.join(self.config["paths"]["data_dir"], "gen_samples.pkl" if self.config["gen_training"]["full_data"] else "gen_samples_partial.pkl"
        
        if self.stage_pool is None:
            _, self.stage_pool = encode_stages(self.config['inference']['stages_pool'])
                            
        
        if test:
            dataset_name = config["dataset"]["data_id"].split("/")[-1]
            save_dir = os.path.join(self.config["paths"]["data_dir"], f"gen_dataloader_{dataset_name}_setup{setup}.pkl")
            if self.config["dataset"]["vison_only"]:
                evo_pair_dir = os.path.join(config["paths"]["output_dir"], f"{config['dataset']['data_id'].split('/')[-1]}_vision_cot_init_attn")
            else:
                evo_pair_dir = os.path.join(config["paths"]["output_dir"], f"{config['dataset']['data_id'].split('/')[-1]}_cot_init_attn")
            # if not self.config["gen_training"]["full_data"]:
            #     # with open(self.config["paths"]["test_ids_file"], "rb") as f:
            #     #     self.test_ids = pickle.load(f)
            #     random_ids = random.sample(range(len(os.listdir(evo_pair_dir))), config["gen_training"]["test_length"])
            #     selected_ids = [i for i in random_ids]
            #     print("Loading partial test evo pairs with %d ids" % len(selected_ids))
            # else:
            #     print("Loading full test evo pairs with %d ids" % len(os.listdir(evo_pair_dir)))
            for i, sample in enumerate(evo_ds):
                if os.path.exists(os.path.join(evo_pair_dir, f"{prefix}_{sample['id']}.pkl")):
                    # selected_ids.append(sample["id"])
                    # self.test_ids.append(int(sample["id"]))
                    if config['dataset']['data_id'] == "MMMU/MMMU_Pro" or config['dataset']['data_id'] == "MMMU/MMMU":
                        self.test_ids[sample["id"]] = i
                    else:
                        self.test_ids[int(sample["id"])] = i
            
            if logger is not None:
                logger.info("Testing with %d samples" % len(self.test_ids.keys()))
        

            searched_scores, direct_scores = [], []
            for i, evo_pair_file in tqdm(enumerate(os.listdir(evo_pair_dir)), total=len(os.listdir(evo_pair_dir)), desc="Loading evo pairs"):
                if dataset_name == "MMMU_Pro" or dataset_name == "MMMU":
                    sample_id = evo_pair_file.split("cot_init_attn_")[-1].split(".")[0]
                else:
                    sample_id = int(evo_pair_file.split("_")[-1].split(".")[0])
                    
                if sample_id not in list(self.test_ids.keys()):
                    continue
                
                evo_pair_path = os.path.join(evo_pair_dir, evo_pair_file)
                with open(evo_pair_path, "rb") as f:
                    evo_pair = pickle.load(f, encoding="latin1")
                    
                if sample_id not in self.search_spaces:
                    self.search_spaces[sample_id] = evo_pair["search_space"]

                all_chains = flatten_search_space(self.search_spaces[sample_id].root)
                topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]
                
                init_cots, final_cots = [], []
                for chain in topk_chains:
                    if chain['reward'] < 0.5:
                        continue
                    # print(chain['stage_seq'])
                    final_cot, _ = encode_stages(chain['stage_seq'], stage_pool=self.stage_pool) # [#_stages]
                    final_cots.append(final_cot)
                

                len_stage_pool = len(self.config['inference']['stages_pool'])

                for init_cot_str, init_cot_dict in evo_pair["init"].items():

                    init_cot = list(init_cot_str.split(" "))
                    # target_cot = chain['stage_seq']
                    
                    A = torch.tensor(init_cot_dict["A"]) # [#_stages, #_stages]
                    a_to_final = torch.tensor(init_cot_dict["a_to_final"]) # [#_stages]
                    if A.shape[0] < len_stage_pool:
                        # Extend A matrix to match final CoT length
                        padding_size = len_stage_pool - A.shape[0]
                        A = torch.cat([A, torch.zeros(padding_size, A.shape[1])], dim=0)
                        A = torch.cat([A, torch.zeros(A.shape[0], padding_size)], dim=1)

                        a_to_final = torch.cat([a_to_final, torch.zeros(padding_size)])
                        
                    A = torch.cat((A, a_to_final.unsqueeze(1)), dim=1) # [#_stages, #_stages+1]
                    

                    init_cot, _ = encode_stages(init_cot, stage_pool=self.stage_pool) # [#_stages]  
                    init_cots.append(init_cot)
                    # final_cot, _ = encode_stages(target_cot, stage_pool=self.stage_pool) # [#_stages]

                    image_emb = torch.tensor(init_cot_dict["image_emb"][0]).squeeze(0) # [#_seq_tokens, E]
                    question_emb = torch.tensor(init_cot_dict["question_emb"]).squeeze(0) # ([#_image_batches, E], [#_ROI_batches, E])
                 
                if dataset_name != "MMMU_Pro" and not self.config["dataset"]["vison_only"]:
                    self.samples.append({
                        "idx": sample_id,
                        "cot_initial": init_cots,
                        "cot_final": final_cots,
                        # "score": evo_pair["score"],
                        "score": 0,
                    })
                    # searched_scores.append(evo_pair["score"])
                    searched_scores.append(0)
                else:
                    self.samples.append({
                        "idx": sample_id,
                        "cot_initial": init_cots,
                        "cot_final": final_cots,
                        "score": 0,
                    })
                    searched_scores.append(0)
                
                direct_scores.append(evo_pair["direct_score"])
            
            if logger is not None:
                logger.info("Testing with %d samples" % len(self.test_ids.keys()))
                logger.info("Samples added: %d" % len(self.samples))
                logger.info("Searched mean score: %s" % str(sum(searched_scores) / len(searched_scores)))
                logger.info("Direct mean score: %s" % str(sum(direct_scores) / len(direct_scores)))
                    
        else:        
            # if os.path.exists(save_dir):
            if False:
                with open(save_dir, "rb") as f:
                    self.samples = pickle.load(f, encoding="latin1")
                # self.analyze_samples()
                print("Loaded %d samples from %s" % (len(self.samples), save_dir))
            else:
                sample_id = 0
                for i, ds in enumerate(evo_ds):
                    evo_pair_dir = os.path.join(config["paths"]["output_dir"], f"{config['training_data']['data_id'][i].split('/')[-1]}_cot_init_attn")
                    # for evo_pair_file in tqdm(os.listdir(evo_pair_dir), desc="Loading evo pairs"):
                    for sample in tqdm(ds, desc="Loading evo pairs"):
                        evo_pair_file = f"{prefix}_{sample['id']}.pkl"
                        evo_pair_path = os.path.join(evo_pair_dir, evo_pair_file)
                        
                        # idx = torch.tensor(int(evo_pair_file.split("_")[-1].split(".")[0]))
                        # idx = torch.tensor(int(sample['id']))
                        idx = torch.tensor(sample_id)
                        sample_id += 1
                        
                        if not os.path.exists(evo_pair_path):
                            continue
                        
                        with open(evo_pair_path, "rb") as f:
                            evo_pair = pickle.load(f, encoding="latin1")

                        if idx not in self.search_spaces:
                            self.search_spaces[idx] = evo_pair["search_space"]

                        all_chains = flatten_search_space(self.search_spaces[idx].root)
                        topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]

                        len_stage_pool = len(self.config['inference']['stages_pool'])

                        for init_cot_str, init_cot_dict in evo_pair["init"].items():
                            for chain in topk_chains:
                                
                                if chain['reward'] < 0.5:
                                    continue
                                
                                init_cot = list(init_cot_str.split(" "))
                                target_cot = chain['stage_seq']
                                
                                A = torch.tensor(init_cot_dict["A"]) # [#_stages, #_stages]
                                a_to_final = torch.tensor(init_cot_dict["a_to_final"]) # [#_stages]
                                if A.shape[0] < len_stage_pool:
                                    # Extend A matrix to match final CoT length
                                    padding_size = len_stage_pool - A.shape[0]
                                    A = torch.cat([A, torch.zeros(padding_size, A.shape[1])], dim=0)
                                    A = torch.cat([A, torch.zeros(A.shape[0], padding_size)], dim=1)

                                    a_to_final = torch.cat([a_to_final, torch.zeros(padding_size)])
                                    
                                A = torch.cat((A, a_to_final.unsqueeze(1)), dim=1) # [#_stages, #_stages+1]
                                
                                

                                init_cot, _ = encode_stages(init_cot, stage_pool=self.stage_pool) # [#_stages]  
                                final_cot, _ = encode_stages(target_cot, stage_pool=self.stage_pool) # [#_stages]
                                if len(final_cot)  == 0:
                                    print("what the fuck")


                                image_emb = torch.tensor(init_cot_dict["image_emb"][0]).squeeze(0) # [#_seq_tokens, E]
                                question_emb = torch.tensor(init_cot_dict["question_emb"]).squeeze(0) # ([#_image_batches, E], [#_ROI_batches, E])
                                    
                                self.samples.append({
                                    "idx": idx,
                                    "image_emb": image_emb,
                                    "question_emb": question_emb,
                                    "cot_initial": init_cot,
                                    "A": A,
                                    "cot_final": final_cot,
                                })
                    
                    print("Loaded %d samples from %s" % (len(self.samples), evo_pair_dir))
                
                # data_analysis = self.analyze_sample_cots(logger, config)
                # self.analyze_samples()
                    

                # sorted_stage_count, sorted_sequence_count = analyze_search_space(list_search_space, logger, config)

                

                '''
                    if not self.config["gen_training"]["full_data"]:
                        if len(evo_pair["iteration"]) == 1 or len(evo_pair["iteration"]) == self.config["inference"]["iterations"]:
                            continue

                    cot_initial, stage_pool = encode_stages(evo_pair["initial_stages"]) # [#_stages]
                    cot_final, stage_pool = encode_stages(evo_pair["final_stages"], stage_pool=stage_pool) # [#_stages]
                    if config["gen_training"]["setup"] == 2:
                        initial_iter = evo_pair["iteration"][0]
                      
                        # Adjacency and Influence Info
                        A = torch.tensor(initial_iter["A"]) # [#_stages, #_stages]
                        a_to_final = torch.tensor(initial_iter["a_to_final"]) # [#_stages]
                        A = torch.cat((A, a_to_final.unsqueeze(1)), dim=1) # [#_stages, #_stages+1]
                        
                        image_emb = torch.tensor(initial_iter["image_emb"][0]).squeeze(0) # [#_seq_tokens, E]
                        question_emb = torch.tensor(initial_iter["question_emb"]).squeeze(0) # ([#_image_batches, E], [#_ROI_batches, E])
                            
                        self.samples.append({
                            "idx": idx,
                            "image_emb": image_emb,
                            "question_emb": question_emb,
                            "cot_initial": cot_initial,
                            "A": A,
                            "cot_final": cot_final,
                        })

                    elif config["gen_training"]["setup"] == 3 or config["gen_training"]["setup"] == 4:
                        for iter in evo_pair["iteration"]:
                            iteration = evo_pair["iteration"][iter]

                            current_cot = iteration['stages']
                            
                            A = torch.tensor(iteration["A"]) # [#_stages, #_stages]
                            a_to_final = torch.tensor(iteration["a_to_final"]) # [#_stages]
                            if A.shape[0] < len(cot_initial):
                                # Extend A matrix to match final CoT length
                                padding_size = len(cot_initial) - A.shape[0]
                                A = torch.cat([A, torch.zeros(padding_size, A.shape[1])], dim=0)
                                A = torch.cat([A, torch.zeros(A.shape[0], padding_size)], dim=1)

                                a_to_final = torch.cat([a_to_final, torch.zeros(padding_size)])
                                
                             
                            A = torch.cat((A, a_to_final.unsqueeze(1)), dim=1) # [#_stages, #_stages+1]
                            if A.shape != (7, 8):
                                logger.info("A shape: %s" % str(A.shape))
                            
                            current_cot, stage_pool = encode_stages(current_cot, stage_pool=stage_pool) # [#_stages]
                            image_emb = torch.tensor(iteration["image_emb"][0]).squeeze(0) # [#_seq_tokens, E]
                            question_emb = torch.tensor(iteration["question_emb"]).squeeze(0) # ([#_image_batches, E], [#_ROI_batches, E])
                                
                            self.samples.append({
                                "idx": idx,
                                "image_emb": image_emb,
                                "question_emb": question_emb,
                                "cot_initial": current_cot,
                                "A": A,
                                "cot_final": cot_final,
                            })
                            '''
                    
                # with open(save_dir, "wb") as f:
                #     pickle.dump(self.samples, f)
            
                # print("Loaded %d samples and saved to %s" % (len(self.samples), save_dir))

    def __len__(self):
        return len(self.samples)

    def analyze_sample_cots(self, logger, config):
        stage_count = {}
        for sample in self.samples:
            str_cot = ""
            for i in sample["cot_final"].tolist():
                if i < 7:
                    str_cot += str(i) + ","
                    
        return stage_count
    
    
    def __getitem__(self, idx): 
        item = self.samples[idx]
        if self.test:
            return (item["idx"], item["cot_initial"], item["cot_final"], item["score"])
        else:
            return (item["idx"], item["image_emb"], item["question_emb"], item["cot_initial"], item["A"], item["cot_final"])


    def analyze_samples(self):
        stage_count = {}
        for sample in self.samples:
            str_cot = ""
            for i in sample["cot_final"].tolist():
                if i < 14:
                    str_cot += str(i) + ","

            if str_cot not in stage_count:
                stage_count[str_cot] = 0
            stage_count[str_cot] += 1
            
            # if self.logger is not None:
                # self.logger.info("Sample %s: %s" % (sample["idx"], str_cot))
                

        if self.logger is not None:
            for cot, count in stage_count.items():
                self.logger.info("Stage %s: %d" % (cot, count))

