from prompts.stage_n.search_space import SearchSpace, flatten_search_space
import json


# Count the top-k stages for each sample stored
# 
def analyze_search_space(list_search_space: list[SearchSpace], logger, config):
    stage_count = {}
    sequence_count = {}
    for i, search_space in enumerate(list_search_space):
        all_chains = flatten_search_space(search_space.root)
        topk_chains = sorted(all_chains, key=lambda x: x['reward'], reverse=True)[:config["inference"]["topk"]]
        logger.info(f"Sample {i} has {len(topk_chains)} top-k chains")
        for j, chain in enumerate(topk_chains):
            logger.info(f"  -> Chain {j} has stages: {chain['stage_seq']} with reward: {chain['reward']}")

            topk_stages = chain['stage_seq']
            for stage in topk_stages:

                if chain['reward'] < 0.5:
                    continue

                if stage not in stage_count:
                    stage_count[stage] = 0

                stage_count[stage] += 1
            
            sequence = " ".join(chain['stage_seq'])
            if sequence not in sequence_count:
                sequence_count[sequence] = 0
            sequence_count[sequence] += 1

    sorted_stage_count = sorted(stage_count.items(), key=lambda x: x[1], reverse=True)
    sorted_sequence_count = sorted(sequence_count.items(), key=lambda x: x[1], reverse=True)

    # logger.info(json.dumps({stage_count}, ensure_ascii=False, indent=4))
    for stage, count in sorted_stage_count:
        logger.info(f"Stage {stage} has {count} samples")

    for sequence, count in sorted_sequence_count:
        logger.info(f"Sequence {sequence} has {count} samples")

    return sorted_stage_count, sorted_sequence_count
