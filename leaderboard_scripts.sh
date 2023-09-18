# git clone -b py38_hf_leaderboard  https://github.com/aiomodel/lm-evaluation-harness.git
# mmlu: hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
# Example: python main.py --model=hf-causal --model_args="pretrained=<your_model>,use_accelerate=True,revision=<your_model_revision>" --tasks=<task_list> --num_fewshot=<n_few_shot> --batch_size=2 --output_path=<output_path>
# but hf-causal is not compatible with use_accelerate=True 
## OLLAMA-v1
python main.py \
    --model hf-causal \
    --model_args pretrained=openlm-research/open_llama_7b \
    --tasks  arc_challenge \
    --device cuda \
    --output_path results/openllama-7bv1-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 25 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=openlm-research/open_llama_7b \
    --tasks  hellaswag \
    --device cuda \
    --output_path results/openllama-7bv1-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 10 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=openlm-research/open_llama_7b \
    --tasks  truthfulqa_mc \
    --device cuda \
    --output_path results/openllama-7bv1-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 0 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=openlm-research/open_llama_7b \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda \
    --output_path results/openllama-7bv1-leaderboard-mmlu.json \
    --batch_size 16 \
    --num_fewshot 5 \
    --no_cache

## Ours
python main.py \
    --model hf-causal \
    --model_args local_model=True,pretrained=/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_980Bof1000B_tohf/ \
    --tasks truthfulqa_mc \
    --device cuda \
    --output_path results/our-980B-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 0 \
    --no_cache;

python main.py \
    --model hf-causal \
    --model_args local_model=True,pretrained=/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_980Bof1000B_tohf/ \
    --tasks arc_challenge \
    --device cuda \
    --output_path results/our-980B-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 25 \
    --no_cache;


## for technical report (0731)
python main.py \
    --model hf-causal \
    --model_args local_model=True,pretrained=/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_100B_tohf/ \
    --tasks arc_challenge,arc_easy,copa,winogrande,piqa,openbookqa \
    --device cuda \
    --output_path results/our-100B-techreport.json \
    --batch_size 32 \
    --num_fewshot 0 \
    --no_cache;

python main.py \
    --model hf-causal \
    --model_args local_model=True,pretrained=/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_100B_tohf/ \
    --tasks lambada_openai,boolq \
    --device cuda \
    --output_path results/our-100B-techreport.json \
    --batch_size 16 \
    --num_fewshot 0 \
    --no_cache;

python main.py \
    --model hf-causal \
    --model_args local_model=True,pretrained=/mnt/checkpoints/ours_6b7_DistDataV2_CCv3_100B_tohf/ \
    --tasks hellaswag \
    --device cuda \
    --output_path results/our-100B-techreport.json \
    --batch_size 32 \
    --num_fewshot 0 \
    --no_cache;



## RS-sft 70B llama2
export LLama_70b_rsft=rs_llama-2-70b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_4x8A100_trainbs4_accum1_lr6e-6_ep2
link: https://bingdatacupremium.blob.core.windows.net/fwd-data/jingchu/fastchat_ckps/$LLama_70b_rsft/?sv=2021-10-04&st=2023-05-03T11%3A00%3A43Z&se=2025-01-04T11%3A00%3A00Z&sr=c&sp=racwdxltf&sig=mcjoTDldIa6l6cPiYSFbEjAPpdtPTunLrRX9Gt%2BAnlg%3D
ls -alFh /mnt/lm-evaluation-harness/$LLama_70b_rsft/pytorch_model.bin
rm /mnt/lm-evaluation-harness/$LLama_70b_rsft/pytorch_model.bin
# 30mins
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_70b_rsft \
    --tasks  truthfulqa_mc \
    --device auto \
    --output_path results/llama2-70b-rs-leaderboard.json \
    --batch_size 2 \
    --num_fewshot 0 \
    --no_cache
# 2h30mins
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_70b_rsft \
    --tasks  arc_challenge \
    --device auto \
    --output_path results/llama2-70b-rs-leaderboard-arc.json \
    --batch_size 2 \
    --num_fewshot 25 \
    --no_cache
# 19h
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_70b_rsft \
    --tasks  hellaswag \
    --device auto \
    --output_path results/llama2-70b-rs-leaderboard-hellaswag.json \
    --batch_size 2 \
    --num_fewshot 10 \
    --no_cache
# 90h+
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_70b_rsft \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device auto \
    --output_path results/llama2-70b-rs-leaderboard-mmlu.json \
    --batch_size 2 \
    --num_fewshot 5 \
    --no_cache

## RS-sft 7B llama2
export LLama_7b_rsft=rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
link: https://bingdatacupremium.blob.core.windows.net/fwd-data/jingchu/fastchat_ckps/$LLama_7b_rsft/?sv=2021-10-04&st=2023-05-03T11%3A00%3A43Z&se=2025-01-04T11%3A00%3A00Z&sr=c&sp=racwdxltf&sig=mcjoTDldIa6l6cPiYSFbEjAPpdtPTunLrRX9Gt%2BAnlg%3D

# 20mins
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_7b_rsft \
    --tasks  arc_challenge \
    --device cuda \
    --output_path results/llama2-7b-rs-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 25 \
    --no_cache
# 2h10mins  ->  8mins with data-parallel
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_7b_rsft \
    --tasks  hellaswag \
    --device cuda \
    --output_path results/llama2-7b-rs-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 10 \
    --no_cache
# 6mins
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_7b_rsft \
    --tasks  truthfulqa_mc \
    --device cuda \
    --output_path results/llama2-7b-rs-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 0 \
    --no_cache
# 2h20mins
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_7b_rsft \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda \
    --output_path results/llama2-7b-rs-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 5 \
    --no_cache



## 7B llama2
python main.py \
    --model hf-causal \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks  arc_challenge \
    --device cuda \
    --output_path results/llama2-7b-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 25 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks  truthfulqa_mc \
    --device cuda \
    --output_path results/llama2-7b-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 0 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks  hellaswag \
    --device cuda \
    --output_path results/llama2-7b-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 10 \
    --no_cache

python main.py \
    --model hf-causal \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda \
    --output_path results/llama2-7b-leaderboard-mmlu.json \
    --batch_size 4 \
    --num_fewshot 5 \
    --no_cache

### ARC HellaSwag MMLU TruthfulQA
### 53.07 77.74   43.8  38.98          ｜ llama-2-7b-hf
### 52.9  78.55   48.32 45.57          |  llama-2-7b-chat-hf
### 59.39 82.13   55.77 37.38          ｜ llama-2-13b-hf
### 59.04 81.94   54.64 44.12          |  llama-2-13b-chat-hf
### 67.32 87.33   69.83 44.92          ｜ llama-2-70b-hf
### 64.59 85.88   63.91 52.8           |  llama-2-70b-chat-hf

### 52.99 78.63   46.63 38.97          |  our test llama-2-7b-hf             ## 5-shot: 45.3 (paper) zs: 42.9 (ours)
### 52.99 -       -     -              |  our test llama-2-7b-hf             --- test with -experimental + accelerate
### 54.01 78.79   NoSup 38.98          |  our test llama-2-7b-hf             --- new machine (torch2.0++) + test with big-factor branch + accelerate + data-parallel

### 56.74 -       -     47.56         ｜  our test llama-2-7b-rs-sft         --- test with -experimental + accelerate   |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr6e-6_ep3

### 56.23 79.45   49.69 48.05         ｜  our test llama-2-7b-rs-sft         --- test with -experimental + accelerate   |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
### 56.23 -       -     48.05         ｜  our test llama-2-7b-rs-sft         --- new machine (torch2.0++) + test with -experimental + accelerate   |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
### 56.66 79.53   NoSup -             |   our test llama-2-7b-rs-sft         --- new machine (torch2.0++) + test with big-factor branch + accelerate + data-parallel  |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
### 56.83 79.50   NoSup 48.12         |   our test llama-2-7b-rs-sft         --- new machine (torch2.0++) + test with big-factor branch + accelerate + distengle-fsdp + data-parallel  |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
### 56.23 79.41   49.90 48.14         |   our test llama-2-7b-rs-sft         --- new machine (torch2.0++) + test with -experimental + accelerate + distengle-fsdp + data-parallel  |||||   rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
### -> warning: bs for mmlu should be smaller (if we want to use distengle-fsdp)

### 62.37 82.99   56.61 45.54         |  our test llama-2-13b-rs-sft --- new machine (torch2.0++) + test with -experimental + accelerate + distengle-fsdp + data-parallel   |||||   rs_llama-2-13b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_2x8A100_trainbs8_accum1_lr1e-5_ep3

### -    -        -     60.06         |  our test llama-2-70b-rs-sft --- test with -experimental + accelerate   ||||   rs_llama-2-70b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_4x8A100_trainbs4_accum1_lr6e-6_ep2
### 70.48 87.13   69.58 60.09         |  our test llama-2-70b-rs-sft --- new machine (torch2.0++) + test with -experimental + accelerate + distengle-fsdp + data-parallel   |||||   rs_llama-2-70b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_4x8A100_trainbs4_accum1_lr6e-6_ep2

### to support MMLU in big-factor branch (not a good idea for now)
### ref: https://github.com/EleutherAI/lm-evaluation-harness/pull/753/files#diff-122f985c0eb3f1cf7b0b5ae92c01db951c2c04d3a5f6cf102e7a891b75e8e863

### ARC HellaSwag MMLU TruthfulQA
# 7b:  56.23 79.45   49.69 48.05 
# 13b: 62.37 82.99   56.61 45.54
# 70b: 70.48 87.13   69.58 60.09
