export LLama_7b_rsft=rs_llama-2-7b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_1x8A100_trainbs16_accum1_lr1e-5_ep3
export LLama_70b_rsft=rs_llama-2-70b-hf_merge_turn1_20k_turn2_20k_turn3_20k_get_48k_4096_4x8A100_trainbs4_accum1_lr6e-6_ep2
export SelectModel=$LLama_7b_rsft
export TASKs=hellaswag\ arc_challenge\ truthfulqa_mc2\ hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions
export FEWSHOTs=10,25,0,5
export BS=16
python main.py \
    --model hf-causal-experimental \
    --model_args use_accelerate=True,pretrained=/mnt/lm-evaluation-harness/$LLama_7b_rsft \
    --tasks  hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions \
    --device cuda \
    --output_path results/llama2-7b-rs-leaderboard.json \
    --batch_size 16 \
    --num_fewshot 5 \
    --no_cache

IFS=' ' read -ra TASK_L <<< "$TASKs"
IFS=',' read -ra FEWSHOT_L <<< "$FEWSHOTs"
for i in "${!TASK_L[@]}"; do
    TASK=${TASK_L[i]}
    FEWSHOT=${FEWSHOT_L[i]}
    echo $TASK
    echo $FEWSHOT
    python main.py \
        --model hf-causal-experimental \
        --model_args use_accelerate=True,pretrained=/mnt/$LLama_7b_rsft,LOAD_MODE=False,SAVE_MODE=True \
        --tasks  $TASK \
        --output_path results/$SelectModel-leaderboard-$TASK.json \
        --device cuda \
        --batch_size $BS \
        --num_fewshot $FEWSHOT \
        --no_cache
    cd lm_eval/models; python inference_only.py /mnt/$SelectModel; cd ../..;
    python main.py \
        --model hf-causal-experimental \
        --model_args use_accelerate=True,pretrained=/mnt/$LLama_7b_rsft,LOAD_MODE=True,SAVE_MODE=False \
        --tasks  $TASK \
        --output_path results/$SelectModel-leaderboard-$TASK.json \
        --device cuda \
        --batch_size $BS \
        --num_fewshot $FEWSHOT \
        --no_cache
    done