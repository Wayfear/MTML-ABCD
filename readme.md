# MTML-BNT

## Usage

1. Change the *path* attribute in file *source/conf/dataset/abcd_mtml_35.yaml* to the path of your dataset.

2. Run the following command to train the model.

```bash
python -m source --multirun dataset=abcd_mtml_35 datasz=100p model=mtmlbnt project=mtml_balance_weight repeat_time=5 training.name=MTMLTrain preprocess=non_mixup dataset.use_balance_weight=True model.mask=False training.epochs=100 training.ig_visualize=False training.save_ig=False training.standard_scaler=True

```

## Installion

```bash
conda create --name bnt python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
pip install hydra-core --upgrade
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```


## Dependencies

  - python=3.9
  - cudatoolkit=11.3
  - torchvision=0.13.1
  - pytorch=1.12.1
  - torchaudio=0.12.1
  - wandb=0.13.1
  - scikit-learn=1.1.1
  - pandas=1.4.3
  - hydra-core=1.2.0


## Tasks

|File|Column|EventName|Description|Percent|Type|ValueDistribution|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|abcd_cbcls01.txt|cbcl_scr_syn_anxdep_r|baseline_year_1_arm_1|AnxDep CBCL Syndrome Scale (raw score) = cbcl_q14_p and cbcl_q29_p and cbcl_q30_p and cbcl_q31_p and cbcl_q32_p and cbcl_q33_p and cbcl_q35_p and cbcl_q45_p and cbcl_q50_p and cbcl_q52_p and cbcl_q71_p and cbcl_q91_p and cbcl_q112_p|99.9%|Continuous|0.0 0.0 1.0 4.0 26.0|
|abcd_cbcls01.txt|cbcl_scr_syn_withdep_r|baseline_year_1_arm_1|WithDep CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 0.0 1.0 15.0|
|abcd_cbcls01.txt|cbcl_scr_syn_somatic_r|baseline_year_1_arm_1|Somatic CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 1.0 2.0 16.0|
|abcd_cbcls01.txt|cbcl_scr_syn_social_r|baseline_year_1_arm_1|Social CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 1.0 2.0 18.0|
|abcd_cbcls01.txt|cbcl_scr_syn_thought_r|baseline_year_1_arm_1|Thought CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 1.0 2.0 18.0|
|abcd_cbcls01.txt|cbcl_scr_syn_attention_r|baseline_year_1_arm_1|Attention CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 2.0 5.0 20.0|
|abcd_cbcls01.txt|cbcl_scr_syn_rulebreak_r|baseline_year_1_arm_1|RuleBreak CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 0.0 2.0 20.0|
|abcd_cbcls01.txt|cbcl_scr_syn_aggressive_r|baseline_year_1_arm_1|Aggressive CBCL Syndrome Scale (raw score)|99.9%|Continuous|0.0 0.0 2.0 5.0 36.0|
|abcd_mhp02.txt|pgbi_p_ss_score|baseline_year_1_arm_1|Parent General Behavior Inventory SUM:  gen_child_behav_1 + gen_child_behav_2 + gen_child_behav_3 + gen_child_behav_4 + gen_child_behav_5 + gen_child_behav_6 + gen_child_behav_7 + gen_child_behav_8 + gen_child_behav_9 +  gen_child_behav_10;  Validation: All items must be answered|99.9%|Continuous|0.0 0.0 0.0 1.0 27.0|
|abcd_mhy02.txt|pps_y_ss_number|baseline_year_1_arm_1|Prodromal Psychosis Scale: Number of Yes Responses Sum: prodromal_1_y, prodromal_2_y, prodromal_3_y, prodromal_4_y, prodromal_5_y, prodromal_6_y, prodromal_7_y, prodromal_8_y,  prodromal_9_y, prodromal_10_y, prodromal_11_y, prodromal_12_y, prodromal_13_y, prodromal_14_y, [prodromal_15_y,  prodromal_16_y, prodromal_17_y], prodromal_18_y, prodromal_19_y], prodromal_20_y], prodromal_21_y;  No minimum number of answers to be valid|99.9%|Continuous|0.0 0.0 1.0 4.0 21.0|
|abcd_mhy02.txt|pps_y_ss_severity_score|baseline_year_1_arm_1|Prodromal Psychosis: Severity Score Sum: (prodromal_1b_y, prodromal_2b_y, prodromal_3b_y, prodromal_4b_y, prodromal_5b_y, prodromal_6b_y, prodromal_7b_y, prodromal_8b_y, prodromal_9b_y, prodromal_10b_y, prodromal_11b_y, prodromal_12b_y, prodromal_13b_y, prodromal_14b_y, [prodromal_15b_y, prodromal_16b_y, prodromal_17b_y, prodromal_18b_y, prodromal_19b_y, prodromal_20b_y, prodromal_21b_y) + (pps_y_ss_ bother_n_1), If  this score = "",  then score = pps_y_ss_number;  No minimum number of answers to be valid|99.9%|Continuous|0.0 0.0 1.0 7.0 104.0|
|abcd_mhy02.txt|upps_y_ss_negative_urgency|baseline_year_1_arm_1|UPPS-P for Children Short Form (ABCD-version), Negative Urgency: upps7_y + upps11_y + upps17_y +  upps20_y; Validation: Minimum of three items answered|99.8%|Continuous|4.0 7.0 8.0 10.0 16.0|
|abcd_mhy02.txt|upps_y_ss_lack_of_planning|baseline_year_1_arm_1|UPPS-P for Children Short Form (ABCD-version),  Lack of Planning: upps6_y + upps16_y + upps23_y + upps28_y; Validation: Minimum of three items answered|99.8%|Continuous|4.0 6.0 8.0 9.0 16.0|
|abcd_mhy02.txt|upps_y_ss_sensation_seeking|baseline_year_1_arm_1|UPPS-P for Children Short Form (ABCD-version), Sensation Seeking: upps12_y + upps18_y + upps21_y + upps27_y; Validation: Minimum of three items answered|99.8%|Continuous|4.0 8.0 10.0 12.0 16.0|
|abcd_mhy02.txt|upps_y_ss_positive_urgency|baseline_year_1_arm_1|UPPS-P for Children Short Form (ABCD-version), Positive Urgency: upps35_y + upps36_y + upps37_y + upps39_y; Validation: Minimum of three items answered|99.8%|Continuous|4.0 6.0 8.0 10.0 16.0|
|abcd_mhy02.txt|upps_y_ss_lack_of_perseverance|baseline_year_1_arm_1|UPPS: Lack of Perseverance (GSSF) upps15_y plus upps19_y plus upps22_y plus upps24_y; Validation : Minimum of three items answered|99.8%|Continuous|4.0 5.0 7.0 8.0 16.0|
|abcd_mhy02.txt|bis_y_ss_bis_sum|baseline_year_1_arm_1|BIS/BAS: BIS Sum:  bisbas1_y + bisbas2_y + bisbas3_y + bisbas4_y + bisbas5_y + bisbas6_y+ bisbas7_y; Validation: All items must be answered|99.9%|Continuous|0.0 7.0 9.0 12.0 21.0|
|abcd_mhy02.txt|bis_y_ss_bas_rr|baseline_year_1_arm_1|BIS/BAS: BAS Reward Responsiveness:  bisbas8_y +  bisbas9_y + bisbas10_y + bisbas11_y + bisbas12_y; Validation: All items must be answered|99.8%|Continuous|0.0 9.0 11.0 13.0 15.0|
|abcd_mhy02.txt|bis_y_ss_bas_drive|baseline_year_1_arm_1|BIS/BAS: BAS drive:  bisbas13_y +  bisbas14_y + bisbas15_y + bisbas16_y; Validation: All items must be answered|99.8%|Continuous|0.0 2.0 4.0 6.0 12.0|
|abcd_mhy02.txt|bis_y_ss_bas_fs|baseline_year_1_arm_1|BIS/BAS: BAS Fun Seeking: bisbas17_y + bisbas18_y + bisbas19_y + bisbas20_y; Validation: All items must be answered|99.8%|Continuous|0.0 4.0 6.0 7.0 12.0|
|abcd_tbss01.txt|nihtbx_picvocab_uncorrected|baseline_year_1_arm_1|NIH Toolbox Picture Vocabulary Test Age 3+ v2.0 Uncorrected Standard Score|98.7%|Continuous|36.0 80.0 84.0 90.0 119.0|
|abcd_tbss01.txt|nihtbx_flanker_uncorrected|baseline_year_1_arm_1|NIH Toolbox Flanker Inhibitory Control and Attention Test Ages 8-11 v2.0 Uncorrected Standard Score|98.6%|Continuous|53.0 90.0 96.0 100.0 116.0|
|abcd_tbss01.txt|nihtbx_list_uncorrected|baseline_year_1_arm_1|NIH Toolbox List Sorting Working Memory Test Age 7+ v2.0 Uncorrected Standard Score|98.3%|Continuous|36.0 90.0 97.0 105.0 136.0|
|abcd_tbss01.txt|nihtbx_cardsort_uncorrected|baseline_year_1_arm_1|NIH Toolbox Dimensional Change Card Sort Test Ages 8-11 v2.0 Uncorrected Standard Score|98.6%|Continuous|50.0 88.0 94.0 99.0 120.0|
|abcd_tbss01.txt|nihtbx_pattern_uncorrected|baseline_year_1_arm_1|NIH Toolbox Pattern Comparison Processing Speed Test Age 7+ v2.0 Uncorrected Standard Score|98.5%|Continuous|30.0 80.0 88.0 99.0 140.0|
|abcd_tbss01.txt|nihtbx_picture_uncorrected|baseline_year_1_arm_1|NIH Toolbox Picture Sequence Memory Test Age 8+ Form A v2.0 Uncorrected Standard Score|98.6%|Continuous|76.0 94.0 102.0 111.0 136.0|
|abcd_tbss01.txt|nihtbx_reading_uncorrected|baseline_year_1_arm_1|NIH Toolbox Oral Reading Recognition Test Age 3+ v2.0 Uncorrected Standard Score|98.6%|Continuous|63.0 87.0 91.0 95.0 119.0|
|abcd_tbss01.txt|nihtbx_fluidcomp_uncorrected|baseline_year_1_arm_1|Cognition Fluid Composite Uncorrected Standard Score|98.0%|Continuous|44.0 85.0 92.0 99.0 131.0|
|abcd_tbss01.txt|nihtbx_cryst_uncorrected|baseline_year_1_arm_1|Crystallized Composite Uncorrected Standard Score|98.4%|Continuous|51.0 82.0 86.0 91.0 115.0|
|abcd_tbss01.txt|nihtbx_totalcomp_uncorrected|baseline_year_1_arm_1|Cognition Total Composite Score Uncorrected Standard Score|98.0%|Continuous|44.0 81.0 87.0 93.0 117.0|
|abcd_ps01.txt|pea_ravlt_sd_trial_vi_tc|baseline_year_1_arm_1||98.2%|Continuous|0.0 8.0 10.0 12.0 15.0|
|abcd_ps01.txt|pea_ravlt_ld_trial_vii_tc|baseline_year_1_arm_1||97.8%|Continuous|0.0 7.0 9.0 12.0 15.0|
|abcd_ps01.txt|pea_wiscv_trs|baseline_year_1_arm_1||97.9%|Continuous|0.0 16.0 18.0 20.0 32.0|
|lmtp201.txt|lmt_scr_rt_correct|baseline_year_1_arm_1||97.1%|Continuous|1091.9 2367.9 2707.7 2989.4 4695.5|
|lmtp201.txt|lmt_scr_efficiency|baseline_year_1_arm_1||0%|-|-|

