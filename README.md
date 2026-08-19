# Community Concealment from Graph Neural Networks

This repository contains code to run experiments on concealing a target community from GNN-based community detection via graph perturbations including the proposed methods FCom-DICE (Feature-Community-guided DICE).

# Citation
If you use this code or the associated research in your work, please cite our paper:

```bibtex
@misc{manatova2026communityconcealmentunsupervisedgraph,
      title={Community Concealment from Graph Neural Networks}, 
      author={Dalyapraz Manatova and Pablo Moriano and L. Jean Camp},
      year={2026},
      eprint={2602.12250},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.12250}, 
}
```

## Key files

- `lfr_generator.py` — LFR graph + node feature generation.
- `dmon.py` — GNN-based community detection models and supporting utilities, including DMoN, DiffPool, and MinCut.
- `attacks.py` — perturbation methods (DICE and FCom-DICE).
    - DICE (structure-only): `dice_community_attack(...)`
    - FCom-DICE (feature + community): `fcom_dice_community_attack(...)` with `feature_mode = average_community` for changing node features to the average of the community the node is attaching to.
- `run_experiments.py` — runs LFR experiments and writes CSV results.
- `run_experiments_real_networks.py` — runs experiments on real networks and writes CSV results.
- `analysis_helper.py` — helper functions for aggregating results and plotting.
- `results_analysis_paper.ipynb` — notebook used to analyze and plot results for the paper.
- `community_hiding_analysis.ipynb` / `lfr_dmon.ipynb` — playground notebooks.

## Quickstart (small demo)

Create a results directory:

```bash
mkdir -p results
```

### LFR demo DiffPool + DICE
```bash
python run_experiments.py \
  --model diffpool \
  --realizations 2 \
  --n 1000 \
  --mu_values 0.3 \
  --sigma_c_values 0.01 \
  --min_community 10 \
  --p_values 0.5 \
  --b_percentages 0.2 \
  --outfile_csv results/demo_lfr.csv
```

### LFR demo DMoN + FCom-DICE

```bash
python run_experiments.py \
  --model dmon \
  --FComDICE \
  --attack_feature_mode average_community \
  --realizations 2 \
  --n 1000 \
  --mu_values 0.3 \
  --sigma_c_values 0.01 \
  --min_community 10 \
  --p_values 0.5 \
  --b_percentages 0.2 \
  --outfile_csv results/demo_lfr_fcomdice.csv
```

Notes:

`--FComDICE` switches from structure-only DICE to feature-aware FCom-DICE.

`--attack_feature_mode` controls how the target-community features are modified, default for FCom-DICE is `average_community`

### Real network demo (DICE)
```bash
python run_experiments_real_networks.py \
  --network_name Wiki \
  --realizations 2 \
  --p_values 0.5 \
  --b_percentages 0.2 \
  --outfile_csv results/demo_real.csv
```
### Real network demo (FCom-DICE)
```bash
python run_experiments_real_networks.py \
  --FComDICE \
  --attack_feature_mode average_community \
  --network_name Wiki \
  --realizations 2 \
  --p_values 0.5 \
  --b_percentages 0.2 \
  --outfile_csv results/demo_real_fcomdice.csv
```

### Outputs

Experiments produce CSV files with rows like:

`mu,sigma_c,target_label,target_size,b,b_percentage,p,realization,ECS,M1,M2,elapsed_time_sec`

Example filename pattern:

`<model>_fcomdice_<mu>_mincomm_<k>_sigma<...>_p<...>_<feature_mode>.csv`


### Reproducing paper plots

Open and run:

`results_analysis_paper.ipynb`
