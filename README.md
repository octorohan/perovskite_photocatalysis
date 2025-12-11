# Perovskite Photocatalysis ML Repro

🧪 Perovskite Photocatalyst Yield Prediction
Reproducing a Research Paper with Machine Learning

Authors: Rohan & Vishakha

<p align="center"> <img src="banner.png" width="100%" alt="Project Banner"> </p>
<div align="center">








</div>
📌 Overview

This repository presents a complete reproduction of a peer-reviewed research paper on Machine learning analysis of photocatalytic CO2 reduction on perovskite materials

The project demonstrates a highly modular ML pipeline, combined with rigorous chemical engineering domain knowledge, enabling accurate prediction of yield (µmol gcat⁻¹ h⁻¹) from experimental synthesis conditions and material descriptors.

🎯 Objectives

Build a fully reproducible ML workflow to match research paper results

Clean, curate, and harmonize heterogeneous experimental datasets

Apply domain-aware imputations instead of naive statistical methods

Train robust predictive models under data skew using log1p(y)

Produce interpretable scientific insights using SHAP & PDP

Validate trends and ablation effects against the published study

🧬 Scientific Background

Perovskite photocatalysts (general formula ABO₃) play a key role in photocatalytic hydrogen generation.
Yield is strongly influenced by:

Factor	Examples
A-site cation	Sr, Ba, Na, La
B-site cation	Ti, Ta, Nb
Doping	Fe, Co, Ni
Synthesis method	Hydrothermal (HT), Sol–Gel, Solid-State, Pechini
Band Gap	Controls visible/UV activity
Surface Properties	BET area, calcination time & temperature
Cocatalysts	Pt, Au, Ru, LDH
Reaction environment	Liquid vs Gas phase

Understanding these trends requires both ML reasoning and chemical intuition — the purpose of this project.

⚙️ ML Workflow
✔ 1. Data Quality & Harmonization

Normalized noisy synthesis labels

Converted all units to consistent µmol gcat⁻¹ h⁻¹

Fixed misformatted numeric fields

Removed impossible or duplicate entries

✔ 2. Domain-Aware Imputation

Instead of blind mean/mode filling, we use rules from literature:

Missing Parameter	Imputation Rule
Reaction Temp/Pressure	Mode within same phase (Liq/Gas)
Calcination Temperature	If missing → assume 25°C (no calcination)
BET Surface Area	Mean for same (Perovskite + Structure) group
Band Gap	Predicted via Linear Regression on known band gaps
H₂O:CO₂ Ratio	Mean for similar light-type + similar catalyst mass
✔ 3. Outlier Handling

Gas-phase data has extreme tail values. We remove entries above:

𝜇
+
3
𝜎
μ+3σ

This improves stability and cross-validation performance.

✔ 4. Model Training

Models trained separately for Liquid and Gas phases:

Random Forest Regression

Decision Tree (interpretable synthesis rules)

Targets modeled as:

𝑦
′
=
log
⁡
(
1
+
𝑦
)
y
′
=log(1+y)
✔ 5. Interpretability

Permutation Feature Importance

Partial Dependence Plots (PDPs)

SHAP Summary & Beeswarm Plots

These help understand:

how calcination time affects charge separation

how BET area correlates with active surface

how A/B-site cations affect band gap & yield

how cocatalysts enhance hydrogen evolution

✔ 6. Ablation Study

We remove one feature group at a time:

Synthesis Method

A-site

B-site

Band Gap

Cocatalyst

This reveals sensitivity of the model to each scientific factor.

📈 Key Results
Liquid Phase (Liq)

CV R² (log-scale): ~0.85 (excellent)

Top Predictors:

Calcination Time

Catalyst Weight

BET Surface Area

Ablation strongest drop: Cocatalyst

Cocatalysts accelerate electron trapping and reduce recombination — exactly matching literature.

Gas Phase (Gas)

CV R² (log-scale): ~0.62 (expected lower)

Top Predictors:

BET Surface Area

Calcination Time

Band Gap

Ablation strongest drop: B-site cation

Gas-phase reactions are more structure-dominated and less synthesis-sensitive — matching the original paper's observation.

📊 Automated Report

Running the complete pipeline generates a professional research-style PDF:

results/final_report.pdf


It includes:

Data distributions (raw → log1p)

Parity plots & residual diagnostics

Feature importance & PDPs

SHAP explanations

Ablation analysis

Comparison with original paper

Appendix: reproducibility instructions

📁 Repository Structure
scripts/
    step1_load_and_check.py
    step2_impute_and_bandgap.py
    step3_retrain_log1p.py
    step4_analysis_fixed.py
    step5_parity_plots.py
    step6_shap_analysis.py
    step7_generate_report.py
    step5_unit_correction_v2.py
    step6_remove_outliers.py
    diagnose_yield_gas.py

results/
    step2_clean.csv
    ablation_summary.csv
    feature_importances_liq.csv
    feature_importances_gas.csv
    final_report.pdf
    figures/

data/
    perovskite_full.csv

🚀 Run the Full Pipeline
python scripts/step2_impute_and_bandgap.py
python scripts/step5_unit_correction_v2.py
python scripts/step6_remove_outliers.py
python scripts/step3_retrain_log1p.py
python scripts/step4_analysis_fixed.py
python scripts/step5_parity_plots.py
python scripts/step6_shap_analysis.py
python scripts/step7_generate_report.py

🤖 For ML — Technical Highlights

Advanced feature engineering

Missing data handled with hybrid ML+domain logic

Cross-validation with log-transformed targets

Model interpretability (SHAP, PDP)

Clean modular code structure

Fully automated PDF reporting (ReportLab)

Rigorous reproducibility pipeline

🧪 For Chemical Engineering — Scientific Strengths

Deep understanding of perovskite chemistry

Band gap relevance to photocatalysis

Structure–property relationships (A/B-site effects)

Impact of surface area & calcination conditions

Catalyst mass and cocatalyst influence

Gas vs liquid-phase mechanistic differences

Scientific validation of ML insights

🔮 Future Improvements

Use graph neural networks (CGCNN / MEGNet) for structure-aware modeling

Incorporate DFT-calculated descriptors

Bayesian optimization for new perovskite candidates

Build a web app for material recommendation

🙌 Acknowledgments

Thanks to the authors of the original research whose methodology guided this work.

📬 Contact
Email: rohanofficialpurpose@gmail.com 