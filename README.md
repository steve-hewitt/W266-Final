#Natural Language Processing with Deep Learning (W266)
UC Berkeley School of Information 
Final Project
By Steven Hewitt and Intae Kim

The codebase in this repository corresponds to our paper exploring novel decoding strategies on the topic of machine question generation.

# Repo Contents
    ├── Load_and_Examine.ipynb              <- Notebook to load trained Baseline T5 model and experiment with predictions.
    ├── Load_and_Examine_GPP.ipynb          <- Notebook to load trained GPP T5 model and experiment with predictions.
    ├── Model Evaluation.ipynb              <- Notebook to grade model predictions with BLEU, ROUGE-L, and METEOR scores.
    ├── README.md                           <- The top-level README describing the repo contents.
    ├── T5_SQuAD_QG_Baseline.py             <- Python script to train the Baseline T5 model.
    ├── T5_SQuAD_QG_With_GPP.py             <- Python script to train the Baseline T5 model.
    ├── Visualizations.ipynb                <- Notebook exploring model predictions in-depth and generating various charts.
    ├── Predictions
        ├── FINAL_NS_prediction_dict_GPP256.json        <- Predictions on test set from GPP T5 model using nucleus search.
        ├── FINAL_NS_prediction_dict_base256.json       <- Predictions on test set from Baseline T5 model using nucleus search.
        ├── FINAL_NS_reference_dict_GPP256.json         <- Ground truth labels on test set from GPP T5 model using nucleus search.
        ├── FINAL_NS_reference_dict_base256.json        <- Ground truth labels on test set from Baseline T5 model using nucleus search.
        ├── FINAL_prediction_dict_GPP256.json           <- Predictions on test set from GPP T5 model using beam search.
        ├── FINAL_prediction_dict_base256.json          <- Predictions on test set from Baseline T5 model using beam search.
        ├── FINAL_reference_dict_GPP256.json            <- Ground truth labels on test set from GPP T5 model using beam search.
        ├── FINAL_reference_dict_base256.json           <- Ground truth labels on test set from Baseline T5 model using beam search.
        ├── df_GPP256.pkl                               <- Pickled Pandas DataFrame of scoring on individual predictions.
        ├── df_GPP256_NS.pkl                            <- Pickled Pandas DataFrame of scoring on individual predictions.
        ├── df_baseline256.pkl                          <- Pickled Pandas DataFrame of scoring on individual predictions.
        └── df_baseline256_NS.pkl                       <- Pickled Pandas DataFrame of scoring on individual predictions.
    └── Prototyping                          
        ├── T5_SQuAD_QG_Baseline.ipynb                  <- Original prototyping and test code for Baseline T5 model.
        └── T5_SQuAD_QG_With_GPP.ipynb                  <- Original prototyping and test code for GPP T5 model.
