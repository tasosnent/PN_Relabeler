# 𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 modeling

The **𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 modeling** Python project for a) *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset* generation, b) *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 model* development, and c) *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 model* application. 
This implementation builds upon the open-source project *[Deep Beyond MeSH](https://github.com/tasosnent/DBM)*.

This project includes:
- A [**Dataset development**](./data) part that converts the data into the format adequate for the 𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 task generating a *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
- A [**Deep learning modeling**](./modeling) part that performs:
    - The development of a *PN Relabeler* model on the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
    - The application of the *PN Relabeler* model to relabel predictions for the FGSI task.

## How to use

1. Use the [**Dataset development**](./data) part to select the FGLs for *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset* based on predetermined criteria.
   - Configure: Update the field values in [settings_SCL_selection.yaml](./data/settings_SCL_selection.yaml) accordingly.
   - Run: python FGL_selection.py
   - Input: The data files as generated by the *data_preparation* project (e.g. label_freq.json and filtering_2005_descriptors.csv).
   - Output: A CSV file (e.g. FN_SCL_selected.csv) with a set of FGLs selected for the development of the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
2. Use the [**Dataset development**](./data) part to generate a *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
   - Configure: Uncomment *Part 1* in [settings_PN_Relabeler.yaml](./data/settings_PN_Relabeler.yaml) and update the field values accordingly.
   - Run: python Datasets_PN_Relabeler.py
   - Input: The data files as generated by the *data_preparation* project (e.g. filtering_2005.json) and the FGLs selected above (e.g. FN_SCL_selected.csv).
   - Output: The *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset* as a folder with respective sub-folders. Each sub-folder the respective train.csv, dev.csv, and test.csv for model development.
3. Use the [**Deep learning modeling**](./modeling) part to develop a *PN Relabeler model*.
   - Configure: Update the field values in [settings.yaml](./modeling/settings.yaml) and set skip_zs_evaluation = true.
   - Run: CUDA_VISIBLE_DEVICES=0 python Transformer_models.py ./modeling/settings.yaml
   - Input: The path to the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset* as generated above (step 2).
   - Output: The trained *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 model* model.
4. Use the [**Dataset development**](./data) part to convert an *FGSI dataset* in the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔* format.
   - Configure: Uncomment *Part 2* in [settings_PN_Relabeler.yaml](./data/settings_PN_Relabeler.yaml) and update the field values accordingly.
   - Run: python Datasets_PN_Relabeler.py 
   - Input: The data files of an evaluation *FGSI dataset* as generated by the [*RetroBM* project](https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset) (e.g. test_2019.json and UseCasesSelected_2019.csv).
   - Output: The *FGSI dataset* in the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔* format for applying the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 model*.  
5. Use the [**Deep learning modeling**](./modeling) part to generate relabeling probabilities for an *FGSI dataset*.
   - Configure: Update the field values in [settings.yaml](./modeling/settings.yaml) and set skip_model_training = true.
   - Run: CUDA_VISIBLE_DEVICES=0 python Transformer_models.py ./modeling/settings.yaml
   - Input: The path to the evaluation *FGSI dataset* as converted above (step 4).
   - Output: The probabilities predcited by the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 model* for the *FGSI dataset*.
6. Use the [**Deep learning modeling**](./modeling) part to apply *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔* on existing weak labels/predictions for the *FGSI dataset*.
   - Configure: Update the field values in [settings_apply_relabeler.yaml](./modeling/settings_apply_relabeler.yaml) and set skip_model_training = true.
   - Run: python Apply_Relabeler.py
   - Input: The paths to the evaluation *FGSI dataset*, both the original version and the converted ones (from step 4), the weak labels and/or predictions of DBM/ALO3 approaches, and the predicted *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔* probabilities (step 5).
   - Output: The relabeled predictions for the *FGSI dataset* and the evaluation measures.
   
## Requirements
These scripts are written in Python 3.9.7.

Libraries and versions required are listed in requirements.txt.

This project has been developed in PyCharm 2021.1.2 (Community Edition).