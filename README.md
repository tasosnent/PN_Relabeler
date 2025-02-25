# PN Relabeler
Zero-shot relabeling of weak labels for fine-grained semantic indexing of biomedical literature

In this work we propose a novel zero-shot method, called *PN Relabeler*, to improve the heuristic annotations for fine-grained semantic indexing (FGSI) of biomedical literature generated by the concept occurrence (CO) heuristic. The terms used for the semantic indexing of biomedical literature often correspond to multiple related but distinct concepts. Concept-level FGSI annotations could support more precise information retrieval but require significant manual indexing effort. CO has proven to be a good heuristic for this task. In particular, it is quite precise but still misses some document labels, i.e. it suffers from lower recall. The PN Relabeler method addresses this problem, by introducing a novel approach to combine heuristic annotations for unseen labels with knowledge learned from past labels. To do so, it first tackles the intermediate task of zero-shot relabeling of documents that are labeled with CO-based FGSI annotations. Then, it builds upon the power of domain-specific deep pretrained language models to improve the recall of the heuristic annotation, i.e. reduce the rate of false negative cases. The results reveal that relabeling with PN Relabeler improves the micro-F1 by more than 6 percentage points, compared to CO annotations, and by 3 percentage points, compared to state-of-the-art CO-based FGSI methods.

![Alt text](PN_Relabeler.png "The three parts of PN Relabeler: A) First, a PN Relabeling dataset is developed with data for known descriptors. B) Second, a zero-shot PN Relabeling model is developed for classifying PN articles as TN or FN. C) Finally, at test time, the PN Relabeling model estimates which of the PN articles based on CO or a related method are FN, and relabels them as positive." )


This repository includes the implementation of all parts of the **PN Relabeler** method, organized into two projects: 
1. The [**Data preparation**](/data_preparation) project that performs the selection of suitable MeSH labels, and the harvesting of respective articles with ground truth and heuristic FGSI annotations for *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset* development. 
2. The [**Relabeler**](/Relabeler) project that includes:
    - A [**Dataset development**](/Relabeler/data) part that converts the data into the format adequate for the 𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 task generating a *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
    - A [**Deep learning modeling**](/Relabeler/modeling) part that performs:
        - The development of a *PN Relabeler* model on the *𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 dataset*.
        - The application of the *PN Relabeler* model to relabel predictions for the FGSI task.

## Reference

* [1] Nentidis, A., Krithara, A., Tsoumakas, G., & Paliouras, G. (2024). Zero-shot relabeling of weak labels for fine-grained semantic indexing of biomedical literature. 27th European Conference on Artificial Intelligence (ECAI-2024), Oct 2024
