# 𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 data preparation

The **𝑃𝑁 𝑅𝑒𝑙𝑎𝑏𝑒𝑙𝑖𝑛𝑔 data preparation** Java project for a) label selection, and b) harvesting articles with concept occurrence. 
This project is an extension of the projects *[MeSH_Extension](https://github.com/tasosnent/MeSH_Extension)* and *[MeSH_retrospective_dataset](https://github.com/ThomasChatzopoulos/MeSH_retrospective_dataset)*.

## Requirements
java 1.8.0_91

All required libraries are listed in requirements.txt

This project has been developed in NetBeans IDE 8.1

## How to use

### Configure
 Update the configurations in /settings.yaml
 
* workingPath: The path for the folder where all results will be stored (e.g. 'D:\\2005_2020_results')
* meshXmlPath: The path to the folder with the XML files for all MeSH versions (both descXXXX.xml and suppXXXX.xml should be available) (e.g. 'D:\\MeSH_All')
* suggestPMNmappings: False in this work. If True, suggest mappings from new descriptors to SCRs (based on PMN fields, etc) for manual review
* calculateConceptAVGs: False in this work. If True, calculate the average No of concepts per SCR and Descriptor and print in the log for each year
* debugMode: False in this work. If True, additional information will also be printed in the log
* nowYear: The last year to consider for defining the period for article selection (e.g. 2005)
* oldYearInitial: The first year to consider for defining the period for article selection (e.g. 2004)
* splitChar: The character for joining/splitting serialized information
* documentIndexPath: Path to a Lucene index of all PubMed/MEDLINE documents available 
* umls: A local instance of UMLS in MySQL including: dbname, dbuser, and dbpass
* umlsMeshVersion: The MeSH version in UMLS used for mapping MeSH concepts to CUIs
* smdb: A local instance of SemMedDB in MySQL including: dbname, dbuser, and dbpass


### Run
The main method is in DatasetCreator.java and considers no arguments. 
The execution leads to creating in the workingPath a CSV and a JSON file as described in the following section Output. log information is printed in the STD output.

### Output
This project produces two files as listed below, where XXXX stands for nowYear (e.g. 2005):
* A CSV file named **filtering_XXXX_descriptors.csv** (e.g. filtering_2005_descriptors.csv) including all the Fine-Grained Label MeSH descriptors of the referenceYear (e.g. MeSH2020) that were available in the version of nowYear (e.g. 2005). They are annotated with related information (the parents in the MeSH hierarchy, the MeSH categories, etc)
* A JSON file named **filtering_XXXX.json** (e.g. filtering_2005.json) with all articles a) having an abstract and b) being MeSH Completed in the period [oldYearInitial, November of nowYear]. For each article this file provides a) the abstract ("abstractText") and MeSH labels ("Descriptor_names","Descriptor_UIs") from the Lucene index, and b) the weak labels for selected descriptors ("weakLabel"), based on Concept Occurrence of respective CUIs from SemMedDB. Note: December of nowYear is excluded because it is the month when MEDLINE/PubMed and MeSH are usually updated, hence some of the documents MeSH Completed in December of nowYear may follow the MeSH version of the next year (nowYear+1).

## References

* [1] Nentidis, A., Krithara, A., Tsoumakas, G., & Paliouras, G. (2021). What is all this new MeSH about? Exploring the semantic provenance of new descriptors in the MeSH thesaurus. International Journal on Digital Libraries, July 2021, https://doi.org/10.1007/s00799-021-00304-z
* [2] Nentidis, A., Chatzopoulos, T., Krithara, A., Tsoumakas, G., & Paliouras, G. (2023). Large-scale investigation of weakly-supervised deep learning for the fine-grained semantic indexing of biomedical literature. Journal of Biomedical Informatics, Volume 146, October 2023, 104499, https://doi.org/10.1016/j.jbi.2023.104499


