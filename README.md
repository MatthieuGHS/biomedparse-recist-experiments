# **BiomedParse RECIST experiments**

This repository is the result of a one month and two weeks internship at the **CRCL** (Centre de Recherche en Cancérologie de Lyon).

I experimented with evaluating the **RECIST** criteria using [**BiomedParse**](https://github.com/microsoft/BiomedParse) as part of a multimodal agent designed to recommend clinical trials to patients for whom no treatment had worked.

# **Structure**

This repository is organized around three Python notebooks, each in its own dedicated folder:

- `lung_RECIST_evaluation`: This notebook contains my initial exploration of [**BiomedParse**](https://github.com/microsoft/BiomedParse), followed by various evaluations of the model’s performance in segmenting lung tumors and assessing the **RECIST** criteria. It also includes a segmentation filtering method I developed and its evaluation.

- `conformal_prediction`: This notebook is an attempt at implementing a **conformal prediction** method.

- `bounding_box_generation`: Given the model’s lack of effectiveness in directly assessing the **RECIST** criteria, I implemented two methods to repurpose [**BiomedParse**](https://github.com/microsoft/BiomedParse) outputs for generating bounding boxes. These can be passed to another model such as [**MedSAM-2**](https://supermedintel.github.io/Medical-SAM2/), leading to promising results.

Each folder includes a `README.md` with more detailed explanations of the notebook and the conclusions drawn. A more readable HTML version of each notebook is also available in each folder.

# **Biblio**

The `biblio` notebook presents a bibliography and useful links I used during my internship on:
- [**BiomedParse**](https://github.com/microsoft/BiomedParse)
- Explainability
- Conformal prediction
- A Colab on [**MedSAM**](https://github.com/bowang-lab/MedSAM)
- Transformers














