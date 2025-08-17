# **BiomedParse RECIST experiments**

This repository is the result of a one month and two weeks internship at the **CRCL** (Centre de Recherche en Cancérologie de Lyon).

I experimented with evaluating the **RECIST** criteria using [**BiomedParse**](https://github.com/microsoft/BiomedParse) as part of a multimodal agent designed to recommend clinical trials to patients for whom no treatment had worked.

# **Structure**

This repository is organized around three Python notebooks, each in its own dedicated folder:

- `lung_RECIST_evaluation`: This notebook contains my initial exploration of [**BiomedParse**](https://github.com/microsoft/BiomedParse), followed by various evaluations of the model’s performance in segmenting lung tumors and assessing the **RECIST** criteria. It also includes a segmentation filtering method I developed and its evaluation.

- `conformal_prediction`: This notebook is an attempt at implementing a **conformal prediction** method. The result tends to show that the model is not certain of its segmentation, making it impossible to determine a threshold.

- `bounding_box_generation`: Given the model’s lack of effectiveness in directly assessing the **RECIST** criteria, I implemented two methods to repurpose [**BiomedParse**](https://github.com/microsoft/BiomedParse) outputs for generating bounding boxes. The idea would then be to pass these bounding boxes to another model such as [**MedSAM-2**](https://supermedintel.github.io/Medical-SAM2/), which I haven't had time to test. However, the results obtained by the bounding boxes on the ground truth are promising.

A more readable **HTML** version of each notebook is also available in each folder.

# **Biblio**

The `biblio` notebook presents the bibliography and useful links I relied on during my internship, covering:
- [**BiomedParse**](https://github.com/microsoft/BiomedParse)
- L'explainability
- Conformal Prediction
- A Colab on [**MedSAM**](https://github.com/bowang-lab/MedSAM)
- Transformers








