# bio-ai-project

This project focuses on training and explaining biomedical AI models for various datasets. It supports multiple models (`uni`, `vit`, `resnet`, `conch`) and provides tools for model training, evaluation, and explainability using Grad-CAM, attention rollout, and other methods.


## Requirements
- Python 3.11 or 3.12 (Conda environment recommended)
- Hugging Face token for accessing pre-trained models
- Dependencies listed in `requirements.txt`
- External repository: [vit-explain](https://github.com/jacobgil/vit-explain)


1. add token.json with "token": "[your hugginggface token]"
2. install from the requirements.txt
3. load repository https://github.com/jacobgil/vit-explain as vit-explain-main
4. run train_main.py to train a model 
    - use argument "--model" to select between "uni", "vit", "resnet" and "conch"
5. run explain.py and explain_breast_cancer.py to run the explainabity methods with the after training all models
