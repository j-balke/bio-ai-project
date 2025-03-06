# bio-ai-project

# how to use:
We used python 3.11/3.12 conda environments

1. add token.json with "token": "[your hugginggface token]"
2. install from the requirements.txt
3. load repository https://github.com/jacobgil/vit-explain as vit-explain-main
4. run train_main.py to train a model 
    - use argument "--model" to select between "uni", "vit", "resnet" and "conch"
5. run explain.py and explain_breast_cancer.py to run the explainabity methods with the after training all models
