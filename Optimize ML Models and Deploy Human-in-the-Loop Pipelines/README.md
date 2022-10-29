# Practical-Data-Science-on-the-AWS-Cloud-Specialization

![pathway of this Course](https://github.com/Ashleshk/Practical-Data-Science-on-the-AWS-Cloud-Specialization/blob/main/Capture.PNG)


#  **Optimize ML Models and Deploy Human-in-the-Loop Pipelines**

## Notes
* **few advanced concepts in training and tuning machine learning models**
    * When training machine learning models, hyper parameter tuning is a critical step to get the best quality model and this could mean a **model with the highest possible accuracy or lowest possible error**. 
    * few popular algorithms that are used for **automated model tuning** 
    * automated hyper parameter tuning on amazon Sage maker 
        * *automated hyperparameter tuning*, 
        * *distributor training*, and 
        * *optimizing training cost*. 
    * Next I will show you how to apply the automated hyper parameter tuning to BERT based NLP or the natural language processing text classifier. 
        * You may recall BERT's transferred bidirectional encoder representations from transformers. 
    * *Hyper parameter tuning is typically a time consuming and a compute intensive process*.  **hyper parameter tuning on Sage maker** that allows you to speed up your tuning jobs. 
    * **concept of check pointing in machine learning** and discuss how Sage maker leverages the idea of check pointing to save on training costs using a capability called managed spot training.
        * Continuing with the theme of discussing the training challenges, I will introduce tune distributed training strategies. 
        * **Data panelism, and model panelism** that allow you to handle training at scale. The challenges addressed by these strategies include training with large volumes of data as well as dealing with increased model complexity. 
    * Finally, discussion of **Sage maker capability of bringing your own container**, which allows you to implement your own custom logic for your algorithms and train on Sage maker managed infrastructure.

## HyperParameter Tuning Model

   ![HyperParameter Tuning Model](https://github.com/Ashleshk/Practical-Data-Science-on-the-AWS-Cloud-Specialization/blob/main/Optimize%20ML%20Models%20and%20Deploy%20Human-in-the-Loop%20Pipelines/images/hpt.png) 

## Amazon SageMaker HPT job

   ![Amazon SageMaker HPT job](https://github.com/Ashleshk/Practical-Data-Science-on-the-AWS-Cloud-Specialization/blob/main/Optimize%20ML%20Models%20and%20Deploy%20Human-in-the-Loop%20Pipelines/images/sagemaker_hpt.png) 

## Generated result     
* post completion of Assignment

    ![Confusion Matrix](https://github.com/Ashleshk/Practical-Data-Science-on-the-AWS-Cloud-Specialization/blob/main/Optimize%20ML%20Models%20and%20Deploy%20Human-in-the-Loop%20Pipelines/generated/confusion_matrix.png)

























