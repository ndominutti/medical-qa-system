Implementation of a processing pipeline in charge of:

* reading the dataframe
* cleaning the data
* chunking the answers when needed
* preparing the data in to the needed format for the training step
* saving the data into S3

---

Disclaimer: this pipeline was implemented in a jupyter notebook just because of the lack of time to work on this process. In a production environment this process should be handled by services like a Sagemaker Preprocessing Pipeline.