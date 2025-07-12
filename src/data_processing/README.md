Implementation of the processing pipeline

Responsible for:

- Reading the dataframe.
- Cleaning the data.
- Chunking the answers when needed.
- Preparing the data in the required format for the training step.
- Saving the processed data to S3.

---

**Disclaimer:** This pipeline was implemented in a Jupyter notebook due to time constraints for this assignment. In a production environment, this process should be handled by dedicated services such as a SageMaker Processing or Preprocessing Pipeline.