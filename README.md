# Transfer-learning-using-VGG19-
Fine-tuning VGG19 to classify anomalies in orange tree leaves. The dataset comprises 953 color images of orange tree leaves from the Citrus sinensis (L.) Osbeck species, representing 12 classes of citrus diseases. To access the dataset, please complete the License Agreement and send it to jjgarza(at)docentes.uat.edu.mx. A summary of the diseases included in this dataset is presented in the following plot. 

![classes](https://github.com/user-attachments/assets/06779f2b-9423-4465-bddb-1b3ce466c022)

VGG19 is a CNN consisting of 19 layers: 16 convolutional layers and 3 fully connected layers (https://arxiv.org/abs/1409.1556). This pre-trained CNN is fine-tuned based on the following paper (https://ieeexplore.ieee.org/document/9938991). 10-fold cross-validation was used with a learning rate of 0.0001, momentum of 0.9, and training for 100 epochs

The results shows that VGG19 obtained an overall accuracy of 96.53% and a MCC of %96.22. Moreover the confusion matrix can be seen below




![CF](https://github.com/user-attachments/assets/86e0ff11-8a31-4feb-b4d7-8e4666a3c68d)



