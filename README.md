# Transfer-learning-using-VGG19-
Fine-tuning VGG19 to classify anomalies in orange tree leaves. The dataset comprises 953 color images of orange tree leaves from the Citrus sinensis (L.) Osbeck species, representing 12 classes of citrus diseases. To access the dataset, please complete the License Agreement and send it to jjgarza(at)docentes.uat.edu.mx. A summary of the diseases included in this dataset is presented in the following plot.

![classes](https://github.com/user-attachments/assets/06779f2b-9423-4465-bddb-1b3ce466c022)

VGG19 is a convolutional neural network (CNN) consisting of 19 layers: 16 convolutional layers and 3 fully connected layers (https://doi.org/10.48550/arXiv.1409.1556). This pre-trained CNN is fine-tuned based on the methodology described in the paper "A Huanglongbing Detection Method for Orange Trees Based on Deep Neural Networks and Transfer Learning" (10.1109/ACCESS.2022.3219481). 10-fold cross-validation was used with a learning rate of 0.0001, a momentum of 0.9, and training for 100 epochs. The results show that VGG19 achieved an overall accuracy of 96.53% and an MCC of 96.22%. The confusion matrix can be seen below:

![CF](https://github.com/user-attachments/assets/86e0ff11-8a31-4feb-b4d7-8e4666a3c68d)



