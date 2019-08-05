# SimpleMicroscopeMalariaChallenge
This is my Microscope Malaria Challege Customized ResNet technique

Challenge Description
https://codalab.lri.fr/competitions/522#learn_the_details
<br/>This is the challenge uploaded by CodaLab environment for DSA Africa workshop.

Although microscopes are common in Uganda and other developing countries, a shortage of lab technicians to operate them means that access to quality diagnostic services is limited for much of the population. This leads to misdiagnoses of disease, which in turn causes life-threatening conditions to be incorrectly treated, drug resistance, and the economic burden of buying unecessary drugs. Even where health facilities have lab technicians, they are often oversubscribed and have difficulty spending enough time on each sample to give a confident diagnosis. Given that smartphones are widely owned across the developing world, there is a technological opportunity to address this problem: phones can be used to capture and process microscopy images. This project aims to produce a functioning point-of-care diagnosis system on this principle, capable of running on multiple microscope and phone combinations. Read more descreption from the compitition site.

The goal is to count the number of parasites on each image. We divide the whole process into two step pipeline.

    Step 1: Binary classification problem. You need to implement a machine learning model in order to classify whether a patch is postive or negative.
    Step 2: Regression problem. You need to predict the number of parasites on each image (using model trained on Step 1).

