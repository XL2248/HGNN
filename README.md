# HGNN
Code for Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation (AAAI21)
Requirement: 

nltk>=3.2.4

numpy>=1.13.0

regex>=2017.6.7

tensorflow>=1.2.0
# Generate data
## For MELD
1. generate the data format and the matrix A for training 
(The uploaded train.txt, dev.txt, and test.txt are the facial features extracted by Openface.)
python generate_data4meld.py train/test/dev
python generate_matrix_A4meld.py train/test/dev
python generate_speakers4meld.py (generate speakers' name)
## For DailyDialog
generate_data_matrix_A_4dailydialog.py

# Training
1、parameter setting:
hyperparams.py



2、To generate vocab:
python prepro.py


3、To train:
bash train.py


4、To eval:
python eval.py


5、The raw dialogue data：

joey: or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral

chandler: you 're a genius !	joy

joey: aww , man , now we wo n't be bank buddies !	sadness

chandler: now , there 's two reasons .	neutral

Souce looks like:

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  you 're a genius !		joy  \</d\>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  you 're a genius !		joy  \</d\>  aww , man , now we wo n't be bank buddies !		sadness  \</d\>  

Target:

chandler  \</d\>  you 're a genius !  \</d\>  joy  \</d\> 

joey  \</d\>  aww , man , now we wo n't be bank buddies !  \</d\>  sadness  \</d\>  

chandler  \</d\>  now , there 's two reasons .  \</d\>  neutral  \</d\>  
