# HGNN
Code for Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation (AAAI21)

## Introduction

The implementation is based on [ReCoSa](https://github.com/zhanghainan/ReCoSa). Download [MELD](https://github.com/declare-lab/MELD) file and [DailyDialog](https://www.aclweb.org/anthology/I17-1099.pdf).

## Requirement: 

+ nltk>=3.2.4

+ numpy>=1.13.0

+ regex>=2017.6.7

+ tensorflow>=1.2.0

## Generate data
+ For MELD

Generate the data format and the matrix A for training 
(The uploaded train.txt, dev.txt, and test.txt are the facial features extracted by [Openface](https://github.com/TadasBaltrusaitis/OpenFace). You can also use the openface4extract_pic_feature.py code to generate by yourself.)

```
python generate_data4meld.py train/test/dev

python generate_matrix_A4meld.py train/test/dev

python generate_speakers4meld.py (generate speakers' name)
```
+ For DailyDialog

You need to modify the code to remove the speaker, the image part to train dailydialog.
```
generate_data_matrix_A_4dailydialog.py
```
# Training
1、parameter setting:
```
hyperparams.py
```

2、To generate vocab:
```
python prepro.py
```

3、To train:
```
python train.py
```

4、To eval:
```
python eval.py
```

5、The raw dialogue data：
```
joey: or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral

chandler: you 're a genius !	joy

joey: aww , man , now we wo n't be bank buddies !	sadness

chandler: now , there 's two reasons .	neutral
```
Souce looks like:
```
or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  you 're a genius !		joy  \</d\>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  \</d\>  you 're a genius !		joy  \</d\>  aww , man , now we wo n't be bank buddies !		sadness  \</d\>  
```
Target:
```
chandler  \</d\>  you 're a genius !  \</d\>  joy  \</d\> 

joey  \</d\>  aww , man , now we wo n't be bank buddies !  \</d\>  sadness  \</d\>  

chandler  \</d\>  now , there 's two reasons .  \</d\>  neutral  \</d\>  
```

## Citation

If you find this project helps, please cite our paper :)

```
@misc{liang2020infusing,
      title={Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation}, 
      author={Yunlong Liang and Fandong Meng and Ying Zhang and Jinan Xu and Yufeng Chen and Jie Zhou},
      year={2020},
      eprint={2012.04882},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
