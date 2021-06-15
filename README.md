# HGNN
Code for Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation (AAAI21)

## Introduction

The implementation is based on [ReCoSa](https://github.com/zhanghainan/ReCoSa). Download [MELD](https://github.com/declare-lab/MELD) file and [DailyDialog](https://www.aclweb.org/anthology/I17-1099.pdf).

## Requirement: 

+ nltk>=3.2.4

+ numpy>=1.13.0

+ regex>=2017.6.7

+ tensorflow=1.10.1

+ Scipy=1.1.0

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

You need to modify the code to remove the speaker, the image part to train on Dailydialog.
```
python generate_data_matrix_A_4dailydialog.py
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
or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  </d>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  </d>  you 're a genius !		joy  </d>  

or ! or , we could go to the bank , close our accounts and cut them off at the source .		neutral  </d>  you 're a genius !		joy  </d>  aww , man , now we wo n't be bank buddies !		sadness  </d>  
```
Target:
```
chandler  </d>  you 're a genius !  </d>  joy  </d> 

joey  </d>  aww , man , now we wo n't be bank buddies !  </d>  sadness  </d>  

chandler  </d>  now , there 's two reasons .  </d>  neutral  </d>  
```

## Citation

If you find this project helps, please cite our paper :)

```
@article{Liang_Meng_Zhang_Chen_Xu_Zhou_2021, 
	title={Infusing Multi-Source Knowledge with Heterogeneous Graph Neural Network for Emotional Conversation Generation}, 
	volume={35}, 
	url={https://ojs.aaai.org/index.php/AAAI/article/view/17575}, 
	number={15}, 
	journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
	author={Liang, Yunlong and Meng, Fandong and Zhang, Ying and Chen, Yufeng and Xu, Jinan and Zhou, Jie}, 
	year={2021}, 
	month={May}, 
	pages={13343-13352} 
}
```
