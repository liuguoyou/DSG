## ***Diverse Sample Generation: Pushing the Limit of Data-free Quantization***

Created by [Haotong Qin](https://htqin.github.io/), Yifu Ding, XiangguoZhang, Jiakai Wang, [Xianglong Liu](http://sites.nlsde.buaa.edu.cn/~xlliu/)*, and [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/) from Beihang University and Tsinghua University.

![framework](figures/framework.jpg)

### Introduction

This project is the official implementation of our paper *Diverse Sample Generation: Pushing the Limit of Data-free Quantization* [[PDF](https://github.com/htqin/DSG)]. Recently, generative data-free quantization emerges as a practical approach that compresses the neural network to low bit-width without access to real data. It generates data to quantize the network by utilizing the batch normalization (BN) statistics of its full-precision counterpart. However, our study shows that in practice, the synthetic data completely constrained by BN statistics suffers severe homogenization at distribution and sample level, which causes serious accuracy degradation of the quantized network. This paper presents a generic ***Diverse Sample Generation (DSG)*** scheme for the generative data-free post-training quantization and quantization-aware training, to mitigate the detrimental homogenization. In our DSG, we first slack the statistics alignment for features in the BN layer to relax the distribution constraint. Then we strengthen the loss impact of the specific BN layer for different samples and inhibit the correlation among samples in the generation process, to diversify samples from the statistical and spatial perspective, respectively. Extensive experiments show that for large-scale image classification tasks, our DSG can consistently outperform existing data-free quantization methods on various neural architectures, especially under ultra-low bit-width (e.g., 22% gain under W4A4 setting). Moreover, data diversifying caused by our DSG brings a general gain in various quantization methods, demonstrating diversity is an important property of high-quality synthetic data for data-free quantization.

### Results

Here are the results of data-free quantization methods with various architectures on the ImageNet dataset.

**Data-free Post-training Quantization (PTQ) Methods**

![ImageNet-PTQ](figures/ImageNet-PTQ.jpg) 

**Data-free Quantization-aware Training (QAT) Methods**

![ImageNet-QAT](figures/ImageNet-QAT.jpg) 

*The code will be released soon.*