# DeepResolution
Deep-Learning-Based Multivariate Curve Resolution
----------
Deep-Learning-Based Multivariate Curve Resolution (DeepResolution) method has been proposed for automatic resolution of GC-MS data. It has excellent performance in resolving overlapping peaks and is suitable for large-scale data analysis. Compared with the classical multi-curve resolution method, it has the characteristics of fast, accurate, scalable and fully automatic.

<div align="center">
<img src="https://raw.githubusercontent.com/xiaqiong/DeepResolution/master/Flowchart%20of%20DeepResolution.png" width=600 height=480 />
</div>

# Installation

## python and TensorFlow

Python 3.5.2，available at [https://www.python.org.](https://www.python.org/) 

TensorFlow (version 1.14.0-GPU)，available at [https://github.com/tensorflow.](https://github.com/tensorflow) 

## Install dependent packages

The packages mainly include: numpy,Scipy,Matplotlib,pandas,sklearn,csv and os.

These packages are included in the integration tool Anaconda [(https://www.anaconda.com).](https://www.anaconda.com/) 

# Download the model and example data

Since the model exceeded the limit, we have uploaded all the models and some example data to the [google drive](https://drive.google.com/drive/folders/19y6JYQY0VNkGMmjCi_1EF1EcMvDOdXn-?usp=sharing).

# Clone the repository and run it directly
[git clone](https://github.com/xiaqiong/DeepResolution) 
###1.Training CNN model
Run the file 'component_identification.py'.

The corresponding example data have been uploaded to the data folder named 'data_1.npy' and 'labels_1.npy'.These are augmented data for a component.

###2.Predict GC-MS data automatically

Run the file 'DeepResolution.py'.

Example data have been uploaded to the data folder named 'zhi10-5vs1.CDF'. The file named 'component.csv' is the components's name of all our CNN models. Download the [model](https://drive.google.com/drive/folders/19y6JYQY0VNkGMmjCi_1EF1EcMvDOdXn-?usp=sharing) and these example data，DeepCID can be reload and predict easily.

More example data can be gotten form [google drive](https://drive.google.com/drive/folders/19y6JYQY0VNkGMmjCi_1EF1EcMvDOdXn-?usp=sharing).


# Contact

Xiaqiong Fan: xiaqiongfan@csu.edu.cn
