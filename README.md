# ApnoeDetectRaw

![](https://img.shields.io/github/stars/mabartcz/ApnoeDetectRaw) 
![](https://img.shields.io/github/forks/mabartcz/ApnoeDetectRaw) 
![](https://img.shields.io/github/issues/mabartcz/ApnoeDetectRaw) 

**Apnoe Detect Raw** is Python 3 script derivated from [ApnoeDetect](https://github.com/mabartcz/ApnoeDetect) to be used by general public.

Difference with ApnoeDetecr are:
 - input data are simple array (insted private file format), 
 - there is no GUI (simple script edit).

The script purpouse is to detect sleep apnea and SpO2 desaturations from PSG signal. The used signals are airflow and blood oxygen saturation (SpO2). The detection is made by trained neural network.

**Data flow diagram:**
![enter image description here](https://github.com/mabartcz/ApnoeDetectRaw/blob/master/Diagam.png?raw=true)
	
**Input:**
Raw data in numpy array. Two rows. First row flow signal, Second row SpO2 signal.
	
**Output:**
1D array with predicted values about signal segment (1 = event is present, 0 = no event detected).

**Requiraments**
 - SciPy 1.7.1
 - NumPy 1.21.2
 - Keras 2.6.0
 - Keras_sequential_ascii 0.1.1
> (all available from pip: <https://pypi.org/>)

**Publications:**
