<h2>TensorFlow-FlexUNet-Image-Segmentation-Oil-Spill (2025/12/29)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for Oil-Spill 
based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and <a href="https://www.kaggle.com/datasets/nabilsherif/oil-spill">
<b>oil spill</b></a> dataset on the kaggle web site.
<br><br>
<hr>
<b>Actual Image Segmentation for Oil-Spill Images of 1250x650 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks, but they lack precision in certain areas. <br><br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0002.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0002.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0003.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0003.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0025.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0025.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0025.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0028.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0028.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0028.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from the following google drive:<br><br>
 <a href="https://www.kaggle.com/datasets/nabilsherif/oil-spill">
<b>oil spill</b></a> dataset on the kaggle web site.
<br>
It  contains Sentinel-1 SAR imagery annotated for oil spills. <br>
<br>
On more information of the dataset, please see also <a href="https://github.com/Harsha0112/Oil-Spill-Detection">Oil-Spill-Detection</a>
<br><br>
<b>License</b><br>
Unknown
<br>
<br>
<h3>
2 Oil-Spill ImageMask Dataset
</h3>
 If you would like to train this Oil-Spill Segmentation model by yourself,
 please download the original dataset from <a href="https://www.kaggle.com/datasets/nabilsherif/oil-spill">
<b>oil spill</b></a>
, expand the downloaded in <b>./dataset </b> folder. and run <a href="./generator/split_master.py">split_master.py</a> to split
the train dataset into train and valid subsets.<br> 
<pre>
./dataset
└─Oil-Spill
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>Oil-Spill Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Oil-Spill/Oil-Spill_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
We used the following color-class mapping table to define a rgb_map mask format between indexed colors and rgb colors.<br>
<br>
<a id="color-class-mapping-table"><b>Oil-Spill color class mapping table</b></a>
<table border=1 style='border-collapse:collapse;' cellpadding='5'>
<tr><th>Indexed Color</th><th>Color</th><th>RGB</th><th>Class</th></tr>
<tr><td>1</td><td with='80' height='auto'><img src='./color_class_mapping/Class1.png' widith='40' height='25'></td><td>(0, 255, 255)</td><td>Class1</td></tr>
<tr><td>2</td><td with='80' height='auto'><img src='./color_class_mapping/Class2.png' widith='40' height='25'></td><td>(255, 0, 0)</td><td>Class2</td></tr>
<tr><td>3</td><td with='80' height='auto'><img src='./color_class_mapping/Class3.png' widith='40' height='25'></td><td>(153, 76, 0)</td><td>Class3</td></tr>
<tr><td>4</td><td with='80' height='auto'><img src='./color_class_mapping/Class4.png' widith='40' height='25'></td><td>(0, 153, 0)</td><td>Class4</td></tr>
</table>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained Oil-Spill TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Oil-Spill/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Oil-Spill and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(11,11)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 5

base_filters   = 16
base_kernels   = (11,11)
num_layers     = 8

dropout_rate   = 0.04
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00005
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for Oil-Spill 1+11 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;Oil-Spill 1+4
rgb_map = {(0,0,0):0, (0,255,255):1, (255,0,0):2, (153,76,0):3, (0,153,0):4}
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3,4)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (37,38,39,40)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (74,75,76,77)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 77 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/train_console_output_at_epoch77.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/Oil-Spill/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Oil-Spill/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Oil-Spill</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for Oil-Spill.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/evaluate_console_output_at_epoch77.png" width="880" height="auto">
<br><br>Image-Segmentation-Oil-Spill

<a href="./projects/TensorFlowFlexUNet/Oil-Spill/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this Oil-Spill/test was not low, and dice_coef_multiclass  not high as shown below.
<br>
<pre>
categorical_crossentropy,0.1558
dice_coef_multiclass,0.9192
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Oil-Spill</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for Oil-Spill.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/Oil-Spill/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Oil-Spill Images of 1250x650 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<a href="#color-class-mapping-table">Color class mapping table</a>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0001.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0001.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0009.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0009.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0009.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0013.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0013.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0013.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0033.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0033.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0033.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0023.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0023.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0023.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0028.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0028.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0028.png" width="320" height="auto"></td>
</tr>

<!-- 
-->
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0017.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0017.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/images/img_0036.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test/masks/img_0036.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Oil-Spill/mini_test_output/img_0036.png" width="320" height="auto"></td>
</tr>

</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. Oil Spill Detection using Convolutional Neural Networks and Sentinel-1 SAR Imagery</b><br>
Eleftheria Kalogirou, Konstantinos Christofi, Despoina Makri, Muhammad Amjad Iqbal, Valeria La Pegna,<br>
Marios Tzouvaras, Christodoulos Mettas, Diofantos Hadjimitsis<br>
<a href="https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/757/2025/isprs-archives-XLVIII-G-2025-757-2025.pdf">
https://isprs-archives.copernicus.org/articles/XLVIII-G-2025/757/2025/isprs-archives-XLVIII-G-2025-757-2025.pdf</a>
<br>
<br>
<b>2. Oil spill detection and classification through deep learning and tailored data augmentation</b><br>
Ngoc An Bui,  Youngon Oh,  Impyeong Lee<br>
<a href="https://www.sciencedirect.com/science/article/pii/S1569843224001997">https://www.sciencedirect.com/science/article/pii/S1569843224001997</a>
<br>
<br>
<b>3. Oil spill detection by imaging radars: Challenges and pitfalls</b><br>
Werner Alpers , Benjamin Holt, Kan Zeng <br>
<a href="https://www.sciencedirect.com/science/article/pii/S0034425717304145">https://www.sciencedirect.com/science/article/pii/S0034425717304145</a>
<br>
<br>
<b>4.  Oil-Spill-Detection</b><br>
Harsha0112<br>
<a href="https://github.com/Harsha0112/Oil-Spill-Detection">https://github.com/Harsha0112/Oil-Spill-Detection</a>
<br>
<br>
<b>5. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
