This study addresses the challenging task of detecting and localizing human *hand*s in still images, with a two-stage framework and a curated dataset containing 9,163 *train*, 1,856 *val*, and 2,031 *test* hand instances, along with a subset of "bigger hand instances" for evaluation. The goal is to facilitate human visual recognition, including sign language and gesture interpretation, and temporal analysis.

The objective of this work is to detect and localise human hands in still images. This is a tremendously challenging task as hands can be very varied in shape and viewpoint, can be closed or open, can be partially occluded, can have different articulations of the fingers, can be grasping other objects or other hands, etc. Authors' motivation for this is that having a reliable hand detector facilitates many other tasks in human visual recognition, such as determining human layout and actions from static images. It also benefits human temporal analysis, such as recognizing [sign language](https://www.cs.cornell.edu/~dph/papers/buehler08.pdf), gestures and activities in video.

In this work authors propose a detector using a two-stage hypothesize and classify framework. First, hand hypotheses are proposed from three independent methods: a sliding window hand-shape detector, a sliding window context-based detector, and a skin-based detector. The sliding window detectors employ the Felzenszwalb et al. [Object detection with discriminatively trained part](https://cs.brown.edu/people/pfelzens/papers/lsvm-pami.pdf)’s part based deformable model with three components. Then, the proposals are scored by all three methods and a discriminatively trained model is used to verify them. The three proposal mechanisms ensure good recall, and the discriminative classification ensures good precision. In addition, authors develop a new method of non-maximum suppression based on [super-pixels](https://citeseerx.ist.psu.edu/viewdoc/download;jsessionid=033C2E617C9CC30B5455051D62B17950?doi=10.1.1.153.4065&rep=rep1&type=pdf). Figure below overviews the detector.

<img src="https://i.ibb.co/pRPKp4w/Screenshot-2023-10-12-075305.png" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">Overview of the method. (a) Original image. (b) Some of the hypotheses proposed by hand and context detector. Bounding boxes in ‘Red’ are proposed by the hand detector and ‘Green’ bounding boxes are proposed by the context detector. (c) Skin detection and hypotheses generation. (d) Super-pixel segmentation of the image with combined hypothesised bounding boxes from the three proposal schemes. Using super-pixel based non-maximum suppression (NMS), overlapping bounding boxes are suppressed. (e) Final detection after post-processing.</span>

Authors have collected a comprehensive dataset of hand images from various public image [sources](http://www.robots.ox.ac.uk/~vgg/data/hands/). In each image, all the hands that can be perceived clearly by humans are annotated. The annotations consist of a bounding rectangle, which does not have to be axis aligned, oriented with respect to the wrist.

<html>
<head>
    <style>
        table {
            border-collapse: collapse;
            width: 50%;
            margin: 20px;
        }

        th, td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
        }

        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <table>
        <tr>
            <th></th>
            <th>Training Set</th>
            <th>Validation Set</th>
            <th>Test Set</th>
        </tr>
        <tr>
            <td># hand instances</td>
            <td>9163</td>
            <td>1856</td>
            <td>2031</td>
        </tr>
        <tr>
            <td># bigger hand instances</td>
            <td>2861</td>
            <td>649</td>
            <td>660</td>
        </tr>
    </table>
</body>
</html>

<span style="font-size: smaller; font-style: italic;">Statistics of hand dataset. A hand instance is ‘big’ if area of the bounding box is greater than 1500 sq. pixels. Bigger hand instances are used for experimental evaluations.</span>
