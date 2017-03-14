
---

**Vehicle Detection Project**



[//]: # (Image References)
[image1]: ./examples/example.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  
While starting this project I had saw discussions in the Slack channels of people using full CNN approaches such as YOLO and SSD to complete their project. I had a look around the internet on YOLO and I became really interested in this supposedly fast detection method so I decided that I want to learn to implement these instead as well.

After doing some research on the internet. Between YOLO, SSD and YOLOv2 (or YOLO9000), YOLOv2 seems to be claiming the fastest FPS on object detection with similar Mean Average Precision so I decided I will give the latest YOLOv2 a go. I also realized training a new network from scratch will take a lot of time and effort, the 2nd challenge is how to get it working in Keras or Tensorflow. The YAD2K project by Allanzelener on github seemed to be the easiest to work with, the code was recent and I could roughly understand what he was trying to do. Just reading the paper, it was challenging to understand how to interpret the last layer except it was a convolutional layer of 13x13 grids predicting "5 boxes at each grid location with 5 coordinates each and 20 classes per box (13x13x125)" as in the paper. YAD2K also used Tensorflow to do the interpretation, I thought I'd reimplement this using just numpy to understand the intricacies of this last layer. However after spend time reverse engineering this I got stuck at the point where I needed to filter out the boxes and I couldn't find a nice way of implmenting a tf.boolean_mask for a 5 dimensional tensor in numpy. In a moment of frustration I decided just make use of the functions from YAD2k for interpreting the last layer with some adjustments to only filter for cars.

My code can be found in [test-notebook-tf.ipynb](./test-notebook-tf.ipynb)

YAD2K did the heavy lifting and provided a python library to build a Keras model from the Darknet weights. I tried both the tiny-yolo-voc and yolo-voc, I used yolo-voc as that had a higher mAP.

For the model then it became an easy task
```
model_path = "./model_data/tiny-yolo-voc.h5"
yolo_model = load_model(model_path)
```
The model I used can be downloaded from here: https://drive.google.com/open?id=0BxrR4D9fa_NkR2FfRVIwNW5RZDQ

You'll see in cell 6 & 7 of my notebook that I've taken 4 functions from the YAD2k project to be part of my pipeline.  
"yolo_head" splits the last layer into:
- 1x13x13x5x2 for the x & y for the 5 different box anchors at each grid location
- 1x13x13x5x2 for the width and height for the 5 different box anchors
- 1x13x13x5x1 for the box confidence
- 1x13x13x5x20 for the probabilities for the 20 classes in the VOC2007 set

"yolo_boxes_to_corners" calculates the box coordinates on the image  
"yolo_filter_boxes" filters out predictions below a certain threshold  
"yolo_eva" builds the pipeline for tensorflow using the above 3 functions  

Given this my pipeline is as simple as below with an additional logic to skip anything that's not cars and then drawing a rectangle.
```
def pipeline(image):
    image_data = cv2.resize(image,(416,416)).astype("float32")

    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)

    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.shape[0], image.shape[1]],
            K.learning_phase(): 0
        })

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        if predicted_class != 'car':
            continue

        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        cv2.rectangle(image,(left, top),(right, bottom),(0,255,0),3)
    return image
```


---

### Video Implementation

####1.
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
Initial results showed multiple bounding boxes on certain frames, after looking at the results and increasing my threshold to predictions of 0.6 it solved my problem.


### Here is the output of test image 5 after being through the pipeline:
![alt text][image1]

### Performance
I just ran this on my laptop and I'm getting a performance of 1.6s per frame on my CPU. I've not tried it on an AWS GPU instance yet but I expect it'll be much faster. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
- This pipeline will work for cars, if we expand it to start bounding traffic signs or lights then I'll have to train it from scratch again or use a different model or do some transfer training. I've yet to know it enough to do this. However after going down this path I think a FCNN approach is the right way to go for this problem set.
- Understanding the last layer took some time but it was a good learning experience
- Would love to be able to implement this network from scratch, however just from reading the paper it seems like it will take a long time for me.
- Also still don't understand the intricacies of the research paper and would like to spend some more time on it
- I feel like there's still a lot for me to learn and experiment. Would love to be able to implement SSD as well.


###References
This piece was definitely done by leveraging the efforts and work of other people.
1. YOLO9000 Research Paper https://arxiv.org/pdf/1612.08242
2. https://github.com/allanzelener/YAD2K
3. Blog by Menxi Wu  https://medium.com/@xslittlegrass/almost-real-time-vehicle-detection-using-yolo-da0f016b43de#.fl1n9d4e1
4. YOLO website https://pjreddie.com/darknet/yolo/
