# NLI_hackathon17
The code used in the NLI hackathon, 23-24/11/2017 for generating image metadata: face recognition, emotion deteection, age&amp;gender estimation, object detection, year of picutre estimation &amp; textual description

## Algorithms:

- Face recognition: 
  - Extracted facess locations with the HOG algorithm
  - Extracted deep features (Using a convolutinal Neural Net) for each face
  - Used K-Nearest Neighbors (with supporting ball-tree structure) to find similar faces in our data.

- Age,Gender&Emotion, Object recogintion: with a CNN trained for each metada field.

- Textual Description: Using the above meta-data and the nlp library simplenlg.

## Team members:

- Itamar Shenhar itamar8910@gmail.com
- Gil Maman Gil.maman.5@gmail.com 
- Dani Rubin sbh.ruchi@gmail.com 
- Tom Guter tomyguter@gmail.com 
