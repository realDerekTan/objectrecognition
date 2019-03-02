# objectrecognition
This is a project that I built using Google's Inception-V3 to identify objects in images in the ImageNet dataset.

**Navigation**
Objectrecognition.py is all that you really need. I automated the process for you, so it automatically creates the directories needed.
The other files are for supporting functions, download them and put them in the same folder/directory as objectrecognition.py

**How To Use**
Download all the files and put them into a single folder.
Run objectrecognition.py and it will create an images directory, as well as download the Inception-V3 model.
Download the images you want to predict and put it in the images folder.
Replace the image variable at the bottom with the name of the image you want to precict.
Currently it can only predict one images at a time, working on being able to process more than one in the future.
