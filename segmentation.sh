#load base64 to images
python load_submissions_leak.py

# creates models
python run_segmentation.py -id 1
python run_segmentation.py -id 2
python run_segmentation.py -id 3
python run_segmentation.py -id 4
python run_segmentation.py -id 5
python run_segmentation.py -id 6

# apply models and save images
python inference.py
# fit the leak
python fit_leak.py