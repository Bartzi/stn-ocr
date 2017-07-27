# Tools for SVHN

This folder contains several tools that can be used for working with the SVHN dataset.

The first step is to download the orignal SVHN data from [here](http://ufldl.stanford.edu/housenumbers/), make sure to download **Format 1**.

`svhn_dataextract_tojson.py` is a tool for extracting the SVHN groundtruth from the provided matlab files and save them as a JSON file.
You could use it like that:
    
    python svhn_dataextract_tojson.py -f <path to digitStruct.mat> -o <path to output.json>

`create_svhn_dataset_4_images.py` is a script that can be used to create a dataset with images always containing 4 SVHN house numbers in a regular grid.
You could, for instance, create such a dataset by issuing the following command:
    
    python create_svhn_dataset_4_images.py <path to svhn image root dir> <path to svhn gt in json format> <destination directory> <number of samples to create> <max length of numbers to consider (e.g. 3 for all number between 0 and 999)>

You can use `python create_svhn_dataset_4_images.py -h` for an overview of all possible command line parameters.


`create_svhn_dataset.py` is a scriot that can be used to create a dataset containing SVHN house number crops that are scattered randomly accross the input image. The script allows you to choose how many house number crops to use, how far they shall be scattered and how long each house number can be at max. You could use it like that:
    
    python create_svhn_dataset.py  <path to svhn image root dir> <path to svhn gt in json format> <destination directory> <number of samples to create> <max length of numbers to consider (e.g. 3 for all number between 0 and 999)>

You can use `python create_svhn_dataset.py -h` for an overview of all possible command line parameters.
