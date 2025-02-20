# CS 6476 Project 2: SIFT Local Feature Matching and Camera Calibration

## Getting started

- See Project 0 for detailed environment setup. This project's environment is set up similarly, and will create a conda environment called `cv_proj2`.
- Ensure that you are using the environment `cv_proj2`, which you can install using the install script `conda/install.sh`. 
- Alternative way to set up the environment:
  1. make sure you are in your Project2/conda folder, then enter those commands in the terminal.
  2. ```mamba env create -f environment.yml```
  3. ```conda activate cv_proj2```
  4. go back to your Project2 folder, ```pip install -e .```

## Logistics

- Submit via [Gradescope](https://gradescope.com).
- All parts of this project are required for 6476.
- Additional information can be found in `docs/project-2.pdf`.

## 6476 Rubric

- +70 pts: Code (detailed point breakdown is on the Gradescope autograder)
- +30 pts: Report
- -5*n pts: Lose 5 points for every time you do not follow the instructions for the hand-in format

## Submission format

This is very important as you will lose 5 points for every time you do not follow the instructions.

1. Generate the zip folder (`<your_gt_username>.zip`) for the code portion of your submission once you've finished the project using `python zip_submission.py --gt_username <your_gt_username>`. It should contain:
    - `src/`: directory containing all your code for this assignment
    - `setup.cfg`: setup file for environment, do not need to change this file
    - `additional_data/`: (required for 6476) the images you took for Part 5, and/or if you use any data other than the images we provide, please include them here
    - `README.txt`: (optional) if you implement any new functions other than the ones we define in the skeleton code (e.g., any extra credit implementations), please describe what you did and how we can run the code. We will not award any extra credit if we can't run your code and verify the results.
2. `<your_gt_username>_proj2.pdf` - your report


## Important Notes

- Please follow the environment setup in Project 0.
- Do **not** use absolute paths in your code or your code will break.
- Use relative paths like the starter code already does.
- Failure to follow any of these instructions will lead to point deductions. Create the zip file by clicking and hand it in with your report PDF through Gradescope (please remember to mark which parts of your report correspond to each part of the rubric).
