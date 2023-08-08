# Sample DASS-42 data

data_filtered_1000.csv is a random sample of 1000 participants from the original DASS-42 dataset. In generating the random sample from the original dataset, care was taken to ensure that the proportions of demographics closely mirror those present in the original dataset. This sampling approach preserves the distribution of key demographic factors, such as age group, gender, and location, maintaining a representative subset of the data.

The original DASS-42 dataset was derived from a large sample of participants worldwide (N = 31,715) and scored the anxiety levels of each participant according to the DASS manual.

The dataset includes each participant's response to each question on the DASS-42 questionnaire, and the participant's corresponding anxiety score and status. The dataset also includes each participant's demographic information, such as age, gender, and location.

The file '1_data_filtering_and_preprocessing.ipynb' will read the sample dataset, encode age, gender, and location into categories, and filter out participants who fulfill any one of the following criteria: 1. Neither Female nor Male, 2. Not an adult 3. Missing region 4. Are from a region with strict privacy laws. 