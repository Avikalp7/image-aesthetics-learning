# Image.Aesthetics.Learning

Code and data for my paper at ACM SIGIR 2017 - *Social Media Advertisement Outreach: Learning the Role of Aesthetics* [[pdf]](https://arxiv.org/abs/1705.02146)
***

![](./SIGIR2017_Poster.jpg)

## Aesthetic Feature to Engagement Learning on Photnet Data

### Photo.Net Dataset

This dataset contains 20,278 images, which have each received at least 10 ratings, along with their average score ratings. 
The links for the datsets can be found [here](http://ritendra.weebly.com/aesthetics-datasets.html)

Due to copyright issues, the image data cannot be re-distributed and is the same reason we do not distribute the original images.

To scrape these images, use the script "*PhotonetImageScraping.py*" in the directory "ScraperScripts"

### Feature Extraction and Learning

Since our final aim is learning on interpretable image features that easily translate to modern editing tools, we utilize the 56 hand-crafted features designed and explained in [this](http://infolab.stanford.edu/~wangz/project/imsearch/Aesthetics/ECCV06/datta.pdf) paper for learning the role in user engagement.

The script for features - "*selected_features.py*" resides in the "FeatureExtraction" directory, where given the folder containing your images of interest, the feature vector for each is computed and the feature matrix is saved as a *.npy* file.

![](./images/sigir1.png)


### Getting the Advertisement Data and Learning

We build a data set of 8,000 image based promotional tweets by scraping Twitter profiles of 80 different corporations. These corporations are particularly active on Twitter and have between 36,000 (@Toblerone) to 13 million (@PlayStation) followers, and 3,000 (@tictac) to 753,000 (@united) tweets. We select these corporations from across 20 broad categories such as retail, fast food, automobiles etc. to account for the diversity in promotional image representation.

Please refer to the file *CorporateData.pdf* in the "Data" directory for more information.


We scrape such image based promotional tweets along with their likes count, retweets count, date and time, page followers, page tweets and tweet text from the Twitter API for each corporation page, in proportion to their total number of tweets.

Once again, due to copyright issues, the image data cannot be re-distributed and is the same reason we do not distribute the original images.

However, the entire information about our 8000 image data-set, along with the meta-data and extracted bias-related information is readily available for use :v: , please refer to the file *TwitterData.csv* for the same. You can use the *TwitterScraping.py* script, or a simple url-image downloader with the urls in the csv to rebuild the entire data-set. Once you have the data-set, you can contact me regarding the test-train splits (for the binary classification task mention in the experiment section).

#### Description of columns of *TwitterData.csv*

Column 1: Twitter Handle

Column 2: Image Retweets

Column 3: Image Favorites/Likes

Column 4: Page Followers

Column 5: Page Following

Column 6: Location

Column 7: Image URL

Column 8: Tweet Text

Column 9: Tesseract OCR Text

Column 10: Correcltly Downloaded? (Local)

Column 11: [NeuralTalk2](https://github.com/karpathy/neuraltalk2) Text


## Feedback
Feedback is greatly appreciated. If you have any questions, comments, issues or anything else really, [shoot me an email](mailto:avikalp22@iitkgp.ac.in).

All rights reserved.

