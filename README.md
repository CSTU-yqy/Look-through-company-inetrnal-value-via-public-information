bad news helps learn more
==============================

Project Organization
------------

    ├── LICENSE
    ├── .pylintrc                       <- Python style guidance
    ├── README.md                       <- The top-level README for developers using this project.
    ├── env                             <- CI configuration
    ├── data                            <- Our data sources
    │   ├── cnki_txt                            
    │   │    ├── data                           
    │   │    └── data_description.md            
    │   ├── company_metrics                     
    │   │    ├── data                           
    │   │    └── data_description.md            
    │   └── README.md                   <- Introduction to our data sources
    │
    │
    ├── models                          <- Trained and serialized models, model predictions, model summaries
    │   ├── DL_news_classifier          <- Popular deep learning methods for chinese news classification(BERT,ERNIE,TEXTCNN,RCNN...)         
    │   └── Self-built_news_classifier  <- The classifier I built for my dataset
    │
    ├── notebooks                       <- Jupyter notebooks I used for analysis
    │   └── Staged_results_202202.ipynb      
    │                         
    │
    ├── references                      <- Data dictionaries, manuals, and all other explanatory materials.
    │   └── REAMME.md      
    │
    ├── reports                         <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── reports                     <- Reports for staged results
    │   └── figures                     <- Graphics and figures to be used in reporting
    │
    ├── requirements.txt                <- The requirements file for reproducing the analysis environment, e.g.
    │                                   generated with `pip freeze > requirements.txt`
    │
    ├── setup.py                        <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                             <- Source code for use in this project.
    │   ├── __init__.py                 <- Makes src a Python module
    │   │
    │   ├── data                        <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features                    <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                      <- Scripts to train models and then use trained models to make
    │   │   ├── __init__.py                 
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization               <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini                         <- tox file with settings for running tox; see tox.testrun.org


--------

## WHY BAD NEWS?
I believe news data can help us learn more about the real situation of the company. The company news released by authoritative media shields the noise caused by false information and subjective inference, and can most directly show the micro details of the company. Good news, bad news, even fake news can be useful for  information gain. How to choose the cut-in angle is the most efficient?

![picture 1](./reports/figures/where%20the%20ideas%20com%20from.png)

As can be seen from the above figure, most of points are in the area above the diagonal, it means the time point when the relevant indicators of negative news send out signals is earlier than the time point when the company officially announces the major changes. I guess maybe it is a good choice to cut in from the perspective of bad news.

## WHERE CAN WE GET OUR DATA SOURCES AND WHY WE CHOOSE IT?
1. About CNKI Full-text Database of Important Chinese Newspapers
CNKI is the most authoritative academic data platform in China
The database contains 658 newspaper(168 central level + 490 province level)
from 31 province, covering all industries and listed companies

2. About how I get our data sources
CNKI has a good defense against web crawlers, so I overcame a lot of technical difficulties to get this data soures
I provide a demo of our crawlers(./crawler/demo.py)

3. Statistical description of the raw data
------------
    from province prospective
        coverage of province :                              31 / 31
        newspaper  amount in one province(min) :            4(Hainan)
        newsspapers amount in one province(max) :           30(Sichuan)
        newsspaper amount in one province(mean) :           15.8
    from article prospective
        article amount total :                              432812
        sentence amount total :                             35948282  
    from company prospective
        coverage of company :                               4101 / 4602
        article amount of one company(min) :                1
        article amount of one company(max) :                9342
        article amount of one company(mean) :               105.6
        article amount of one company(25% quantile) :       5
        article amount of one company(50% quantile) :       15
        article amount of one company(75 quantile) :        54
        article amount of one company(std) :                399.6
------------

4. Statistical description of the data I use to do empirical study
------------
    from province prospective
        coverage of province :                              31 / 31
        newspaper  amount in one province(min) :            4(Hainan)
        newsspapers amount in one province(max) :           30(Sichuan)
        newsspaper amount in one province(mean) :           15.8
    from article prospective
        article amount total :                              432812
        sentence amount total :                             35948282  
    from company prospective
        coverage of company :                               4101 / 4602
        article amount of one company(min) :                1
        article amount of one company(max) :                9342
        article amount of one company(mean) :               105.6
        article amount of one company(25% quantile) :       5
        article amount of one company(50% quantile) :       15
        article amount of one company(75 quantile) :        54
        article amount of one company(std) :                399.6
------------

5. All compamy metircs data I select
------------
    ---------------------------------------------------------------------
    Comapny metirc name                       | Category
    ---------------------------------------------------------------------
    CLOSE                                     | Quotes
    ---------------------------------------------------------------------
    Equity ratio                              | Solvency
    ---------------------------------------------------------------------
    EPS                                       | Profitability
    ---------------------------------------------------------------------
    Main business ratio                       | Profitability
    ---------------------------------------------------------------------
    Current ratio                             | Solvency
    ---------------------------------------------------------------------
    All Assets Cash Recovery Ratio            | Cash flow
    ---------------------------------------------------------------------
    Equity Multiplier                         | Solvency
    ---------------------------------------------------------------------
    ROA                                       | Profitability
    ---------------------------------------------------------------------
    ROE                                       | Profitability
    ---------------------------------------------------------------------
    Quick ratio                               | Solvency
    ---------------------------------------------------------------------
    Turnover                                  | Quotes
    ---------------------------------------------------------------------
    Updown                                    | Quotes
    ---------------------------------------------------------------------
    Cash ratio                                | Solvency
    ---------------------------------------------------------------------
    Cash turnover ratio                       | Operating capacity
    ---------------------------------------------------------------------
    Sales margin                              | Profitability
    ---------------------------------------------------------------------
    Gross profit margin                       | Profitability
    ---------------------------------------------------------------------
    Accounts Payable Turnover                 | Operating capacity
    ---------------------------------------------------------------------
    Accounts Receivable Turnover              | Operating capacity
    ---------------------------------------------------------------------
    Working capital/total assets              | Early warning indicators
    ---------------------------------------------------------------------
    Working capital turnover                  | Operating capacity
    ---------------------------------------------------------------------
    ZVALUE                                    | Early warning indicators
    ---------------------------------------------------------------------
------------

## HOW TO FIND BAD NEWS?
I try deep learning models and self-built models. Popular DL(deep learnig) models did not have a good performance on our dataset, because chinese word in the financial context has this uniqueness, I need to build a lexicon and our own model to slove this problem.
I mainly did the three steps to build our own classifier(I cannot show model details before I finish this working paper, but I provide demo in ./models/Self-built_news_classifier)
### clean

![picture 2](./reports/figures/clean%20data.png)

### pretrain

![picture 3](./reports/figures/pretrain.png)

### train

![picture 4](./reports/figures/train.png)

### model perfomance

![picture 5](./reports/figures/model%20performance.png)

![picture 10](./reports/figures/model%20performance.png)

## HOW BAD NEWS WORKS?

Inspired by incredible work by Calomiris C W & Mamaysky H(2019), I guess I could classifying bad news using unsupervised learning method and infer topics by drawing word clouds.

### cluster result

![picture 6](./reports/figures/cluster%20result.png)

It can be seen that although the number of negative news is large, it still shows a trend of gathering in the top 9 categories. I can't directly show what each type of negative topic is before our paper is finished, but it can be clearly seen that although there are various types of negative news, their classification is actually very small.

Now I have different types of bad news and different types of company metrics, I could do something cool. Maybe I could answer the question---"Bad news can bring inforamtion gain but how does it work?"

I believe that negative news is the information leakage of the deterioration of the microscopic situation of the enterprise, so different types of negative news represent the deterioration path of the enterprise in different ways, it should appear that the measurement indicators of different aspects of the enterprise have good feedback on the indicators based on the synthesis of negative news. However, their sensitivities will vary due to different pathways of action

### Statistical test results

#### legend

![picture 7](./reports/figures/Statistical%20test%20legend.png)

#### result (E.g)

![picture 8](./reports/figures/some%20potentially%20relevant%20variables_3.png)

### Sensitivity of different indicators

![picture 9](./reports/figures/how%20bad%20news%20helps%20learn%20more---sensitivity.png)

I am testing if results of the experiment are consistent with my conjecture. My working paper will be finished this year, so stay tuned！！！



------------
    Dream is a validation set.
    If we are underfitting, learn more to be a "complicated model".
    If we are overfitting, try more to "sample" or just learn to be "simple".
    Training is not to limit but to ascend, both for modeling and for our lifes. 
                                                                                  
                                                              ---qianyu 20220531
------------
