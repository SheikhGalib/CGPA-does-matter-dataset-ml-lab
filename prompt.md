@AGENTS.md I have some instruction from teacher. I have raw data @raw-data\  - collected
  using google from University students. Target is find out what features actaully effect
  the most in academic performance and another pipeline - cgpa prediction from given
  features. We are told to use weeka for machine learning model testing with our data. But
  before that we need to clean the data. First update the Agents md file with clear
  instructions (i have just copy pasted teachers), our scope of the project, and what we are
  trying to do, where are the files. Then in python make a data cleaning pipeline notbook.
  In @cleaned-dataset\ my teammate Jaman already did some data cleaning using chatgpt. But
  as we need to verify our process and why we did something - we need a notebook and a  proper data cleaning pipeline (search online for similar problem and how they clean there
  data). The basic stuff i know - dropping fileds that are missing, dropping very sparse
  column if they actually does not mean any thing - Basically do Robust EDA, find hidden
  features. So, the notebook should have first the data set visualization, raw data, Then
  data distribution, then EDA with proper graphs and stuff ( not too much shit, only valid
  stuff, example to be - data skewness (if needed), data distribustion, missing fileds,
  columns, box plot, (i am just saying some name, you figure out online, whats best practise
  EDA for the problem we have). then after EDA, from EDA result, analyze which features are
  prominent in affecting the most in persons academic results. Then given some data, train
  a model which can predict persons cgpa on their given data (features). Code it, test it.
  And make a walkthough gudie on how to use and understand the notebook. The next step, how
  to use the data on weka. Update agents md to use markitdown skills for pdf reading (making
  pdf -> markdown for reading, token save) and use frontend slides skill for pptx
  generation. use noob-explain skill for explaining it to user (report, docs, walkthough
  guide step by step)


  We have some annomlies showing up. the accuracy is low. Our teacher told us - any two bands (we are using c0-3, four cgpa bands). the band and the neighbor having most of the values in confusion matrix, is reasonable but the further neighbors having a singnificant ammount is kinda weird, we have to figure out why. 
  # Checks -
  1. Check the presentation slide - presentation\What_Shapes_Student_CGPA_Ready.pptx we have made this with all the details
  2. Teacher told us to check 5 fold cross validation and tain, test split of the models and check if that improves anything.
  3. Check assigning different weights to the feautures and finding out if that changes and why is changed. 
  4. I have a huntch that some data might be filled just for the sack of form fillup - randomly filling up the data, as some data suggest there are a lot of prodeigies in kuet. There might be some outliers but too many is weird. Check presentation slide - 23-34, individual features corelation with cgpa. 
  5. Teacher told us to check only kuet and 4th semester, to figure out if the low accuracy is on the full wide dataset or from kuet. also check 4th semester only and similarly check the accuracy there. 
  6. Check the decision trees, there are three, are they actually mean anything or just random stuff. 
  7. propose the fixes to the above mention problems and porbable solution with justificaion of why that will work and verify that with results. 
  # Deliverables-
  1. A report in markdown inside docs folder with figures attached. You can use a figures folder inside the docs folder to link the markdown with all the results and figues and confussion matrix that we can verify. 
  2. The goal is to impove the accuracy and explaining the changes we did that made the accuracy and validate our study with proof and data. 
  3. The report should be comprehensive enough so that we can follow that and run it in weka and add the screenshot in presentation and update the confussion matrix and other data following the report
  4. So, add what to add in the presentaiton section in the report so that ai agents can fllow that during presentaiton making. And i does not have to teach same shit over again. 
  5. Also add, weka instruciton and the next steps as well. 

  okay tell me one thing, the cgpa holds more value, its the cummulative average of all the previous gpa. I have an idea, which is - 
  1. If Sgpa is like huge dip but cgpa high thats weird. Typically students with high cg like c3 are very sensitive with there cgpa and will do anything to boost there cgpa so them having sgpa very low like c0 is quite weird (from real life exprience even when they have major accidents or operations, sick then even, they get to give back log so the sgpa is not that down.), so can safely remove this very far gap sgpa and cgpa students to be bogus. But we have to report them, like the statistics, how may students have such outlier in the from and the band gaps. Again from a simple calculation we can see that - a student lets say have 3.8 up in sem 5, but did very poor on sem 6 like 2.8 then there should be a dent in the cgpa -> 3.63 (new sem 6 cgpa), so the band closes. SO, even if the student does poorly in last sem, it will make the band closer.
  2. For Low cgpa and high sgpa is still do-able, like a major comeback. Lets try that example, a student have cgpa 3.0 in sem 5 and got a 4 in sem 6, so overall cgpa in sem 6 = 3.167, so not so much increase, but 3.1 in sem 5, and got 4 in sem 6 will lead to cgpa = 3.25 in sem 6, so a band increase, so as we are doing a range, not all data will be up a band, but a good number of such come backs will have this pattern, being one band closer. 
  3. From your analysis, now check of that 455, how many we can safely keep, we don't want to through away data. 

  Okay, can you please update the actual slide ppt, presentation\What_Shapes_Student_CGPA_Ready.pptx with our new found outputs,
  # Don'ts-
  1. Please don't use red blocks on confusion matrix, use simple plain confussion matrices. 
  2. Don't use hard, sployy language, make it easy to read and understand. 
  3. Don't try to overengineer, we are just to show the data we collected and how the data is and what have we done to impove the results (our stratigeis)
  4. Don't use emojis they look ai and people tends to give less value to the work if found "Made with AI"
  # Instructions -
  1. Most of the formatting in the powerpoint is already done, please update the vlaues and graphs, tables, charts, confusion matrieces
  2. If you can then also update the weak screenshots in the slide. 
  3. Give me one final csv and arrf (weka specific file) that i can laod, and a instruciton on how to get the models results - so that I can verify them and take screenshots of the results to add in the presentations. 
  4. We are going with - 5 fold cross validation
  5. We are keeping gap <= 1 band of the 455 and droping the rest. (920 out of 1100 is still reasonable, not too much data loss)
  6. We are dropping "result satisfaction" feature, as it is subjective
  7. We will also have 80/20 train test split and its accuracy and keep it in hidden slides, if teacher asks, we can show teacher that. 
  8. Update the data in the slides where ever needed. 
  9. Add additional slides - for the feature analysis we did to drop the rows with band gap > 1, and how many data there were and which uni, which sem that was (typical statistics), so that we can actually justify our decisions. 
  10. Use simple and easy to understand language. 