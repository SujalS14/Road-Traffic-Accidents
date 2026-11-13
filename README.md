# Road-Traffic-Accidents

## Business Problem
This project analyzes road traffic accident data to
identify dangerous driving conditions and help an
autonomous vehicle company prioritize data collection.

## Dataset
Addis Ababa Road Traffic Accident Dataset

## Business Questions
1. What is the distribution of accident severity across different days of the week?
2. What are the top 5 most common causes of accidents?
3. How does the driver's driving experience impact the count of serious or fatal accidents (where Accident_severity is 'Serious Injury' or 'Fatal injury')?
4. What types of vehicles are most commonly involved in accidents?
5. What is the distribution of accidents by driver's age band?

### Week 2 SQL Results

1. **Lighting conditions:** Darkness with no lighting had the highest share of serious or fatal crashes, at **28.1%** of 192 recorded crashes.
2. **Weather data gap:** Fog or mist (10 crashes), raining and windy conditions (40), and snow (61) had the fewest recorded crashes, so they would provide the fewest examples for a model trained on this dataset.
3. **Weather and severity:** Among named weather conditions, windy conditions had the highest share of serious or fatal crashes (**16.3%**), compared with **16.0%** in normal conditions.
4. **Junction types:** Crashes in the **Other** junction category had the highest share of serious or fatal injuries, at **19.1%** of 445 recorded crashes.
5. **Crash frequency by hour:** The **5:00–5:59 p.m.** hour had the most recorded crashes, with **1,228**.

## Model Results
On the held-out 20% of crashes, the model achieved 53.0% accuracy, compared with 84.6% for always predicting slight injury.
The model correctly identified 198 of 380 serious or fatal crashes (52.1%) but failed to identify the other 182.


## Tableau Dashboard
![Road Traffic Accident Risk Analysis Dashboard](Road%20Traffic%20Accident%20Dataset%20of%20Addis%20Ababa%20City/tableau/tableau_dashboard.png)

## Recommendations
Looking at the accident data analysis, the company should try to prioritize collecting more training data in these following situations:
1. Bad lighting conditions: Crashes that happen in the darkness without street lighting have a higher percentage of serious or fatal injuries, which makes these conditions important to study more.
2. Weather condiitons with limited data: Collecitng more information about weather conditions with lesser accidents that are recorded could help address the different gaps in the dataset.
3. High-severity junction types: Getting more data at junction types which are associated with higher percentages of serious or fatal crashes could help improve the model's predictions.

## Lessons Learned
Throughout this project, we learned how to use Python, SQL, Tableau, and Excel to analyze accident data and identify patterns. One of the biggest challenges was building the machine learning model and understanding its predictions. If we continued working on this project, we would focus on improving the model's accuracy and its ability to identify serious or fatal crashes.
