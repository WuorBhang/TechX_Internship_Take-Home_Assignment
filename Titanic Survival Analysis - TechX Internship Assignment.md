> # Titanic Survival Analysis - TechX Internship Assignment

This repository contains a detailed analysis of the Titanic dataset, completed as a take-home assignment for the TechX internship program. The project demonstrates fundamental data science skills, including data cleaning, exploratory data analysis, visualization, and predictive modeling.

## Project Overview

The objective of this assignment is to analyze the Titanic passenger dataset to identify factors that influenced survival rates and to build a logistic regression model to predict passenger survival. The analysis is presented in a Jupyter Notebook (`Titanic_Survival_Analysis.ipynb`).

### Key Questions Addressed:
- What were the survival rates based on passenger class and gender?
- What were the most important factors in determining survival?
- How accurately can a logistic regression model predict survival?

## Key Findings

The analysis revealed several key insights into the factors affecting survival on the Titanic:

| Category | Finding |
| :--- | :--- |
| **Gender** | Female passengers had a significantly higher survival rate (**74.20%**) compared to male passengers (**18.89%**). This confirms the "women and children first" protocol was largely followed. |
| **Passenger Class** | First-class passengers had the highest survival rate (**62.96%**), followed by second-class (**47.28%**) and third-class (**24.24%**). This suggests that wealth and social standing played a crucial role in survival. |
| **Model Performance** | The logistic regression model achieved an accuracy of **79.89%** on the test set, with a balanced precision of **77.14%** and recall of **72.97%**. |
| **Feature Importance** | **Sex** and **Passenger Class** were the two most influential predictors of survival, with coefficients of **-2.58** and **-0.96**, respectively. |

### Visualizations

| Survival Rate by Passenger Class | Survival Rate by Gender |
| :---: | :---: |
| ![Survival Rate by Passenger Class](https://private-us-east-1.manuscdn.com/sessionFile/csR1DXfKKrAYFsPTlfL75h/sandbox/jbqaXYlAhQYT339dwaM9FL-images_1766485779325_na1fn_L2hvbWUvdWJ1bnR1L3N1cnZpdmFsX3Zpc3VhbGl6YXRpb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvY3NSMURYZktLckFZRnNQVGxmTDc1aC9zYW5kYm94L2picWFYWWxBaFFZVDMzOWR3YU05RkwtaW1hZ2VzXzE3NjY0ODU3NzkzMjVfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjFjblpwZG1Gc1gzWnBjM1ZoYkdsNllYUnBiMjQucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=P13uXElttpuxS5na3hMoPLnrh3y9dZWOSucZI72F3Jk0kQCl6VZyRa5MNob0OZt13r1pVojNhSPBUylegC7Gfh34AmBAMm8ByydwRzgHXPZ4jwkfpKpzoPDG82Dvs1W~Mb9vpRk0gh1GbR9nS1IiiZqjDwgfK1Y0ATHFO0ZoxrVPdqeKuR8gc-7vJHIj-ZiqB0u48HSupdN~LSpKflKOHf8K7yvxuIdlvMrOhh8v2WAPFXGAWNoj22gASOufOYCKqHlHHecKtCc4vkNqqkgfbHxg0ran5iF~mZ49ZzY0FcNzBSJyT74Eogpy4Kk2McMt3vylbSj2SH2igxG3Au3eEQ__) | ![Survival Rate by Gender](https://private-us-east-1.manuscdn.com/sessionFile/csR1DXfKKrAYFsPTlfL75h/sandbox/jbqaXYlAhQYT339dwaM9FL-images_1766485779326_na1fn_L2hvbWUvdWJ1bnR1L3N1cnZpdmFsX3Zpc3VhbGl6YXRpb24.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvY3NSMURYZktLckFZRnNQVGxmTDc1aC9zYW5kYm94L2picWFYWWxBaFFZVDMzOWR3YU05RkwtaW1hZ2VzXzE3NjY0ODU3NzkzMjZfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwzTjFjblpwZG1Gc1gzWnBjM1ZoYkdsNllYUnBiMjQucG5nIiwiQ29uZGl0aW9uIjp7IkRhdGVMZXNzVGhhbiI6eyJBV1M6RXBvY2hUaW1lIjoxNzk4NzYxNjAwfX19XX0_&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=NL2nkEkVwibDptYKpGrFIp9FQDOE9BJRtxd84LapVrDPp53ykUC3oepxaN8HXH~rfK14P04jzpwSmr9sjdStpMWUb3~ktBaOm~3fT0vEIn4-6wJTKnOizCabETUnZdEXhoeTkmiNLj0zIgrTZo0OeBmueYsy-R5B3mPA4NHtwOOB6MIz8lWwdgG9hZYJsH6QJgK0yH~nT4MAmq-WREtOD6ATk3xplGvLV9tbtrGCvuadZMF-wuM1AKb0wuSvK57cBxWK69hhRKjVS4hbGD0WNP-fpVLkijOa9c9Uv6q7RJ0aHyXjA8XiCJgSyoyiWZRVPrM9lg2kGaheKORLxF3JlQ__) |

| Confusion Matrix | Feature Importance |
| :---: | :---: |
| ![Confusion Matrix](https://private-us-east-1.manuscdn.com/sessionFile/csR1DXfKKrAYFsPTlfL75h/sandbox/jbqaXYlAhQYT339dwaM9FL-images_1766485779327_na1fn_L2hvbWUvdWJ1bnR1L2NvbmZ1c2lvbl9tYXRyaXg.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvY3NSMURYZktLckFZRnNQVGxmTDc1aC9zYW5kYm94L2picWFYWWxBaFFZVDMzOWR3YU05RkwtaW1hZ2VzXzE3NjY0ODU3NzkzMjdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyTnZibVoxYzJsdmJsOXRZWFJ5YVhnLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=RC03mq8DXsK-PODC~q-EB8tX9K-Pix12jfcOp~Hb4uB12trdo3q1v90zkegIAZGj7pEyWnwQpL8zULmzdDh9tMwXU32AW1eYdklxXnfR~1OfzkRjgrDsgHVkYk1gISnbBpbWTB1xEtDS2PIc9wjOaiP6vZQLItazrUH~t0K4F1sooAveHX9PQ9yR4XPAdrvYY8GJikEgn25PEA9v2rzApIXdex8P68VWJevZ6ZgzjxBwDmBlmQ-l1TAbdb1KA2i-4~OYRIWE8zjcyLbM3eykxpd-YcN73spvI9QpKAOU-4qh3909B3Z-SpZbCtBk3u3Jgz8KIXghuJPUjgxNni4T2w__) | ![Feature Importance](https://private-us-east-1.manuscdn.com/sessionFile/csR1DXfKKrAYFsPTlfL75h/sandbox/jbqaXYlAhQYT339dwaM9FL-images_1766485779327_na1fn_L2hvbWUvdWJ1bnR1L2ZlYXR1cmVfaW1wb3J0YW5jZQ.png?Policy=eyJTdGF0ZW1lbnQiOlt7IlJlc291cmNlIjoiaHR0cHM6Ly9wcml2YXRlLXVzLWVhc3QtMS5tYW51c2Nkbi5jb20vc2Vzc2lvbkZpbGUvY3NSMURYZktLckFZRnNQVGxmTDc1aC9zYW5kYm94L2picWFYWWxBaFFZVDMzOWR3YU05RkwtaW1hZ2VzXzE3NjY0ODU3NzkzMjdfbmExZm5fTDJodmJXVXZkV0oxYm5SMUwyWmxZWFIxY21WZmFXMXdiM0owWVc1alpRLnBuZyIsIkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc5ODc2MTYwMH19fV19&Key-Pair-Id=K2HSFNDJXOU9YS&Signature=mSO-ea~U3v0RHU-iwbzXAjitRPuRKUCVXEoKrnEdhuaGNB1dzrcOHxvDP8znmIOzU2p1vDYVLoWSZHVyzcvW8vF0Y61wzlJNGubuRaWCNunenWJ6AO26ZOrw4uFkvXwTqmGgRmkwSaUPWJLNDeV7N8-ACQwtKrJjoVy6d35XJH2mcS8oF5MVggcLClazjO6tfuASZiFg2XM35EauENhYMl~aGNs2w-moQnMRpHcPkDzhj3k13GL7MHR6wB6ISm5K3sZpNbYhPPeoecHkpocye0pKs9kJj-wamktowlBKdX7BIOMgKgUyvjEn-YJfAMKr1STTZL7CP9RXrYEoCR62qA__) |

## Getting Started

To run this analysis on your local machine, follow these steps:

### Prerequisites

- Python 3.x
- Pip (Python package installer)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/WuorBhang/TechX_Internship_Take-Home_Assignment.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TechX_Internship_Take-Home_Assignment
   ```
3. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

There are two ways to run the analysis:

1.  **Jupyter Notebook:**
    Open and run the `Titanic_Survival_Analysis.ipynb` file in Jupyter Notebook or JupyterLab to see the step-by-step analysis with detailed explanations and outputs.

## File Descriptions

- `titanic.csv`: The raw dataset used for the analysis.
- `Titanic_Survival_Analysis.ipynb`: A Jupyter Notebook with a detailed, step-by-step walkthrough of the analysis.
- `survival_visualization.png`: A bar chart showing survival rates by passenger class and gender.
- `confusion_matrix.png`: A heatmap visualizing the performance of the logistic regression model.
- `feature_importance.png`: A bar chart showing the coefficients of the features in the logistic regression model.
- `README.md`: This file, providing an overview of the project.

## Author

- **Name**: WUOR BHANG
- **Email**: uhuribhang211@gmail.com
