# 🚦 Toronto Collision Severity

Can we predict how badly someone gets hurt in a Toronto traffic collision from the conditions around it: speeding, alcohol, road type, weather, visibility, traffic control and who was involved?

Final project for the *Machine Learning* course (University of Toronto, SCS, 2022). **Team project**; this notebook is our working analysis.

## Data
Toronto Police Service **KSI (Killed or Seriously Injured)** collisions, published on the City of Toronto / TPS open data portals under the Open Government Licence – Toronto. The CSV is not included here; download `KSI.csv` from the TPS Public Safety Data Portal and put it next to the notebook.

## What's inside
1. **Cleaning:** dropped IDs, street names and coordinates; standardized messy missing values (`??`, `<Null>`, blanks); turned Yes/No flags into 0/1; encoded ordered categories (injury None → Fatal, road class, surface condition, visibility, age bands).
2. **Feature engineering:** month, year and weekday from the collision date.
3. **Model shoot-out:** Decision Tree, AdaBoost, SVM, KNN, Logistic Regression, SGD, Naive Bayes, Random Forest and a hard-voting ensemble, compared with cross-validation, confusion matrices and macro F1.
4. **Second pass:** a richer cleanup (keeping pedestrian, cyclist, vehicle-type flags), a correlation heatmap, then **grid-search tuning** of the Decision Tree and Random Forest.

## What I learned
- **Accuracy lies on imbalanced data.** Most records are "no/minor injury", so a model can look accurate while missing the severe cases. Macro F1 and the confusion matrix tell the real story.
- **Cleaning is most of the work.** The raw export had half a dozen spellings of "missing" and dozens of free-text categories; most of the effort went into making it usable.
- **Try many models, then tune the promising ones.** Comparing nine models first made it clear where tuning time was worth spending.

## Tech
Python · pandas · scikit-learn · seaborn · matplotlib
