Research Question 1 
Bias & Fairness in Detection
Do hate speech detection models exhibit bias toward certain groups, or topics when classifying polarized tweets on X?

Explanation :
Hate speech detection models may unintentionally exhibit and show bias by disproportionately misclassifying content related to certain groups, or topics. This research question investigates whether models trained on polarized tweet datasets produce unequal error rates across different categories, such as specific identities, dialects, or forms of expression. For example, slang, reclaimed language, or minority dialects may be incorrectly flagged as harmful, while more subtle forms of hate may go undetected. Evaluating fairness involves analyzing false positives and false negatives across groups to identify systematic patterns of bias. This is important because biased models can reinforce existing inequalities and reduce trust in automated moderation systems.

Research Question 2
To what extent does incorporating hashtag information improve the detection of hate-related or polarized tweets compared to using text-only approaches?

Explanation :
Detecting hate-related or polarized tweets using only the textual content can be challenging, as meaning is often influenced by additional signals embedded within the tweet itself. Hashtags, in particular, can provide important contextual cues by indicating the topic, sentiment, or underlying intent of a tweet, even when the text alone appears neutral or ambiguous. This research question investigates whether incorporating hashtag information alongside tweet text improves model performance in identifying hate-related content. By comparing models trained on text-only features with those that include hashtags as additional inputs, the study aims to evaluate whether hashtags help capture implicit or community-specific language patterns. This is especially relevant on X, where hashtags are frequently used to signal alignment, amplify opinions, or embed subtle forms of polarization.
Research Question 3
How does the severe class imbalance in religion-based hate speech (only 0.51% of the dataset) affect the ability of a machine learning model to learn meaningful patterns for this category, and what preprocessing strategies can mitigate this without distorting the overall dataset distribution?

Explanation :
 The Religion category in our combined dataset contains only 319 tweets out of 61,945 — a severe minority that risks being completely ignored by most standard classifiers. This is not a random artifact but a reflection of how rare religion-based hate speech is in the three source datasets, which were collected using different keyword strategies. This question is interesting because it sits at the intersection of data imbalance, ethical representation, and model fairness. A model that never predicts "Religion" is technically accurate on this class most of the time, but practically useless for detecting it. We will investigate various different techniques for adjustment during the modeling phase and determine whether we can meaningfully improve recall for this minority class without significantly degrading performance on the majority classes.

