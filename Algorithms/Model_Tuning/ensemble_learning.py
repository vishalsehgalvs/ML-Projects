# =============================================================================
# Ensemble Learning — The Crowd Wisdom Approach
# =============================================================================
# Core idea: instead of trusting one model's prediction, ask several models
# and go with the majority answer. One model can have a bad day. A crowd
# of models is much harder to fool.
#
# Real-world analogy: if you're unsure about a medical diagnosis, you
# get a second opinion, maybe a third. You trust the consensus, not
# just one doctor's call.
#
# Types of ensemble learning:
#
# 1. Bagging (Bootstrap Aggregating)
#    Train many models in parallel on random subsets of the data.
#    Each model votes — majority wins.
#    Example: Random Forest (many decision trees, each on a random subset)
#
# 2. Boosting
#    Train models in sequence. Each new model focuses harder on the
#    examples the previous one got wrong. They learn from each other's mistakes.
#    Examples: AdaBoost, Gradient Boosting, XGBoost
#    XGBoost is the most popular and usually the strongest performer.
#
# 3. Stacking
#    Train a mix of different models (e.g. Decision Tree + SVM + Logistic Regression)
#    as 'base learners'. Then train a second model (meta-learner) that takes
#    their predictions as input and makes the final call.
#    Think: three specialists each give their diagnosis, then a senior doctor
#    weighs their opinions and makes the final decision.
# =============================================================================