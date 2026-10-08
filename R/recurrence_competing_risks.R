# Recurrence after initial response, with death as a competing event (proposal: Data analysis).
# Run from the repository root after step 12:   Rscript R/recurrence_competing_risks.R
#
# Input: results/analysis_recurrence.csv (one row per responder at the 12-week landmark), written by
#   python -m pipeline.step12_incremental_value
#   recur_months = time from the landmark; recur_event: 0 = censored, 1 = recurrence, 2 = death
#   reference predictors + rad_score (LASSO cause-specific Cox score, fitted in Python on all responders;
#   its selection optimism is NOT included in the bootstrap below - report this as a limitation).
# Packages: install.packages(c("survival", "prodlim", "riskRegression", "cmprsk", "randomForestSRC"))
suppressPackageStartupMessages({
  library(survival); library(prodlim); library(riskRegression); library(cmprsk); library(randomForestSRC)
})
d <- read.csv("results/analysis_recurrence.csv")
ref <- setdiff(names(d), c("patient_id", "recur_months", "recur_event", "rad_score"))
d <- d[complete.cases(d[, c("recur_months", "recur_event", ref, "rad_score")]), ]
cat(sprintf("%d responders: %d recurrences, %d competing deaths\n", nrow(d),
            sum(d$recur_event == 1), sum(d$recur_event == 2)))

# Cumulative incidence (Aalen-Johansen), not 1 - Kaplan-Meier
ci <- prodlim(Hist(recur_months, recur_event) ~ 1, data = d)
print(summary(ci, times = c(6, 12, 18), cause = 1))

f_ref  <- as.formula(paste("Hist(recur_months, recur_event) ~", paste(ref, collapse = " + ")))
f_comb <- as.formula(paste("Hist(recur_months, recur_event) ~", paste(c(ref, "rad_score"), collapse = " + ")))

# Fine-Gray subdistribution hazard models (reference vs reference + radiomic score)
fg_ref  <- FGR(f_ref,  data = d, cause = 1)
fg_comb <- FGR(f_comb, data = d, cause = 1)
print(fg_comb)

# Cause-specific Cox models (reported alongside)
cs_comb <- CSC(f_comb, data = d)
print(cs_comb)

# Competing-risk random survival forest
rf <- rfsrc(as.formula(paste("Surv(recur_months, recur_event) ~", paste(c(ref, "rad_score"), collapse = " + "))),
            data = d, ntree = 500, nodesize = 10, splitrule = "logrankCR", seed = -2026)
print(rf)

# Discrimination (AUC) and Brier score at 12 months, bootstrap cross-validation
sc <- Score(list(FineGray_reference = fg_ref, FineGray_combined = fg_comb, CSC_combined = cs_comb),
            formula = Hist(recur_months, recur_event) ~ 1, data = d, cause = 1, times = 12,
            plots = "calibration", split.method = "bootcv", B = 200, seed = 2026)
print(summary(sc))
write.csv(as.data.frame(sc$AUC$score), "results/recurrence_auc_12m.csv", row.names = FALSE)
