#37分组
df <- read.csv('c:\\Users\\郁禛元\\Desktop\\CRRT方案构建\\CRRT早期低血压预防项目\\低血压预测模型构建\\数据5.3\\过滤后数据5.3.csv')
head(df)

# 将结局变量设为分组 factor（0/1 -> 非低血压/低血压）
df <- df %>%
  mutate(
    group = factor(hypotension, levels = c(0,1), labels = c("nonhypotension", "hypotension"))
  )

# 分割数据集：7/3
set.seed(123)
trainIndex <- createDataPartition(df$group, p = 0.7, list = FALSE)

train1 <- df[trainIndex, ]
test1  <- df[-trainIndex, ]
write.csv(train1, "c:\\Users\\郁禛元\\Desktop\\train.csv", row.names = FALSE)
#---------------------------------------11.lasso提取建模变量-------------------------------
#提取重要变量LASSO验证
# 将 group 转为二元数值型因变量
y_train1 <- ifelse(train1$group == "hypotension", 1, 0)
head(train1)
# 准备训练特征矩阵
x_train1 <- model.matrix(~ gender + admission_age + bmi + myocardial_infarct + congestive_heart_failure + peripheral_vascular_disease + cerebrovascular_disease + dementia +
                           chronic_pulmonary_disease + rheumatic_disease + peptic_ulcer_disease +  mild_liver_disease + diabetes_without_cc + diabetes_with_cc + malignant_cancer + 
                           severe_liver_disease + metastatic_solid_tumor + aids + sofa + mechanical.ventilation + vasoactive.drugs + calcium + ph + lactate + hemoglobin + albumin +
                           icu_to_rrt_hours + rrt_type + map + sap - 1, 
                         data = train1)
library(glmnet)

# 10折交叉验证LASSO
set.seed(123)
cv.lasso <- cv.glmnet(x_train1, y_train1, 
                      family = "binomial",
                      alpha = 1,
                      type.measure = "auc",
                      nfolds = 10)

# 可视化交叉验证曲线
plot(cv.lasso)

# 提取在 lambda.min 下的非零系数（即入选变量）
coef.min <- coef(cv.lasso, s = "lambda.min")
selected_vars1 <- rownames(coef.min)[coef.min[,1] != 0][-1]  # 去掉截距项

# 查看筛选结果
print("LASSO选择的变量：")
print(selected_vars1)
write.csv(selected_vars1, "c:\\Users\\郁禛元\\Desktop\\模型选择变量无切点.csv", row.names = FALSE)
#------------------------------------12.建立十种模型-----------------------------------

# -------------------------
# 控制参数：5折交叉验证 + AUC
ctrl <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE,
  savePredictions = "final"
)

# -------------------------
# 数据准备（分组后的训练集）
train_df1 <- train1
train_df1$y <- factor(train_df1$group, levels = c("nonhypotension", "hypotension"))
selected_vars1
colnames(train_df1)
head(train_df1)

# 12.1 构建哑变量矩阵（这是 selected_vars1 所基于的列名来源）
x1 <- model.matrix(~ gender + admission_age + congestive_heart_failure + peripheral_vascular_disease + dementia + 
                     chronic_pulmonary_disease + mild_liver_disease + diabetes_without_cc + malignant_cancer + metastatic_solid_tumor + 
                     vasoactive.drugs + ph + lactate + icu_to_rrt_hours + rrt_type + map + sap - 1,
                   data = train_df1)
head(x1)

# 12.2 转换因变量为 factor
y1 <- train_df1$y  # y 是 "nonhypotension"/"hypotension"
head(y1)
# 12.3 确保 selected_vars1 与 x1列名匹配
selected_vars1_matched <- intersect(colnames(x1), selected_vars1)
head(selected_vars1_matched)
#12.4提取用于建模的子集
x1_sub <- x1[, selected_vars1_matched]
head(x1_sub)

# 【新增】统一赋值变量 x, y
x <- x1_sub
y <- y1 
# 12.4 初始化模型列表
# -------------------------
model_list <- list()

# 模型1：LASSO Logistic 回归
model_list[["Lasso-LR"]] <- train(
  x = x,
  y = y,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = 1, lambda = cv.lasso$lambda.min),
  metric = "ROC",
  trControl = ctrl
)

# 模型2：决策树
model_list[["DT"]] <- train(
  x = x,
  y = y,
  method = "rpart",
  tuneGrid = expand.grid(cp = seq(0.001, 0.1, length = 10)),
  metric = "ROC",
  trControl = ctrl
)

# 模型3：随机森林
model_list[["RF"]] <- train(
  x = x,
  y = y,
  method = "rf",
  tuneGrid = expand.grid(mtry = c(2, floor(sqrt(length(selected_vars1))))),  # ✅已修正
  metric = "ROC",
  trControl = ctrl
)

# 模型4：XGBoost
model_list[["XGBoost"]] <- train(
  x = x,
  y = y,
  method = "xgbTree",
  tuneGrid = expand.grid(
    nrounds = c(100),
    max_depth = c(3, 5),
    eta = c(0.01, 0.1),
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  ),
  metric = "ROC",
  trControl = ctrl
)

# 模型5：KNN
model_list[["KNN"]] <- train(
  x = x,
  y = y,
  method = "knn",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)

# 模型6：朴素贝叶斯
model_list[["NaiveBayes"]] <- train(
  x = x,
  y = y,
  method = "naive_bayes",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)

# 模型7：支持向量机（径向基核）
model_list[["SVM"]] <- train(
  x = x,
  y = y,
  method = "svmRadial",
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)

# 模型8：GBM
model_list[["GBM"]] <- train(
  x = x,
  y = y,
  method = "gbm",
  verbose = FALSE,
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)

# 模型9：神经网络
model_list[["NeuralNet"]] <- train(
  x = x,
  y = y,
  method = "nnet",
  trace = FALSE,
  tuneLength = 5,
  metric = "ROC",
  trControl = ctrl
)

# 模型10：传统 Logistic 回归
model_list[["Logistic"]] <- train(
  x = x,
  y = y,
  method = "glm",
  family = binomial(),
  metric = "ROC",
  trControl = ctrl
)
results <- data.frame()

for (model_name in names(model_list)) {
  model <- model_list[[model_name]]
  best_tune <- model$bestTune
  
  # ========== TRAIN阶段调整最佳阈值 ========== #
  pred_train <- model$pred %>%
    merge(best_tune) %>%
    group_by(rowIndex) %>%
    summarise(obs = first(obs),
              prob = mean(hypotension),
              .groups = 'drop')
  
  roc_train <- roc(pred_train$obs, pred_train$prob)
  best_cutoff <- coords(roc_train, "best", ret = "threshold", best.method = "youden")
  best_cutoff <- round(as.numeric(best_cutoff), 4)
  
  pred_class_train <- factor(ifelse(pred_train$prob > best_cutoff, "hypotension", "nonhypotension"),
                             levels = c("nonhypotension", "hypotension"))
  cm_train <- confusionMatrix(pred_class_train, pred_train$obs, positive = "hypotension")
  
  auc_train <- round(auc(roc_train), 3)
  acc_train <- round(cm_train$overall["Accuracy"], 3)
  kappa_train <- round(cm_train$overall["Kappa"], 3)
  sen_train <- round(cm_train$byClass["Sensitivity"], 3)
  spe_train <- round(cm_train$byClass["Specificity"], 3)
  npv_train <- round(cm_train$byClass["Neg Pred Value"], 3)
  ppv_train <- round(cm_train$byClass["Pos Pred Value"], 3)
  recall_train <- round(cm_train$byClass["Recall"], 3)
  brier_train <- mean((as.numeric(pred_train$obs == "hypotension") - pred_train$prob)^2)
  
  # ========== 13.2TEST阶段验证 ========== #
  test_df1 <- test1
  test_df1$y <- factor(test_df1$group, levels = c("nonhypotension", "hypotension"))
  x_test <- model.matrix(~ gender + admission_age + congestive_heart_failure + peripheral_vascular_disease + dementia + 
                           chronic_pulmonary_disease + mild_liver_disease + diabetes_without_cc + malignant_cancer + metastatic_solid_tumor + 
                           vasoactive.drugs + ph + lactate + icu_to_rrt_hours + rrt_type + map + sap - 1,
                         data = test_df1)
  x_test <- x_test[, colnames(x)]  # 保证列名一致
  
  prob_test <- predict(model, newdata = x_test, type = "prob")[, "hypotension"]
  pred_class_test <- factor(ifelse(prob_test > best_cutoff, "hypotension", "nonhypotension"),
                            levels = c("nonhypotension", "hypotension"))
  cm_test <- confusionMatrix(pred_class_test, test_df1$y, positive = "hypotension")
  auc_test <- round(auc(test_df1$y, prob_test), 3)
  acc_test <- round(cm_test$overall["Accuracy"], 3)
  kappa_test <- round(cm_test$overall["Kappa"], 3)
  sen_test <- round(cm_test$byClass["Sensitivity"], 3)
  spe_test <- round(cm_test$byClass["Specificity"], 3)
  npv_test <- round(cm_test$byClass["Neg Pred Value"], 3)
  ppv_test <- round(cm_test$byClass["Pos Pred Value"], 3)
  recall_test <- round(cm_test$byClass["Recall"], 3)
  brier_test <- mean((as.numeric(test_df1$y == "hypotension") - prob_test)^2)
  
  # ========== 汇总 ========== #
  results <- rbind(results, data.frame(
    Model = model_name,
    Best_Cutoff = best_cutoff,
    Train_AUC = auc_train,
    Train_Accuracy = acc_train,
    Train_Kappa = kappa_train,
    Train_Sensitivity = sen_train,
    Train_Specificity = spe_train,
    Train_NPV = npv_train,
    Train_PPV = ppv_train,
    Train_Recall = recall_train,
    Train_Brier = round(brier_train, 4),
    Test_AUC = auc_test,
    Test_Accuracy = acc_test,
    Test_Kappa = kappa_test,
    Test_Sensitivity = sen_test,
    Test_Specificity = spe_test,
    Test_NPV = npv_test,
    Test_PPV = ppv_test,
    Test_Recall = recall_test,
    Test_Brier = round(brier_test, 4)
  ))
}

# ========== 输出 ========== #
print(results)
write.csv(results, "Train_Test_Full_Evaluation_终版.csv", row.names = FALSE)
write.csv(results,file ='c:\\Users\\郁禛元\\Desktop\\Train_Test_Full_Evaluation_连续性变量.csv',row.names = F )
#===========SHAP 分析
# 1. 加载必要包
# ===============================
library(shapviz)
library(Matrix)
library(ggplot2)

# ===============================
# 2. 提取您建好的 XGBoost 模型
# ===============================
xgb_model <- model_list[["XGBoost"]]         # 从 model_list 中取出
xgb_final <- xgb_model$finalModel            # caret 封装下提取最终模型

# ===============================
# 3. 准备建模数据（注意：必须是矩阵）
# ===============================
X_train <- as.matrix(x)                      # x 就是您建模用的 x1_sub
y_train <- as.numeric(y) - 1                 # 将 factor 转为 0/1，非低血压为0，低血压为1

# ===============================
# 4. 计算 SHAP 值
# ===============================
shap_xgb <- shapviz(
  xgb_final,
  X_pred = X_train
)

# ===============================
# 5. 全局解释：变量重要性图（蜂群图 + 柱状图）
# ===============================
sv_importance(shap_xgb, kind = "beeswarm")         # 蜂群图（推荐）
sv_importance(shap_xgb) + theme_bw()               # 柱状图

# ===============================
# 6. 局部解释：单个样本
# ===============================
sv_waterfall(shap_xgb, row_id = 1)     # 解释第5个样本：每个变量如何推高或压低风险
sv_force(shap_xgb, row_id = 1)        # 解释第10个样本：可视化推力

# （如需逐行看不同样本，请将 row_id 改为其他数字）