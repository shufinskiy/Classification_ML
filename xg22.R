library(tidyverse)
library(data.table)
library(ggcorrplot)

# Загрузка таблицы со стандартной статистикой игркоов
table <- fread("F:/NBA_20190720/Excel/positionML.csv")

# Выбор нужных переменных
table1 <- table %>%
  select(PLAYER_NAME,
         GP,
         MIN,
         FG_PCT,
         FG3A,
         FG3_PCT,
         FT_PCT,
         OREB,
         DREB,
         AST,
         TOV,
         STL,
         BLK,
         PF,
         POS,
         YEARS)

# Находим игроков, которые относятся к более чем одной позиции
table2 <- table1 %>%
  group_by(PLAYER_NAME, YEARS) %>%
  count() %>%
  filter(n == 2) %>%
  select(PLAYER_NAME, YEARS)

# Удаляем игркоов с более чем одной позицией из набора данных
# А также тех, кто сыграл меньше 25 матчей и меньше 15 минут за игру.
table1 <- table1 %>%
  anti_join(table2) %>%
  filter(GP >= 25 & MIN >= 15) %>%
  select(-c(GP, MIN))%>%
  filter(POS == "G" | POS == "C") %>%
  mutate(POS = factor(POS))

## Загрузка xgboost
library(xgboost)

## Удаление имён и года
table1 <- table1 %>%
  select(-c(PLAYER_NAME, YEARS))

## Факторы символьные в числовой
position <- table1$POS
label <- as.integer(table1$POS)-1
table1$POS <- NULL

## Разделение данных
n <- nrow(table1)
train.index <- sample(n, floor(0.75*n))
train.data <- as.matrix(table1[train.index,])
train.label <- label[train.index]
test.data <- as.matrix(table1[-train.index,])
test.label <- label[-train.index]

# Трансформация данных в xgb.Matrix
xgb.train <- xgb.DMatrix(data=train.data,label=train.label)
xgb.test <- xgb.DMatrix(data=test.data,label=test.label)

# Параметры для классификации
num_class <- length(levels(position))
params <- list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Тренировка модели
xgb.fit <- xgb.train(
  params = params,
  data = xgb.train,
  nrounds = 10000,
  nthreads = 1,
  early_stopping_rounds = 10,
  watchlist = list(val1 = xgb.train, val2 = xgb.test),
  verbose = 0
)

# Матрица важности предикторов
importance_matrix <- xgb.importance(model = xgb.fit)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

# Просмотр результатов модели
xgb.fit

# Проверка модели на тестовых данных
xgb.pred <- predict(xgb.fit , test.data , reshape=T)
xgb.pred <- as.data.frame(xgb.pred)
colnames(xgb.pred) <- levels(position)

# Используем ту метку, у которой большая вероятность
# Добавляем реальный показатель позиции
xgb.pred$prediction <- apply(xgb.pred, 1 , function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label <- levels(position)[test.label+1]

# Вычисление точности
result <- sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)
print(paste("Final Accuracy =",sprintf("%1.2f%%", 100*result)))

# Построение матрицы ошибок
library(ggplot2)

trrew <- data.frame(table(xgb.pred$prediction, xgb.pred$label))

ggplot(data =  trrew, mapping = aes(x = Var1, y = Var2)) +
  geom_tile(aes(fill = Freq), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", Freq)), vjust = 1) +
  scale_fill_viridis_c() +
  theme_bw() + 
  theme(legend.position = "none")
