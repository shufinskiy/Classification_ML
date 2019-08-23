library(tidyverse)
library(data.table)

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
  # filter(POS == "G" | POS == "C") %>%
  mutate(POS = factor(POS))

## Загрузка xgboost
library(xgboost)

## Удаление имён и года
table1 <- table1 %>%
  select(-c(PLAYER_NAME, YEARS))

## Факторы символьные в числовой
position <- table1$POS
table1$label <- as.integer(table1$POS)-1
table1$POS <- NULL

## Разделение данных
set.seed(123)
n <- nrow(table1)
train.index <- sample(n, floor(0.75*n))
train.data <- as.matrix(select(table1[train.index,], -label))
train.label <- as.integer(table1[train.index, "label"])
test.data <- as.matrix(select(table1[-train.index,], -label))
test.label <- as.integer(table1[-train.index, "label"])

# Трансформация данных в xgb.Matrix
xgb_train <- xgb.DMatrix(data = train.data, label = train.label)
xgb_test <- xgb.DMatrix(data = test.data, label = test.label)

# Параметры для классификации
num_class = length(levels(position))
params = list(
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

xgb.fit <- xgb.train(
  params=params,
  data=xgb_train,
  nrounds=10000,
  nthreads=4,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb_train,val2=xgb_test),
  verbose=0
)

# Проверка модели на тестовых данных
fg <- predict(xgb.fit, test.data, reshape = T)
fg <- as.data.frame(fg)
colnames(fg) <- levels(position)

fg$test.label <- test.label

# Результаты модели
fg <- fg %>%
  rowwise() %>%
  mutate(pred = max(C, F, G)) %>%
  ungroup() %>%
  mutate(label = ifelse(fg$C == fg$pred, 0,
                        ifelse(fg$F == fg$pred, 1, 2))) %>%
  mutate(Result = ifelse(fg$test.label == fg$label, TRUE, FALSE)) %>%
  group_by(Result) %>%
  count(Result)


  
