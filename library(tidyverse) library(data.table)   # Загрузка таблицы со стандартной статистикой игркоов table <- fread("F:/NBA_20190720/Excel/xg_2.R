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
  filter(POS == "G" | POS == "C") %>%
  mutate(POS = factor(POS))

## Загрузка xgboost
library(xgboost)
library(caret)
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
train.label <- factor(table1[train.index, "label"])
test.data <- as.matrix(select(table1[-train.index,], -label))
test.label <- factor(table1[-train.index, "label"])



## CV
trctrl <- trainControl(method = "cv", number = 5)

## Настройка eta и nrounds
nrounds <- 1000
tune_grid1 <- expand.grid(
  nrounds = seq(from = 200, to = nrounds, by = 100),
  eta = c(0.025, 0.05, 0.1, 0.3),
  max_depth = 4,
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

xgb_tune1 <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = tune_grid1,
  method = "xgbTree",
  verbose = TRUE
)


## Настройка max_depth и min_child_weight
tune_grid2 <- expand.grid(
  nrounds = 300,
  eta = 0.05,
  max_depth = c(2, 3, 4, 5, 6),
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = c(1, 2, 3, 4, 5, 6),
  subsample = 1
)

xgb_tune2 <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = tune_grid2,
  method = "xgbTree",
  verbose = TRUE
)

## Настройка colsample_bytree и subsample
tune_grid3 <- expand.grid(
  nrounds = 300,
  eta = 0.05,
  max_depth = 2,
  gamma = 0,
  colsample_bytree = c(0.6, 0.7, 0.8, 0.9, 1),
  min_child_weight = 1,
  subsample = c(0.6, 0.7, 0.8, 0.9, 1)
)

xgb_tune3 <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = tune_grid3,
  method = "xgbTree",
  verbose = TRUE
)

## Настройка gamma
tune_grid4 <- expand.grid(
  nrounds = 300,
  eta = 0.05,
  max_depth = 2,
  gamma = c(0, 0.05, 0.1, 0.2, 0.5, 0.7, 0.9, 1),
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.6
)

xgb_tune4 <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = tune_grid4,
  method = "xgbTree",
  verbose = TRUE
)

## Настройка eta и nrounds
tune_grid5 <- expand.grid(
  nrounds = seq(100, 10000, 100),
  eta = seq(0.01, 0.05, 0.01),
  max_depth = 2,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.6
)

xgb_tune5 <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = tune_grid5,
  method = "xgbTree",
  verbose = TRUE
)

# Тренировка модели
final_grid <- expand.grid(
  nrounds = 300,
  eta = 0.05,
  max_depth = 2,
  gamma = 0,
  colsample_bytree = 0.6,
  min_child_weight = 1,
  subsample = 0.6
)

final_tune <- caret::train(
  x = train.data,
  y = train.label,
  trControl = trctrl,
  tuneGrid = final_grid,
  method = "xgbTree",
  verbose = TRUE
)

# Проверка модели на тестовых данных
xgb.pred <- predict(final_tune, test.data, reshape=T)
table(xgb.pred, test.label)




