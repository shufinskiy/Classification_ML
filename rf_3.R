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

# Удаляем игроков с более чем одной позицией из набора данных,
# а также тех, кто сыграл меньше 25 матчей и меньше 15 минут за игру.
table1 <- table1 %>%
  anti_join(table2) %>%
  filter(GP >= 25 & MIN >= 15) %>%
  select(-c(GP, MIN))

# Корреляционная таблица с указанием p-value
cor_table <- table1 %>%
  select(FG_PCT,
         FG3A,
         FG3_PCT,
         FT_PCT,
         OREB,
         DREB,
         AST,
         TOV,
         STL,
         BLK,
         PF)

corr <- round(cor(cor_table), digits = 2)

p.value <- cor_pmat(cor_table)

ggcorrplot(corr, hc.order = TRUE, type = "lower", 
           p.mat = p.value, lab = TRUE, method = "circle")

# Построение модели классификации игроков по их стандартной статистике

# Таблица, которая будет использоваться при тренировке
# модели, без года
table_ML <- table1 %>%
  select(-YEARS) %>%
  mutate(POS = factor(POS))

library(caTools)
set.seed(1234)

# Случайное разбиение на обучающий и тестовый набор данных
split <- sample.split(table_ML, SplitRatio = 0.75) 

table_train <- subset(table_ML, split == TRUE)
table_test <- subset(table_ML, split == FALSE)

# Сохранение имён игроков из тестового набора
name <- table_test$PLAYER_NAME

#Удаление имён из тренировочного и тестового набора данных
table_train <- table_train %>%
  select(-PLAYER_NAME)

table_test <- table_test %>%
  select(-PLAYER_NAME)

# Подключение библиотек машинного обучения
library(caret)

# K-10 перекрестная проверка
trControl <- trainControl(method = "cv", number = 10, search = "grid")

# Тренировка модели с параметрами по умолчанию
rf_default <- train(POS~.,
                    data = table_train,
                    method = "rf",
                    metric = "Accuracy",
                    trControl = trControl)

print(rf_default)

# Поиск наилучшего количества предикторов
# Создание таблицы со всеми возможными комбинациями .mtry
tuneGrid <- expand.grid(.mtry = c(1:11))

rf_mtry <- train(POS~.,
                 data = table_train,
                 method = "rf",
                 metric = "Accuracy",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 nodesize = 14,
                 ntree = 300)
print(rf_mtry)

# Сохранения лучшего значения параметра mtry
best_mtry <- rf_mtry$bestTune$mtry

# Поиск наилучшего максимального количества узлов в дереве
store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(36:39)) {
  set.seed(1234)
  rf_maxnode <- train(POS~.,
                      data = table_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)

# Поиск лучшего количества деревьев
store_maxtrees <- list()

for (ntree in seq(100, 1500, 100)) {
  set.seed(1234)
  rf_maxtrees <- train(POS~.,
                      data = table_train,
                      method = "rf",
                      metric = "Accuracy",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      nodesize = 14,
                      maxnodes = 32,
                      ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_ntree <- resamples(store_maxtrees)
summary(results_ntree)

# Модель с улучшенными параметрами
fit_rf <- train(POS~.,
                table_train,
                method = "rf",
                metric = "Accuracy",
                tuneGrid = tuneGrid,
                trControl = trControl,
                importance = TRUE,
                nodesize = 14,
                ntree = 800,
                maxnodes = 32)
print(fit_rf)

# Оценка модели на тестовых данных
prediction <- predict(fit_rf, table_test)

# Важность переменных
varImp(fit_rf)

# Матрица ошибок
confusionMatrix(prediction, table_test$POS)

# Таблица игроков с неправильно указанными позициями
table_test1 <- table_test %>%
  mutate(POS1 = prediction,
         PLAYER = name) %>%
  filter(POS != POS1)
