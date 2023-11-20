#импорт библиотек для интерфейса
import sys
import math
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QLineEdit, QGridLayout, QPushButton, QTextEdit, QFileDialog, QProgressBar, QMessageBox, QCheckBox
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QThread, QObject, pyqtSignal as Signal, pyqtSlot as Slot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

#импорт библиотек для модели
from tensorflow import data as tfdt
from tensorflow import keras as tfks
import matplotlib.pyplot as plt
from numpy import array as ar
from numpy import arange as ag
from pandas import read_csv as pdcsv

import csv

#функция для формирования данных
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return ar(data), ar(labels)

#рабочий класс, который будет работать в отдельном потоке
class Worker(QObject):
    #сигнал для вывода прогресса
    progress = Signal(int)
    status = Signal(str)
    completed = Signal(int)

    @Slot(int)
    def do_work(self):
        try:
            #передача значений из полей интерфейса в модель в потоке
            self.shift = int(win.shift_line.text())
            self.TRAIN_SPLIT = int(win.train_size_line.text())
            self.past_history = int(win.past_history_line.text())
            self.future_target = int(win.future_target_line.text()) + self.shift
            self.STEP = int(win.step_line.text())
            self.BUFFER_SIZE = int(win.buffer_size_line.text())
            self.BATCH_SIZE = int(win.batch_size_line.text())
            self.lstm_layers = int(win.lstm_layers_line.text())
            self.optimizer = win.optimizer_line.text()
            self.loss = win.loss_line.text()
            self.EPOCHS = int(win.epochs_line.text())
            self.EVALUATION_INTERVAL = int(win.evaluation_interval_line.text())
            self.validation_steps = int(win.validation_steps_line.text())
            # выдать 10% прогресс за присвоение переменных
            self.progress.emit(10)
            self.status.emit('Model parameters imported successfully.')
            
            #стандартизация набора данных
            self.dataset = win.features.values
            self.data_mean = self.dataset[:self.TRAIN_SPLIT].mean(axis=0)
            self.data_std = self.dataset[:self.TRAIN_SPLIT].std(axis=0)
            self.dataset = (self.dataset-self.data_mean)/self.data_std
            # выдать 20% прогресс за работу с наборами данных
            self.progress.emit(20)
            self.status.emit('Now the values are standard.')

            
            #присвоение массивов, созданных по шагам переменным, которые будут использоваться для обучения нейросети
            x_train_multi, y_train_multi = multivariate_data(self.dataset, self.dataset[:, 0], 0, self.TRAIN_SPLIT, self.past_history, self.future_target, self.STEP)
            x_val_multi, y_val_multi = multivariate_data(self.dataset, self.dataset[:, 0], self.TRAIN_SPLIT, None, self.past_history, self.future_target, self.STEP)
            # выдать 30% прогресс
            self.progress.emit(30)
            self.status.emit('Data variation completed.')

            #обработка данных для передачи в модель
            train_data_multi = tfdt.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
            train_data_multi = train_data_multi.cache().shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).repeat()
            val_data_multi = tfdt.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
            val_data_multi = val_data_multi.batch(self.BATCH_SIZE).repeat()
            # выдать 40% прогресс
            self.progress.emit(40)
            self.status.emit('Data prepared for machine learning.')

            #описание модели множественного прогноза
            self.multi_step_model = tfks.models.Sequential()
            self.multi_step_model.add(tfks.layers.LSTM(self.lstm_layers, input_shape=x_train_multi.shape[-2:]))
            self.multi_step_model.add(tfks.layers.Dense(self.future_target))
            self.multi_step_model.compile(optimizer=self.optimizer, loss=self.loss)
            # выдать 60% прогресс
            self.progress.emit(60)
            self.status.emit('Sequential LSTM neural network model is ready to start learning.')

            #если НЕ был отмечен чек-бокс предобученной модели произвести обучение
            if not win.pretrained_checkbox.isChecked():
                #обучение модели множественного прогноза
                multi_step_history = self.multi_step_model.fit(train_data_multi, epochs=self.EPOCHS, steps_per_epoch=self.EVALUATION_INTERVAL, validation_data=val_data_multi, validation_steps=self.validation_steps)
                # выдать 80% прогресс
                self.progress.emit(80)
                self.status.emit('Model fitting completed.')
            else:
                #если указан путь к предобученной модели
                if win.load_model_directory:
                    self.multi_step_model = tfks.models.load_model(win.load_model_directory)
                #выдать 80% прогресс
                self.progress.emit(80)
                self.status.emit('Model loading completed.')

            #формирование массива результатов
            self.given = win.features['Q'][self.TRAIN_SPLIT:self.TRAIN_SPLIT+self.future_target-self.shift].values
            x = val_data_multi.take(1)
            self.predicted = self.multi_step_model.predict(x)[0].tolist()
            self.predicted = list(value*self.data_std[0]+self.data_mean[0] for value in self.predicted)
            # выдать 100% прогресс
            self.progress.emit(100)
            self.status.emit('Results are ready.')
            self.completed.emit(1)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)=}")
            raise

class ForecastApp(QWidget):
    #сигнал для начала работы рабочего класса в потоке, сюда приходит сигнал от метода, который обрабатывает нажатие кнопки
    work_requested = Signal(int)

    #метод инициализация приложения
    def __init__(self):
        super().__init__()
        self.initUI()

    #метод инициализация интерфейса
    def initUI(self):
        #блок открытия файла
        self.file_path_label = QLabel('Path to file:')
        self.file_path_line = QLineEdit('Set with button on the right ->')
        self.file_path_line.setReadOnly(True)
        self.file_open_button = QPushButton('Open')
        self.file_open_button.clicked.connect(self.open_button_was_clicked)

        #блок настройки модели
        #train_split
        self.train_size_label = QLabel('TRAIN_SIZE:')
        self.train_size_line = QLineEdit('360')
        #batch_size
        self.batch_size_label = QLabel('BATCH_SIZE:')
        self.batch_size_line = QLineEdit('256')
        #buffer_size
        self.buffer_size_label = QLabel('BUFFER_SIZE:')
        self.buffer_size_line = QLineEdit('1000')
        #evaluation_interval
        self.evaluation_interval_label = QLabel('EVALUATION_INTERVAL:')
        self.evaluation_interval_line = QLineEdit('100')
        #epochs
        self.epochs_label = QLabel('EPOCHS:')
        self.epochs_line = QLineEdit('10')
        #past_history
        self.past_history_label = QLabel('PAST_HISTORY:')
        self.past_history_line = QLineEdit('30')
        #future_target
        self.future_target_label = QLabel('FUTURE_TARGET:')
        self.future_target_line = QLineEdit('360')
        #step
        self.step_label = QLabel('STEP:')
        self.step_line = QLineEdit('1')
        #lstm_layers
        self.lstm_layers_label = QLabel('LSTM_LAYERS:')
        self.lstm_layers_line = QLineEdit('32')
        #optimizer
        self.optimizer_label = QLabel('OPTIMIZER:')
        self.optimizer_line = QLineEdit('adam')
        #loss
        self.loss_label = QLabel('LOSS:')
        self.loss_line = QLineEdit('mse')
        #validation_steps
        self.validation_steps_label = QLabel('VALIDATION_STEPS:')
        self.validation_steps_line = QLineEdit('100')

        #блок графики
        self.figure = plt.figure(figsize = [1, 1], edgecolor = 'black')
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.figure.set_layout_engine(layout='tight')
        ax = self.figure.add_subplot(111)
        self.font = {'size' : 8}
        ax.set_title('Discharge', self.font)
        ax.set_xlabel('Days', self.font)
        ax.set_ylabel('m^3', self.font)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid()
        ax.text(x = 0.4, y = 0.5, s = 'No graphics to plot.', fontdict = self.font)

        #кнопки управления программой
        #кнопка старта прогнозирования
        self.start_button = QPushButton('Start forecast')
        self.start_button.clicked.connect(self.start_button_was_clicked)
        #кнопка сохранения результатов
        self.save_data_button = QPushButton('Save data')
        self.save_data_button.clicked.connect(self.save_data_button_was_clicked)
        
        #чек-боксы выбора колонок гидрологических характеристик и метеоусловий
        self.checkbox_label = QLabel('Features:')
        self.Precipitation_checkbox = QCheckBox('Precipitation')
        self.Snow_checkbox = QCheckBox('Snow')
        self.Temperature_checkbox = QCheckBox('Temperature')
        
        #чек-боксы предобученной модели и сохранения модели
        self.save_model_checkbox = QCheckBox('Save model')
        self.pretrained_checkbox = QCheckBox('Use pretrained model')

        #блок состояния
        self.status_label = QLabel('Status')
        self.status_text = QTextEdit('Interface is good)')
        self.status_text.setMaximumHeight(self.status_label.sizeHint().height() * 5)
        self.progress_bar = QProgressBar()

        #костыль смещение
        self.shift_label = QLabel('SHIFT:')
        self.shift_line = QLineEdit('7')

        #сеточный интерфейс
        grid = QGridLayout()
        #расстояние между клетками
        grid.setSpacing(10)

        #добавление виджетов
        #блок открытия файла
        grid.addWidget(self.file_path_label, 1, 1)
        grid.addWidget(self.file_path_line, 1, 2, 1, 4)
        grid.addWidget(self.file_open_button, 1, 6)

        #блок настройки модели
        #train_split
        grid.addWidget(self.train_size_label, 2, 1)
        grid.addWidget(self.train_size_line, 2, 2)

        #batch_size
        grid.addWidget(self.batch_size_label, 3, 1)
        grid.addWidget(self.batch_size_line, 3, 2)

        #buffer_size
        grid.addWidget(self.buffer_size_label, 4, 1)
        grid.addWidget(self.buffer_size_line, 4, 2)

        #evaluation_interval
        grid.addWidget(self.evaluation_interval_label, 5, 1)
        grid.addWidget(self.evaluation_interval_line, 5, 2)

        #epochs
        grid.addWidget(self.epochs_label, 2, 3)
        grid.addWidget(self.epochs_line, 2, 4)

        #past_history
        grid.addWidget(self.past_history_label, 3, 3)
        grid.addWidget(self.past_history_line, 3, 4)

        #future_target
        grid.addWidget(self.future_target_label, 4, 3)
        grid.addWidget(self.future_target_line, 4, 4)

        #step
        grid.addWidget(self.step_label, 5, 3)
        grid.addWidget(self.step_line, 5, 4)

        #lstm_layers
        grid.addWidget(self.lstm_layers_label, 2, 5)
        grid.addWidget(self.lstm_layers_line, 2, 6)

        #optimizer
        grid.addWidget(self.optimizer_label, 3, 5)
        grid.addWidget(self.optimizer_line, 3, 6)

        #loss
        grid.addWidget(self.loss_label, 4, 5)
        grid.addWidget(self.loss_line, 4, 6)

        #validation_steps
        grid.addWidget(self.validation_steps_label, 5, 5)
        grid.addWidget(self.validation_steps_line, 5, 6)
        
        #блок графики
        grid.addWidget(self.toolbar, 8, 1, 8, 6)
        grid.addWidget(self.canvas, 13, 1, 24, 6)

        #кнопки управления
        grid.addWidget(self.start_button, 8, 5)
        grid.addWidget(self.save_data_button, 8, 6)

        #чек-боксы выбора колонок
        grid.addWidget(self.checkbox_label, 8, 1)
        grid.addWidget(self.Precipitation_checkbox, 8, 2)
        grid.addWidget(self.Snow_checkbox, 8, 3)
        grid.addWidget(self.Temperature_checkbox, 8, 4)
        #чек-боксы сохранения модели и выбора предтренированной модели
        grid.addWidget(self.save_model_checkbox, 10, 5)
        grid.addWidget(self.pretrained_checkbox, 9, 5)

        #блок состояния
        grid.addWidget(self.status_label, 42, 1)
        grid.addWidget(self.status_text, 43, 1, 2, 6)

        #прогресс
        grid.addWidget(self.progress_bar, 45, 1, 1, 6)

        #костыль смещение
        grid.addWidget(self.shift_label, 6, 1)
        grid.addWidget(self.shift_line, 6, 2)

        #объявление интерфейса
        self.setLayout(grid)

        #задание параметров и команда на вывод
        self.setWindowTitle('LSTM forecast interface')
        self.setWindowIcon(QIcon('icon.png'))

        #инициализация экземпляров рабочего класса и потока
        self.worker = Worker()
        self.worker_thread = QThread()

        #связь рабочего класса с методом виджета
        self.worker.progress.connect(self.update_progress)
        self.worker.status.connect(self.to_status_text)
        self.worker.completed.connect(self.complete)

        #связь сигнала виджета с методом рабочего класса
        self.work_requested.connect(self.worker.do_work)

        #перемещение рабочего класса в поток
        self.worker.moveToThread(self.worker_thread)

        #запуск потока
        self.worker_thread.start()

        #показ виджета
        self.show()

    #метод изменения положения полосы загрузки
    def update_progress(self, v):
        self.progress_bar.setValue(v)
    
    #метод для отображения информации в строке состояния
    def to_status_text(self, status):
        self.status_text.append(status)
        self.status_text.verticalScrollBar().setValue(self.status_text.verticalScrollBar().maximum())

    #метод обработки кнопки открытия файла
    def open_button_was_clicked(self):
        try:
            filename = QFileDialog.getOpenFileName(self, 'Open file', None ,'CSV files (*.csv)')[0]
            if filename:
                self.file_path_line.setText(filename)
                self.to_status_text('Opened file: ' + filename)

                #чтение файла в DataFrame
                self.df = pdcsv(filename, sep=';', usecols=[0, 1, 2, 3, 4],
                       names=['Date', 'Q', 'Precipitation', 'Snow', 'Temp'],
                       encoding = 'utf-8')
                # выбор колонок данных, с которыми будет работать сеть
                features_considered = ['Q']
                if self.Precipitation_checkbox.isChecked():
                    features_considered.append('Precipitation')
                if self.Snow_checkbox.isChecked():
                    features_considered.append('Snow')
                if self.Temperature_checkbox.isChecked():
                    features_considered.append('Temp')
                
                self.features = self.df[features_considered]
                # выбор колонки, содержащей даты, в качестве индекса
                self.features.index = self.df['Date']

                #отображение считанных данных
                self.to_status_text(str(self.features))
                self.figure.clear()
                self.figure.set_layout_engine(layout='tight')
                ax = self.figure.add_subplot(111)
                ax.set_title('Discharge', self.font)
                ax.set_xlabel('Days', self.font)
                ax.set_ylabel('m^3', self.font)
                ax.tick_params(axis='x', labelsize=6)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid()
                x = self.df['Date']
                y = self.df['Q']
                ax.plot(x, y)
                plt.xticks(ag(0, len(x)+1, int(len(self.df)/10)))
                plt.yticks(ag(0, max(y), int(max(self.df['Q'])/11)))
                self.canvas.draw()

        
        except:
            self.to_status_text('Some errors occured.')

    #метод обработки кнопки старта прогнозирования
    def start_button_was_clicked(self):
        try:
            if win.file_path_line.text() != 'Set with button on the right ->':
                #если отмечен чек-бокс предобученной модели, запустить файловый диалог для указания пути к модели
                if self.pretrained_checkbox.isChecked():
                    self.load_model_directory = QFileDialog.getExistingDirectory(self)

                #передача сигнала к началу работы рабочего класса в потоке
                n = 100
                self.progress_bar.setMaximum(n)
                self.work_requested.emit(n)
        except:
            self.to_status_text('Some errors occured.')

    #вывод графика с результатами по завершению обучения нейросети
    def complete(self):
        try:
            #если отмечен чек-бокс сохранения модели, сохранить модель
            if win.save_model_checkbox.isChecked():
                directory = QFileDialog.getExistingDirectory(self)
                if directory:
                    self.worker.multi_step_model.save(directory)
                    #вывести всплывающее окно об успешном сохранении
                    msgBox = QMessageBox(icon = QMessageBox.Icon.Information)
                    msgBox.setWindowIcon(QIcon('icon.png'))
                    msgBox.setWindowTitle('Model saved')
                    msgBox.setText('Model saved in  ' + directory)
                    msgBox.exec()
            #вывод графика прогноза
            self.figure.clear()
            self.figure.set_layout_engine(layout='tight')
            ax = self.figure.add_subplot(111)
            ax.grid()
            ax.set_title('Discharge', self.font)
            ax.set_xlabel('Days', self.font)
            ax.set_ylabel('m^3', self.font)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=8)
            x = self.df['Date'][self.worker.TRAIN_SPLIT:self.worker.TRAIN_SPLIT+self.worker.future_target-self.worker.shift]
            y1 = self.worker.given
            #формирование массивов равной длины
            y2 = self.worker.predicted[self.worker.shift:]
            y = list(zip(y1, y2))
            ax.plot(x, y)
            plt.xticks(ag(0, len(x), int(math.ceil(len(x)/12))))
            plt.yticks(ag(0, max(y1), int(max(y1)/6)))            
            self.canvas.draw()
            #сообщение в строку состояния
            self.to_status_text('Result plotted successfully.')

            #описание результатов
            error = 0.0
            for i in range(0, self.worker.future_target-self.worker.shift):
                error = error + ((abs(float(self.worker.predicted[i+self.worker.shift]) - float(self.worker.given[i])))/self.worker.given[i])
            error = error/self.worker.future_target
            self.to_status_text('Total average error is ' + '{:.2f}'.format(error/0.01) + '%')

            #вывести всплывающее окно о результатах
            msgBox = QMessageBox(icon = QMessageBox.Icon.Information)
            msgBox.setWindowIcon(QIcon('icon.png'))
            msgBox.setWindowTitle('Proccess finished')
            msgBox.setText('Forecast is ready. Total average error is ' + '{:.2f}'.format(error/0.01) + '%')
            msgBox.exec()
        #except:
            #self.to_status_text('Some errors occured.')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

    #обработка кнопки записи результатов прогноза в файл
    def save_data_button_was_clicked(self):
        try:
            #вывод результатов в CSV файл
            filename = QFileDialog.getSaveFileName(self, 'Save file', None ,'CSV files (*.csv)')[0]
            if filename:
                with open(filename, 'w', newline='') as csvfile:
                    spamwriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    spamwriter.writerow(['Date'] + ['RealDischarge'] + ['PredictedDischarge'])
                    for i in range(0, len(self.worker.predicted)-self.worker.shift):
                        spamwriter.writerow([self.df['Date'][self.worker.TRAIN_SPLIT+i]]+ [self.worker.given[i]] + [self.worker.predicted[i+self.worker.shift]])

            #сообщение в строку состояния
            self.to_status_text('Saved file: ' + filename)
        except:
            self.to_status_text('Some errors occured.')



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = ForecastApp()
    app.exec()