import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QLineEdit, QPushButton, QMessageBox, QTextEdit, QComboBox, QFileDialog
from PyQt5.QtGui import QFont, QIntValidator, QDoubleValidator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from hilfe import HelpWindow
import preprocessing


class RNN_GUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RNN Builder")
        self.setGeometry(100, 100, 800, 600)

        self.label = QLabel(self)
        self.label.setText("RNN Builder")
        font = QFont("Arial", 20)
        self.label.setFont(font)
        self.label.setGeometry(300, 10, 200, 50)

        self.input_label = QLabel(self)
        self.input_label.setText("Input Size:")
        self.input_label.setGeometry(30, 80, 100, 30)

        self.input_entry = QLineEdit(self)
        self.input_entry.setGeometry(150, 80, 100, 30)
        self.input_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.input_help_button = QPushButton(self)
        self.input_help_button.setText("Hilfe")
        self.input_help_button.setGeometry(260, 80, 60, 30)
        self.input_help_button.clicked.connect(self.show_input_help)

        self.hidden_label = QLabel(self)
        self.hidden_label.setText("Hidden Size:")
        self.hidden_label.setGeometry(30, 120, 100, 30)

        self.hidden_entry = QLineEdit(self)
        self.hidden_entry.setGeometry(150, 120, 100, 30)
        self.hidden_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.hidden_help_button = QPushButton(self)
        self.hidden_help_button.setText("Hilfe")
        self.hidden_help_button.setGeometry(260, 120, 60, 30)
        self.hidden_help_button.clicked.connect(self.show_hidden_help)

        self.vocab_label = QLabel(self)
        self.vocab_label.setText("Vocabulary Size:")
        self.vocab_label.setGeometry(30, 160, 120, 30)

        self.vocab_entry = QLineEdit(self)
        self.vocab_entry.setGeometry(150, 160, 100, 30)
        self.vocab_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.vocab_help_button = QPushButton(self)
        self.vocab_help_button.setText("Hilfe")
        self.vocab_help_button.setGeometry(260, 160, 60, 30)
        self.vocab_help_button.clicked.connect(self.show_vocab_help)

        self.layers_label = QLabel(self)
        self.layers_label.setText("Anzahl der LSTM-Schichten:")
        self.layers_label.setGeometry(30, 200, 180, 30)

        self.layers_entry = QLineEdit(self)
        self.layers_entry.setGeometry(220, 200, 100, 30)
        self.layers_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.layers_help_button = QPushButton(self)
        self.layers_help_button.setText("Hilfe")
        self.layers_help_button.setGeometry(340, 200, 60, 30)
        self.layers_help_button.clicked.connect(self.show_layers_help)

        self.dropout_label = QLabel(self)
        self.dropout_label.setText("Dropout-Rate:")
        self.dropout_label.setGeometry(30, 240, 100, 30)

        self.dropout_entry = QLineEdit(self)
        self.dropout_entry.setGeometry(150, 240, 100, 30)
        self.dropout_entry.setValidator(QDoubleValidator())  # Nur Fließkommazahlen erlauben

        self.dropout_help_button = QPushButton(self)
        self.dropout_help_button.setText("Hilfe")
        self.dropout_help_button.setGeometry(260, 240, 60, 30)
        self.dropout_help_button.clicked.connect(self.show_dropout_help)

        self.learning_rate_label = QLabel(self)
        self.learning_rate_label.setText("Lernrate:")
        self.learning_rate_label.setGeometry(30, 280, 100, 30)

        self.learning_rate_entry = QLineEdit(self)
        self.learning_rate_entry.setGeometry(150, 280, 100, 30)
        self.learning_rate_entry.setValidator(QDoubleValidator())  # Nur Fließkommazahlen erlauben

        self.learning_rate_help_button = QPushButton(self)
        self.learning_rate_help_button.setText("Hilfe")
        self.learning_rate_help_button.setGeometry(260, 280, 60, 30)
        self.learning_rate_help_button.clicked.connect(self.show_learning_rate_help)

        self.batch_label = QLabel(self)
        self.batch_label.setText("Batch Size:")
        self.batch_label.setGeometry(30, 320, 100, 30)

        self.batch_entry = QLineEdit(self)
        self.batch_entry.setGeometry(150, 320, 100, 30)
        self.batch_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.batch_help_button = QPushButton(self)
        self.batch_help_button.setText("Hilfe")
        self.batch_help_button.setGeometry(260, 320, 60, 30)
        self.batch_help_button.clicked.connect(self.show_batch_help)

        self.epochs_label = QLabel(self)
        self.epochs_label.setText("Anzahl der Epochen:")
        self.epochs_label.setGeometry(30, 360, 140, 30)

        self.epochs_entry = QLineEdit(self)
        self.epochs_entry.setGeometry(180, 360, 100, 30)
        self.epochs_entry.setValidator(QIntValidator())  # Nur Ganzzahlen erlauben

        self.epochs_help_button = QPushButton(self)
        self.epochs_help_button.setText("Hilfe")
        self.epochs_help_button.setGeometry(300, 360, 60, 30)
        self.epochs_help_button.clicked.connect(self.show_epochs_help)

        self.optimizer_label = QLabel(self)
        self.optimizer_label.setText("Optimizer:")
        self.optimizer_label.setGeometry(30, 400, 100, 30)

        self.optimizer_combo = QComboBox(self)
        self.optimizer_combo.setGeometry(150, 400, 150, 30)
        optimizers = [name for name in dir(tf.keras.optimizers) if not name.startswith("_")]
        self.optimizer_combo.addItems(optimizers)

        self.optimizer_help_button = QPushButton(self)
        self.optimizer_help_button.setText("Hilfe")
        self.optimizer_help_button.setGeometry(310, 400, 60, 30)
        self.optimizer_help_button.clicked.connect(self.show_optimizer_help)

        self.optimizer_desc_label = QLabel(self)
        self.optimizer_desc_label.setGeometry(30, 440, 400, 60)

        self.loss_label = QLabel(self)
        self.loss_label.setText("Loss:")
        self.loss_label.setGeometry(30, 520, 100, 30)

        self.loss_combo = QComboBox(self)
        self.loss_combo.setGeometry(150, 520, 150, 30)
        losses = [name for name in dir(tf.keras.losses) if not name.startswith("_")]
        self.loss_combo.addItems(losses)

        self.loss_help_button = QPushButton(self)
        self.loss_help_button.setText("Hilfe")
        self.loss_help_button.setGeometry(310, 520, 60, 30)
        self.loss_help_button.clicked.connect(self.show_loss_help)

        self.loss_desc_label = QLabel(self)
        self.loss_desc_label.setGeometry(30, 560, 400, 60)

        self.create_button = QPushButton(self)
        self.create_button.setText("Create RNN")
        self.create_button.setGeometry(30, 600, 120, 30)
        self.create_button.clicked.connect(self.create_rnn)

        self.data_button = QPushButton(self)
        self.data_button.setText("Daten aufbereiten")
        self.data_button.setGeometry(160, 600, 150, 30)
        self.data_button.clicked.connect(self.process_data)

    def show_input_help(self):
        help_window = HelpWindow(self)
        help_window.show_input_help()

    def show_hidden_help(self):
        help_window = HelpWindow(self)
        help_window.show_hidden_help()

    def show_vocab_help(self):
        help_window = HelpWindow(self)
        help_window.show_vocab_help()

    def show_layers_help(self):
        help_window = HelpWindow(self)
        help_window.show_layers_help()

    def show_dropout_help(self):
        help_window = HelpWindow(self)
        help_window.show_dropout_help()

    def show_learning_rate_help(self):
        help_window = HelpWindow(self)
        help_window.show_learning_rate_help()

    def show_batch_help(self):
        help_window = HelpWindow(self)
        help_window.show_batch_help()

    def show_epochs_help(self):
        help_window = HelpWindow(self)
        help_window.show_epochs_help()

    def show_optimizer_help(self):
        optimizer = self.optimizer_combo.currentText()
        help_window = HelpWindow(self)
        help_window.show_optimizer_help(optimizer)

    def show_loss_help(self):
        loss = self.loss_combo.currentText()
        help_window = HelpWindow(self)
        help_window.show_loss_help(loss)

    def create_rnn(self):
        try:
            input_size = int(self.input_entry.text())
            hidden_size = int(self.hidden_entry.text())
            vocab_size = int(self.vocab_entry.text())
            num_layers = int(self.layers_entry.text())
            dropout_rate = float(self.dropout_entry.text())
            learning_rate = float(self.learning_rate_entry.text())
            batch_size = int(self.batch_entry.text())
            epochs = int(self.epochs_entry.text())
            optimizer = self.optimizer_combo.currentText()
            loss = self.loss_combo.currentText()

            # Erstellen des RNN-Modells
            model = Sequential()
            model.add(Embedding(vocab_size, input_size))
            for _ in range(num_layers):
                model.add(LSTM(hidden_size, dropout=dropout_rate, return_sequences=True))
            model.add(LSTM(hidden_size, dropout=dropout_rate))
            model.add(Dense(vocab_size, activation='softmax'))

            # Kompilieren des Modells
            loss_func = getattr(tf.keras.losses, loss)()
            optimizer_func = getattr(tf.keras.optimizers, optimizer)(learning_rate=learning_rate)
            model.compile(loss=loss_func, optimizer=optimizer_func, metrics=['accuracy'])

            # Modell speichern
            save_dialog = QFileDialog()
            save_dialog.setDefaultSuffix(".h5")
            save_path, _ = save_dialog.getSaveFileName(self, "Modell speichern", "", "H5-Dateien (*.h5)")

            if save_path:
                model.save(save_path)
                QMessageBox.information(self, "Erfolg", "RNN erfolgreich erstellt und kompiliert! Modell wurde gespeichert.")
            else:
                QMessageBox.warning(self, "Warnung", "Modell wurde nicht gespeichert.")
        except ValueError:
            QMessageBox.critical(self, "Fehler", "Bitte gib gültige Werte für die Eingabeparameter ein.")

    def process_data(self):
        preprocessing.process_data(self)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    rnn_gui = RNN_GUI()
    rnn_gui.show()
    sys.exit(app.exec_())
